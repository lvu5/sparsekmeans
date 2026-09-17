"""Extension of CuPy to handle more input formats for sparse--dense matrix multiplication.

This module adapts ``DnMatDescriptor.create`` and
``cupyx.cusparse.spmm`` from CuPy v14.2.0:

https://github.com/cupy/cupy/blob/v14.2.0/cupyx/cusparse.py

The ``_is_csr_type``, ``_is_csc_type``, and ``_is_csr_or_coo_type``
helpers are reused without modification for sparse-format checking.

Modifications made for sparsekmeans:

* ``create`` supports both C-contiguous (row-major) and F-contiguous
  (column-major) dense arrays.
* ``spmm`` supports row-major dense operands because empirically row-major storage
  provides better performance for the dense operations in the
  sparsekmeans workload.
"""

import numpy as np
import cupy as cp
import cupyx
import cupyx.scipy.sparse as csp
from cupy._core import _dtype
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupyx import cusparse as cupyx_cusparse
from cupy.cuda import device as _device
from cupyx.cusparse import check_availability, getVersion

def _is_csr_type(x):
    """True if x is any CSR sparse (array or matrix).

    Adapted from the CSR type check in ``cupyx.cusparse.spmm``.
    """
    return cupyx.scipy.sparse.issparse(x) and x.format == 'csr'


def _is_csc_type(x):
    """True if x is any CSC sparse (array or matrix).

    Adapted from the CSC type check in ``cupyx.cusparse.spmm``.
    """
    return cupyx.scipy.sparse.issparse(x) and x.format == 'csc'


def _is_csr_or_coo_type(x):
    """True if x is any CSR or COO sparse (array or matrix).

    Adapted from the CSR/COO type check in ``cupyx.cusparse.spmm``.
    """
    return cupyx.scipy.sparse.issparse(x) and x.format in ('csr', 'coo')


def create(a):
    """
    Adapted from CuPy's ``DnMatDescriptor.create``. This modified version
    also accepts C-contiguous arrays and selects the corresponding cuSPARSE
    order and leading dimension.
    """
    if a.ndim != 2:
        raise ValueError("expected 2-D matrix")

    rows, cols = a.shape
    cuda_dtype = _dtype.to_cuda_dtype(a.dtype)
    if a.flags.c_contiguous:
        order = _cusparse.CUSPARSE_ORDER_ROW
        ld = cols
    elif a.flags.f_contiguous:
        order = _cusparse.CUSPARSE_ORDER_COL
        ld = rows
    else:
        raise ValueError('expected C-contiguous array or F-contiguous array')

    desc = _cusparse.createDnMat(rows, cols, ld, a.data.ptr, cuda_dtype,
                                 order)
    get = _cusparse.dnMatGet
    destroy = _cusparse.destroyDnMat
    return cupyx_cusparse.DnMatDescriptor(desc, get, destroy)


def spmm(a, b, c=None, alpha=1, beta=0, transa=False, transb=False):
    """Multiplication of sparse matrix and dense matrix.

    Adapted from ``cupyx.cusparse.spmm``.  This version permits ``b`` and
    ``c`` to be either C-contiguous or F-contiguous, provided that they use
    the same memory layout.

    .. math::

        C = \\alpha * op(A) op(B) + \\beta * C

    Args:
        a (cupyx.scipy.sparse.csr_matrix, csc_matrix or coo_matrix):
            Sparse matrix A
        b (cupy.ndarray): Dense matrix B
        c (cupy.ndarray or None): Dense matrix C
        alpha (scalar): Coefficient
        beta (scalar): Coefficient
        transa (bool): If ``True``, op(A) = transpose of A.
        transb (bool): If ``True``, op(B) = transpose of B.

    Returns:
        cupy.ndarray
    """
    if not check_availability('spmm'):
        raise RuntimeError('spmm is not available.')

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError('expected 2-D matrices')
    if c is not None:
        if (c.flags.c_contiguous != b.flags.c_contiguous or
                c.flags.f_contiguous != b.flags.f_contiguous):
            raise ValueError('expected same F-contiguous or C-contiguous array for b and c')

    if _is_csc_type(a):
        aT = a.T
        if not _is_csr_type(aT):
            msg = 'aT must be csr_matrix (actual: {})'.format(type(aT))
            raise TypeError(msg)
        a = aT
        transa = not transa
    if not _is_csr_or_coo_type(a):
        raise TypeError(f'unsupported type (actual: {type(a)})')
    a_shape = a.shape if not transa else a.shape[::-1]
    b_shape = b.shape if not transb else b.shape[::-1]
    if a_shape[1] != b_shape[0]:
        raise ValueError('dimension mismatch')
    if not a.has_canonical_format:
        raise ValueError('expected canonical format for a')

    m, k = a_shape
    _, n = b_shape
    a, b, c = cupyx_cusparse._cast_common_type(a, b, c)
    if c is None:
        c = cp.zeros((m, n), a.dtype, 'F')
    elif c.shape[0] != m or c.shape[1] != n:
        raise ValueError('dimension mismatch')
    if a.nnz == 0:
        c.fill(0)
        return c

    _SPMM_MAX_DENSE_COLS = 65535 * 16  # 1,048,560
    if n > _SPMM_MAX_DENSE_COLS and getVersion() < 12720:
        # Workaround for cuSPARSE csrmm_alg1 gridDim.y overflow (#9850)
        for col0 in range(0, n, _SPMM_MAX_DENSE_COLS):
            col1 = min(col0 + _SPMM_MAX_DENSE_COLS, n)
            if b.flags.f_contiguous:
                if transb:
                    # Row slice of F-contiguous is not F-contiguous; copy needed
                    b_chunk = cp.asfortranarray(b[col0:col1, :])
                else:
                    # Column slice of F-contiguous is F-contiguous; no copy
                    b_chunk = b[:, col0:col1]
                c_chunk = c[:, col0:col1]
                spmm(a, b_chunk, c=c_chunk, alpha=alpha, beta=beta,
                    transa=transa, transb=transb)
            elif b.flags.c_contiguous:
                if transb:
                    # Row slice of C-contiguous is C-contiguous; no copy
                    b_chunk = b[col0:col1, :]
                else:
                    # Column slice of C-contiguous is not C-contiguous; copy needed
                    b_chunk = cp.ascontiguousarray(b[:, col0:col1])

                # Column slice of C-contiguous C is not contiguous; copy needed
                # Compute into a contiguous temporary array, then copy it back
                c_chunk = cp.ascontiguousarray(c[:, col0:col1])
                spmm(a, b_chunk, c=c_chunk, alpha=alpha, beta=beta,
                    transa=transa, transb=transb)

                # c_chunk is a contiguous copy, so copy the result back to c
                c[:, col0:col1] = c_chunk
            else:
                raise ValueError('expected C-contiguous array or F-contiguous array for b')
        return c

    desc_a = cupyx_cusparse.SpMatDescriptor.create(a)
    desc_b = create(b)
    desc_c = create(c)

    handle = _device.get_cusparse_handle()
    op_a = cupyx_cusparse._transpose_flag(transa)
    op_b = cupyx_cusparse._transpose_flag(transb)
    alpha = np.array(alpha, a.dtype).ctypes
    beta = np.array(beta, a.dtype).ctypes
    cuda_dtype = _dtype.to_cuda_dtype(a.dtype)
    alg = _cusparse.CUSPARSE_MM_ALG_DEFAULT
    buff_size = _cusparse.spMM_bufferSize(handle, op_a, op_b, alpha.data,
                                          desc_a.desc, desc_b.desc, beta.data,
                                          desc_c.desc, cuda_dtype, alg)
    buff = cp.empty(buff_size, cp.int8)
    buff_size = _cusparse.spMM(handle, op_a, op_b, alpha.data, desc_a.desc,
                               desc_b.desc, beta.data, desc_c.desc,
                               cuda_dtype, alg, buff.data.ptr)

    return c