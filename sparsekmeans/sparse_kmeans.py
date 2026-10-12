from __future__ import annotations
import os
import sys
import numpy as np
import scipy.sparse as sparse
import time
import warnings
from typing import Union

import graphblas as gb
from graphblas import Matrix, Vector, dtypes

import gc

default_threads = max(1, os.cpu_count() // 2)

def _load_gpu_backend():
    try:
        import cupy as cp
        import cupyx
        import cupyx.scipy.sparse as csp
        from .cupy_utils import spmm
    except ImportError as exc:
        raise ImportError(
            "GPU support requires cupy and CUDA Toolkit."
        ) from exc

    return cp, cupyx, csp, spmm

def _scipy_csr_to_cupy_csr(X: sparse.csr_matrix):
    _, _, csp, _ = _load_gpu_backend()
    # Sum duplicate CSR entries on CPU before moving to CuPy. On GPU, handling duplicated entries causes extra GPU memory usage and slower matrix multiplication.
    X.sum_duplicates()
    X = csp.csr_matrix(X)
    return X

def check_random_state(seed: Union[int, None]):
    # Generate a global random number generator (global RNG) - a RandomState instance
    if seed is None or seed is np.random:
        # Return np.random.RandomState() (variable _rand = np.random.RandomState(), the seed is chosen automatically by system)
        return np.random.mtrand._rand
    if isinstance(seed, int):
        # Returns a RandomState object seeded with given integer
        return np.random.RandomState(seed)

def _labels_dtype(n_clusters: int):
    # Use the smallest integer type that can store all cluster indices
    return np.int16 if n_clusters <= np.iinfo(np.int16).max + 1 else np.int32


def squared_row_norms(X: gb.Matrix, n_threads: int=default_threads):
    # Compute sum_j X_ij^2 by multiplying X with a vector of 2s under the plus_pow semiring.
    # The vector is iso-valued and takes O(1) memory, so no squared copy of X is created.
    twos = Vector.from_scalar(2.0, size=X.shape[1], dtype=dtypes.FP64)

    sq_row_norms = Vector(dtypes.FP64, X.shape[0])
    sq_row_norms(nthreads=n_threads) << X.mxv(twos, gb.semiring.plus_pow)

    return sq_row_norms.to_dense(fill_value=0)


def check_centroids_density(centroids: gb.Matrix):
    gamma = 0.05

    if centroids.ss.format == "fullr":
        is_centroid_dense = True
        centroids_density = centroids.reduce_scalar(gb.agg.count_nonzero).new().value / (centroids.nrows * centroids.ncols)
        if centroids_density <= gamma:
            is_centroid_dense = False
            centroids(mask=centroids.V, replace=True) << centroids
    else:
        is_centroid_dense = False
        centroids_density = centroids.nvals / (centroids.nrows * centroids.ncols)
        if centroids_density > gamma:
            is_centroid_dense = True
            centroids = _to_full(centroids)

    return centroids, is_centroid_dense


def _to_full(A: gb.Matrix, n_threads: int=default_threads):
    # Return a new matrix storing A in the full format, with zeros for the missing entries.
    # Assigning zeros with the mask ~A.S instead keeps A in a sparse format that stores all
    # entries together with their indices, which takes much more memory.
    A_full = Matrix(dtypes.FP64, nrows=A.nrows, ncols=A.ncols)
    A_full << 0
    A_full(accum=gb.binary.plus, nthreads=n_threads) << A

    return A_full


def predict_labels(X: Union[gb.Matrix, csp.csr_matrix], centroids: gb.Matrix, is_centroid_dense: bool, n_threads: int=default_threads, gpu: bool=False):
    """Return the closest centroid of each sample and the corresponding squared distance
    (up to the omitted constant ||x||^2, which does not affect the argmin).
    """
    # Force all data to FP64 to avoid data type casting during matrix multiplication, which may degrade performance.
    if gpu:
        cp, cupyx, csp, spmm = _load_gpu_backend()
        if isinstance(X, gb.Matrix):
            X = sparse.csr_matrix(gb.io.to_scipy_sparse(X, format="csr"), dtype=np.float64)
            X = _scipy_csr_to_cupy_csr(X)
    else:
        if X.dtype != dtypes.FP64:
            X = X.dup(dtype="FP64")
        if centroids.dtype != dtypes.FP64:
            centroids = centroids.dup(dtype="FP64")
    n_samples = X.shape[0]
    n_clusters = centroids.shape[0]

    # x_squared_norms is not needed
    c_squared_norms = squared_row_norms(centroids, n_threads)
    if gpu:
        # Initialize the output with the centroid norms directly on the device and reduce it there,
        # so no (n_samples x n_clusters) array is allocated or transferred on the host.
        distances_to_centroids = cp.empty((n_samples, n_clusters), dtype=cp.float64, order="C")
        distances_to_centroids[...] = cp.asarray(c_squared_norms, dtype=cp.float64)
        centroids = cp.asarray(centroids.to_dense(fill_value=0), dtype=cp.float64, order="C")
        centroids = cp.asfortranarray(centroids)
        distances_to_centroids = spmm(X, centroids.T, distances_to_centroids, -2, 1)
        del centroids

        labels = cp.argmin(distances_to_centroids, axis=1).get()
        closest_distances = cp.min(distances_to_centroids, axis=1).get()

    else:
        # For X * Ct, we use CSR (for X format) x (fullr or CSR for Ct format) for the best efficiency.
        # We use row format to store centroids in other places, so must materialize Ct here. This makes a
        # single copy, which is cheaper than converting a copy of the centroids to column format.
        Ct = Matrix(dtypes.FP64, nrows=centroids.ncols, ncols=n_clusters)
        Ct(nthreads=n_threads) << centroids.T

        labels, closest_distances = _closest_centroids(X, Ct, c_squared_norms, n_threads)

        # GraphBLAS objects are in reference cycles and may be freed only by the garbage collector,
        # so explicitly release the memory of large temporaries.
        Ct.clear()

    return labels, closest_distances


# Minimum upper bound on the size of the dense block of XCt computed at once. Processing samples in blocks
# bounds the memory of distance computation, but each block must be copied out of X. The bound is therefore
# raised to the size of X, so blocks are used only when XCt would take more memory than X itself.
_DISTANCE_BLOCK_BYTES = 256 * 2**20


def _closest_centroids(X: gb.Matrix, Ct: gb.Matrix, c_squared_norms: np.ndarray, n_threads: int=default_threads):
    n_samples = X.shape[0]
    n_clusters = Ct.shape[1]

    labels = np.empty(n_samples, dtype=np.intp)
    closest_distances = np.empty(n_samples, dtype=np.float64)

    block_bytes = max(_DISTANCE_BLOCK_BYTES, X.ss.nbytes)
    block_size = max(n_threads, block_bytes // (8 * n_clusters))
    for start in range(0, n_samples, block_size):
        end = min(start + block_size, n_samples)
        is_block = not (start == 0 and end == n_samples)
        if is_block:
            X_block = Matrix(dtypes.FP64, nrows=end - start, ncols=X.shape[1])
            X_block(nthreads=n_threads) << X[start:end, :]
        else:
            X_block = X

        XCt = Matrix(dtypes.FP64, nrows=end - start, ncols=n_clusters)
        XCt << 0
        XCt(accum=gb.binary.plus, nthreads=n_threads) << X_block.mxm(Ct)
        if is_block:
            # Explicitly release the memory as GraphBLAS objects may be freed only by the garbage collector
            X_block.clear()

        # Take over the dense buffer of XCt without a copy and compute -2 * XCt + ||c||^2 in place.
        # x_squared_norms is not needed
        if XCt.ss.is_iso:
            distances_to_centroids = XCt.to_dense()
        else:
            distances_to_centroids = XCt.ss.export("fullr", give_ownership=True)["values"]
        distances_to_centroids *= -2
        distances_to_centroids += c_squared_norms[np.newaxis, :]

        np.argmin(distances_to_centroids, axis=1, out=labels[start:end])
        np.min(distances_to_centroids, axis=1, out=closest_distances[start:end])

    return labels, closest_distances


def kmeans_predict(X: Union[sparse.csr_matrix, gb.Matrix], centroids: Union[sparse.csr_matrix, gb.Matrix], n_threads: int=default_threads, use_gpu: bool=False) -> np.array:
    """Predict cluster for each sample in X given centroids

    Parameters
    ----------
    X : `gb.Matrix (csr format), scipy.sparse.csr_matrix` 
        Input dataset for clustering
    centroids : `gb.Matrix (csr format), scipy.sparse.csr_matrix` 
        Input centroids
    n_threads : int
        Number of using threads

    use_gpu : `bool`, default=False
        Whether to use GPU for matrix multiplication.

    Returns
    -------
    labels: `np.array`
        Array storing the assigned cluster for each sample.
    """
    if not isinstance(X, gb.Matrix):
        X = gb.io.from_scipy_sparse(X)

    if not isinstance(centroids, gb.Matrix):
        centroids = gb.io.from_scipy_sparse(centroids)

    centroids, is_centroid_dense = check_centroids_density(centroids)

    labels, _ = predict_labels(X, centroids, is_centroid_dense, n_threads, use_gpu)

    return labels


class SparseKmeans:
    def __init__(
        self,
        n_clusters: int = 8,
        n_threads: int = default_threads,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Union[
            np.random.randint, np.random.RandomState, int, None
        ] = None,
        verbose: bool = False,
        use_gpu: bool = False
    ):
        """Sparse K-means clustering.

        Parameters
        ----------
        n_clusters : int, default=8
            The predefined number of clusters

        n_threads : int, default=max(1, os.cpu_count() // 2)
            The predefined number of threads to use 

        max_iter : int, default=300
            Maximum number of iterations of the k-means algorithm.

        tol : float, default=1e-4
            Relative tolerance to declare convergence.

        random_state : int, RandomState instance or None, default=None
            Determines random number generation for centroid initialization.

        use_gpu : `bool`, default=False
            Whether to use GPU for matrix multiplication.

        Attributes
        ----------
        centroids : `gb.Matrix`
            Centroids of each cluster derived from training on input matrix X.

        labels : `np.array`
            Cluster for each sample for input matrix X

        is_fitted: `bool`
            Boolean flag checking whether the model is fitted on datasets or not
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = check_random_state(random_state)
        self.n_threads = n_threads
        self.verbose = verbose
        self.use_gpu = use_gpu

        # Flag to indicate if centroids stored in dense format
        self.is_centroid_dense = False

        self.is_fitted = False

    def _initialize_centroids(self, X: gb.Matrix):

        # The kmeans++ initialization method by Arthur and Vassilvitskii, 2007

        n_samples, n_features = X.shape
        n_clusters = self.n_clusters
        x_squared_norms = self.x_squared_norms

        # Record the selected samples and extract them at once after the selection
        centroid_ids = np.empty(n_clusters, dtype=np.int64)

        # We can simply use self.random_state.choice(n_samples), but instead follow scikit-learn to allow
        # weights for the selection. Also, the two ways may give different results.
        first_centroid_id = self.random_state.choice(n_samples, p=np.full(n_samples, 1 / n_samples))

        centroid_ids[0] = first_centroid_id

        Xc = Vector(dtype=dtypes.FP64, size=n_samples)
        Xc(nthreads=self.n_threads) << X.mxv(X[first_centroid_id, :])
        smallest_sq_dist = (
            x_squared_norms
            - 2 * Xc.to_dense(fill_value=0)
            + x_squared_norms[first_centroid_id]
        )
        # Maintain the potential, sum of each sample's smallest squared distance to existing centroids
        current_potential = smallest_sq_dist.sum()

        n_local_trials = 2 + int(np.log(n_clusters))

        for j in range(1, n_clusters):

            if current_potential < 1e-12:
                warnings.warn(f"The specified number of clusters is larger than the number of distinct points.\nChange the number of clusters to the number of distinct points.", category=UserWarning)
                centroid_ids = centroid_ids[:j]
                self.n_clusters = j
                break

            # Choose centroids candidates by sampling with probability smallest_sq_dist/current_potential, where
            # current_potential is the sum of smallest_sq_dist
            rand_vals = self.random_state.uniform(size=n_local_trials) * current_potential

            candidate_ids = np.searchsorted(np.cumsum(smallest_sq_dist), rand_vals)

            # The product between random values in [0, 1] and current_potential may numerically result in
            # rand_vals > current_potential. Then candidate_ids may be outside the desired range
            candidate_ids = np.minimum(candidate_ids, smallest_sq_dist.size - 1)

            # Compute squared distances to centroid candidates
            if self.use_gpu:
                cp, cupyx, _, _ = _load_gpu_backend()
                Xcandidates_t = cupyx.empty_pinned((X.shape[0], n_local_trials), dtype=self.gpu_X.dtype, order="C")
                candidates = self.gpu_X[candidate_ids]
                candidates = candidates.toarray()
                Xcandidates_t_device = self.gpu_X @ candidates.T
                cp.cuda.Stream.null.synchronize()
                Xcandidates_t_device.get(out=Xcandidates_t, blocking=True)

                sq_distance_to_candidates = (
                    x_squared_norms[:, np.newaxis]
                    - 2 * Xcandidates_t
                    + x_squared_norms[candidate_ids][np.newaxis, :]
                )

            else:
                candidates = Matrix(dtype=dtypes.FP64, nrows=n_local_trials, ncols=n_features)
                candidates << X[candidate_ids, :]
                candidates = candidates.ss.export("csc")
                candidates = gb.Matrix.ss.import_csc(**candidates)

                Xcandidates_t = Matrix(dtype=dtypes.FP64, nrows=n_samples, ncols=n_local_trials)
                Xcandidates_t(nthreads=self.n_threads) << X.mxm(candidates.T)  # CSR * CSR

                sq_distance_to_candidates = (
                    x_squared_norms[:, np.newaxis]
                    - 2 * Xcandidates_t.to_dense(fill_value=0)
                    + x_squared_norms[candidate_ids][np.newaxis, :]
                )
                # Explicitly release the memory as GraphBLAS objects may be freed only by the garbage collector
                Xcandidates_t.clear()

            sq_distance_to_candidates = sq_distance_to_candidates.T

            # Find the smallest squared distance from data to candidtaes
            sq_distance_to_candidates = np.minimum(
                smallest_sq_dist, sq_distance_to_candidates
            )
            candidates_potential = sq_distance_to_candidates.sum(axis=1)

            # Decide which candidate is the best
            best_candidate = np.argmin(candidates_potential)
            current_potential = candidates_potential[best_candidate]
            smallest_sq_dist = sq_distance_to_candidates[best_candidate]

            centroid_ids[j] = candidate_ids[best_candidate]

        centroids = Matrix(dtype=dtypes.FP64, nrows=len(centroid_ids), ncols=n_features)
        centroids(nthreads=self.n_threads) << X[centroid_ids, :]

        return centroids

    def _single_kmeans(self, X: gb.Matrix):
        # Allowing the kmeans procedure to allocate and initialize internal variables
        self._setup_internal_state(X)

        if self.mean_feature_variance <= sys.float_info.min:
            return

        for iter in range(self.max_iter):
            start_iter = time.time()

            # Decide to store centroids in dense/sparse format based on density
            centroids, self.is_centroid_dense = check_centroids_density(self.centroids)
            if centroids is not self.centroids:
                # Explicitly release the memory as GraphBLAS objects may be freed only by the garbage collector
                self.centroids.clear()
                self.centroids = centroids
            del centroids

            if self.use_gpu:
                self._assign_cluster(self.gpu_X)
            else:
                self._assign_cluster(X)

            if iter > 0:
                if self._converged():
                    break

            old_centroids = self.centroids
            self.centroids = self._update_centroids(X)

            # The old centroids are not needed afterward, so the shift is computed in place on them
            self.centroids_shift = self._cal_centroids_shift(old_centroids)
            # Explicitly release the memory as GraphBLAS objects may be freed only by the garbage collector
            old_centroids.clear()
            del old_centroids

            # Allowing the kmeans procedure to update internal variables
            self._update_internal_state()

            end_iter = time.time()
            if self.verbose:
                print(
                    f"Time to conduct iteration: {iter}", end_iter - start_iter, flush=True
                )

        self.is_fitted = True

        return

    def fit(self, X: Union[sparse.csr_matrix, gb.Matrix]):
        """Conducting K-means clustering.

        Parameters
        ----------
        X : `gb.Matrix, scipy.sparse.csr_matrix` (csr format)
            Input dataset for clustering

        Returns
        -------.
        """

        n_samples = X.shape[0]
        if self.verbose:
            print("Total samples: ", n_samples)
        # Force all data to FP64 to avoid data type casting during matrix multiplication, which may degrade performance.
        if self.use_gpu:
            _, _, csp, _ = _load_gpu_backend()
            if isinstance(X, gb.Matrix):
                X_scipy_csr = sparse.csr_matrix(gb.io.to_scipy_sparse(X, format="csr"), dtype=np.float64)
            else:
                X_scipy_csr = sparse.csr_matrix(X, dtype=np.float64)
            self.gpu_X = _scipy_csr_to_cupy_csr(X_scipy_csr)
            # The host copy is no longer needed after being moved to the device
            del X_scipy_csr
        if not isinstance(X, gb.Matrix):
            X = gb.io.from_scipy_sparse(X)
        if X.dtype != dtypes.FP64:
            X = X.dup(dtype="FP64")

        # Squared norms of samples are shared by the initialization, the feature variance, and Elkan's method
        self.x_squared_norms = squared_row_norms(X, self.n_threads)

        start_init_centroids = time.time()
        self.centroids = self._initialize_centroids(X)
        end_init_centroids = time.time()

        if self.verbose:
            print("Initialize Centroids time: ", end_init_centroids - start_init_centroids)

        self._single_kmeans(X)

        self._cleanup()

        return self.labels

    def predict(self, X: Union[sparse.csr_matrix, gb.Matrix], use_gpu: bool = False):
        """Predict the closest cluster for each sample in X

        Parameters
        ----------
        X : `gb.Matrix, scipy.sparse.csr_matrix` (csr format)
            Input dataset for predicting label

        use_gpu : `bool`, default=False
            Whether to use GPU for matrix multiplication.

        Returns
        -------
        labels: `np.array`
            Array storing the assigned cluster for each sample.
        """

        if not self.is_fitted:
            raise AttributeError(
            f"This instance of Kmeans is not trained yet. "
            f"Call 'fit' before using this method."
        )

        if not isinstance(X, gb.Matrix):
            X = gb.io.from_scipy_sparse(X)

        labels, _ = predict_labels(X, self.centroids, self.is_centroid_dense, self.n_threads, use_gpu)

        return labels

    def _assign_cluster(self, *args, **kargs):
        pass

    def _update_centroids(self, X: gb.Matrix):
        n_samples, n_features = X.shape
        n_clusters = self.n_clusters

        cluster_sizes = np.bincount(self.labels, minlength=n_clusters)

        empty_clusters = np.where(cluster_sizes == 0)[0]
        n_empty = len(empty_clusters)

        # To handle empty clusters, we consider points with the largest distances to their assigned clusters, and reassign them to the empty clusters
        # If a selected point belongs to a cluster with only one point, we skip it and move on to the next furthest point.
        # Each empty cluster gets one point
        # We also need to update cluster_sizes
        # See Algorithm E.1 in supplementary materials of Dang et al., CIKM 2025.
        if n_empty > 0:
            far_samples_idx = np.argsort(self.sample_centroids_closest_distance)[:: -1]
            k, j = 0, 0
            
            while k < n_samples - n_empty and j < n_empty:
                origin_cluster = self.labels[far_samples_idx[k]]

                if cluster_sizes[origin_cluster] > 1:
                    target_empty_cluster = empty_clusters[j]
                    cluster_sizes[origin_cluster] -= 1
                    cluster_sizes[target_empty_cluster] = 1
                    self.labels[far_samples_idx[k]] = target_empty_cluster
                    j += 1
                k +=1
            
            if k == n_samples - n_empty:
                raise ValueError("Not enough distinct points to assign to empty clusters.")

        # Get the weight of each sample in its corresponding cluster
        weights = 1 / cluster_sizes
        weights = weights[self.labels]

        weighted_matrix = Matrix.from_coo(
            self.labels, np.arange(len(self.labels)), weights, nrows=n_clusters, ncols=n_samples
        )
        centroids = Matrix(dtypes.FP64, nrows=n_clusters, ncols=n_features)

        if self.is_centroid_dense:
            centroids << 0
            centroids(accum=gb.binary.plus, nthreads=self.n_threads) << weighted_matrix.mxm(X)
        else:
            centroids(nthreads=self.n_threads) << weighted_matrix.mxm(X)

        return centroids

    def _setup_internal_state(self, X: gb.Matrix):
        n_samples = X.shape[0]

        self.labels = np.zeros(n_samples, dtype=_labels_dtype(self.n_clusters))

        # Maintain the distance between each sample and the assigned centroid
        # We need the distances in the function update_centroids for handling empty clusters
        self.sample_centroids_closest_distance = np.zeros(n_samples, dtype=np.float64)

        self.mean_feature_variance = self._cal_feature_variance(X)

        return

    def _cal_centroids_shift(self, old_centroids: gb.Matrix):
        # Overwrite old_centroids by old_centroids - centroids to avoid allocating another matrix.
        # Entries only in centroids keep their sign, which does not affect the squared norms.
        old_centroids(accum=gb.binary.minus, nthreads=self.n_threads) << self.centroids
        centroids_shift_squared = squared_row_norms(old_centroids, self.n_threads)

        centroids_shift = np.sqrt(centroids_shift_squared)

        return centroids_shift

    def _cal_feature_variance(self, X: gb.Matrix):
        n_samples, n_features = X.shape

        # The mean of the feature variances is (sum_j sum_i X_ij^2 - n_samples * sum_j means_j^2) / (n_samples * n_features).
        # The first sum is the sum of the squared row norms, and only nonzero column sums contribute to the second,
        # so no dense array of length n_features is needed.
        column_sums = Vector(dtypes.FP64, n_features)
        column_sums(nthreads=self.n_threads) << X.reduce_columnwise(op="add")
        _, column_sums = column_sums.to_coo(indices=False)
        means = column_sums / n_samples

        return (self.x_squared_norms.sum() - n_samples * (means ** 2).sum()) / (n_samples * n_features)

    def _converged(self):
        pass

    def _update_internal_state(self, *args, **kargs):
        pass

    def _cleanup(self, keep=['centroids', 'labels', 'is_fitted', 'is_centroid_dense', 'n_threads', 'verbose', 'random_state', 'n_clusters', 'max_iter', 'tol', 'use_gpu']):
        for attr in list(self.__dict__.keys()):
            if attr not in keep:
                delattr(self, attr)


class LloydKmeans(SparseKmeans):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _assign_cluster(self, X: Union[gb.Matrix, csp.csr_matrix]):
        self.labels, self.sample_centroids_closest_distance = predict_labels(X, self.centroids, self.is_centroid_dense, self.n_threads, gpu=self.use_gpu)

        # Preventing memory issue with mxm operation
        gc.collect()

        return

    def _converged(self):

        tol = (self.centroids_shift**2).sum()

        return tol <= self.mean_feature_variance * self.tol


class ElkanKmeans(SparseKmeans):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.use_gpu:
            raise ValueError(
                "GPU support is currently available only for LloydKmeans."
            )

    def _assign_cluster(self, X: gb.Matrix):
        n_samples = X.shape[0]
        n_clusters = self.centroids.shape[0]

        c_squared_norms = squared_row_norms(self.centroids, self.n_threads)

        if self.centroids.ss.format == "fullr" and not self.centroids.ss.is_iso:
            # For dense centroids, GraphBLAS computes n_clusters^2 independent dot products of long rows, which is
            # bounded by memory bandwidth. Instead, apply cache-efficient BLAS to the buffer of the centroids,
            # which is temporarily taken from GraphBLAS without a copy.
            centroids_values = self.centroids.ss.unpack("fullr")["values"]
            try:
                CCt = centroids_values @ centroids_values.T
            finally:
                self.centroids.ss.pack_fullr(centroids_values, take_ownership=True)
        else:
            CCt = Matrix(dtypes.FP64, nrows=n_clusters, ncols=n_clusters)
            CCt(nthreads=self.n_threads) << self.centroids.mxm(self.centroids.T)
            CCt = CCt.to_dense(fill_value=0)

        half_centroid_centroid_distances = c_squared_norms[:, np.newaxis] - 2 * CCt + c_squared_norms[np.newaxis, :]
        np.clip(half_centroid_centroid_distances, 0, None, out=half_centroid_centroid_distances)
        half_centroid_centroid_distances = 0.5 * np.sqrt(half_centroid_centroid_distances)

        # We may do row-wise products between samples (i.e., X) and centroids[labels,:], but the expansion of
        # centroids is time/memory inefficient. Instead, we use a matrix-matrix product with mask.
        samples_centroids_product = Matrix(dtypes.FP64, nrows=n_samples, ncols=n_clusters)
        samples_centroids_mask = Matrix.from_coo(
            np.arange(n_samples), self.labels, values=1, nrows=n_samples, ncols=n_clusters
        )

        # Always use dense centroids for calculating samples_centroids_product here and Xcj in the loop
        if self.centroids.ss.format != "fullr":
            centroids = _to_full(self.centroids, self.n_threads)
            # Explicitly release the memory as GraphBLAS objects may be freed only by the garbage collector
            self.centroids.clear()
            self.centroids = centroids

        samples_centroids_product(mask=samples_centroids_mask.S, nthreads=self.n_threads) << X.mxm(self.centroids.T)
        samples_centroids_product = samples_centroids_product.reduce_rowwise(gb.binary.min).to_dense(fill_value=0)

        self.sample_centroids_closest_distance = self.x_squared_norms - 2 * samples_centroids_product + c_squared_norms[self.labels]
        np.clip(self.sample_centroids_closest_distance, 0, None, out=self.sample_centroids_closest_distance)
        self.sample_centroids_closest_distance = np.sqrt(self.sample_centroids_closest_distance)

        for j in range(n_clusters):

            # Apply two conditions first to get a subset as half_centroid_centroid_distances[j, labels] expensively expand a small array to a larger one
            candidate_idx = np.flatnonzero((self.labels != j) & (self.sample_centroids_closest_distance > self.lower_bounds[j, :]))
            candidate_idx = candidate_idx[self.sample_centroids_closest_distance[candidate_idx] > half_centroid_centroid_distances[j, self.labels[candidate_idx]]]

            if len(candidate_idx) == 0:
                continue

            # Start from explicit zeros at the candidates so that the result has exactly one value per candidate,
            # in the order of candidate_idx, without expanding it to a dense vector of length n_samples
            Xcj = Vector.from_coo(candidate_idx, 0.0, size=n_samples, dtype=dtypes.FP64)
            Xcj_mask = Vector.from_coo(candidate_idx, 1, size=n_samples)

            # Update lower bounds according to new centroids
            Xcj(mask=Xcj_mask.S, accum=gb.binary.plus, nthreads=self.n_threads) << X.mxv(self.centroids[j, :])
            _, Xcj = Xcj.to_coo(indices=False)
            distance_to_cj = self.x_squared_norms[candidate_idx] - 2 * Xcj + c_squared_norms[j]
            np.clip(distance_to_cj, 0, None, out=distance_to_cj)
            distance_to_cj = np.sqrt(distance_to_cj)
            self.lower_bounds[j, candidate_idx] = distance_to_cj

            reassignment_mask = self.sample_centroids_closest_distance[candidate_idx] > self.lower_bounds[j, candidate_idx]

            samples_to_reassign_idx = candidate_idx[reassignment_mask]

            if len(samples_to_reassign_idx) == 0:
                continue

            self.labels[samples_to_reassign_idx] = j
            update_distances = distance_to_cj[reassignment_mask]
            self.sample_centroids_closest_distance[samples_to_reassign_idx] = update_distances

        # Preventing memory issue with mxm operation
        gc.collect()

        return

    def _setup_internal_state(self, X: gb.Matrix):
        super()._setup_internal_state(X)

        # self.x_squared_norms is computed in fit
        self.lower_bounds = np.zeros((self.n_clusters, X.shape[0]), dtype=np.float64)

        return

    def _update_internal_state(self):
        self.lower_bounds -= self.centroids_shift[:, np.newaxis]
        np.maximum(self.lower_bounds, 0, self.lower_bounds)

        return

    def _converged(self):
        tol = (self.centroids_shift**2).sum()

        return tol <= self.mean_feature_variance * self.tol
