# sparsekmeans: memory and running-time optimization

## Summary

On the README benchmark datasets (Amazon-670K and Url, K=100), the new version is faster in every configuration (6–43%) and uses 42–58% less peak memory, with identical clustering results (same inertia to all printed digits).

## Results

Setup: 16 threads, `random_state=0`, `max_iter=300`, `tol=1e-4`, one run per configuration. Peak memory is the process's resident-memory high-water mark during `fit`, minus the memory in use before `fit`. Inertia is recomputed independently with scipy from the returned labels and centroids.

### README datasets, K=100

| Dataset | Method | Original | New | Time | Peak memory | Inertia |
|---|---|---|---|---|---|---|
| Amazon-670K (490,449 × 135,876, 37M nnz) | Lloyd | 200.8 s, 4.0 GB | 153.8 s, 1.9 GB | −23% | −54% | identical |
| Amazon-670K | Elkan | 223.1 s, 3.5 GB | 197.5 s, 1.6 GB | −11% | −56% | identical |
| Url (2,396,130 × 3,230,442, 277M nnz) | Lloyd | 525.2 s, 18.4 GB | 492.0 s, 10.8 GB | −6% | −42% | identical |
| Url | Elkan | 733.5 s, 29.7 GB | 420.9 s, 12.6 GB | −43% | −58% | identical |

Amazon-670K is the LIBSVM multi-label training set. Url is LIBSVM's `url_combined_normalized`; the README does not say which Url variant it used. Empty feature columns were dropped from both.

### Additional datasets, K=100

These are from EURLEX-4K. "Label representations" are built the way LibMultiLabel builds its label tree (Yᵀ·X, L2-normalized), which is the main downstream use of this package.

| Dataset | Method | Original | New | Inertia |
|---|---|---|---|---|
| EURLEX-4K label representations (3,801 × 3,065,363) | Lloyd | 194.8 s, 20.2 GB | 130.5 s, 5.9 GB | identical |
| EURLEX-4K label representations | Elkan | 260.0 s, 17.7 GB | 129.3 s, 6.4 GB | identical |
| EURLEX-4K instances (15,449 × 3,065,363) | Lloyd | 37.7 s, 2.1 GB | 29.5 s, 2.2 GB | identical |
| EURLEX-4K instances | Elkan | 241.2 s, 19.1 GB | 83.6 s, 4.1 GB | identical |

The largest gains are where centroids become dense, which is typical of very high-dimensional data such as these.

## What changed

### Memory

- **Distances reduced block by block.** `predict_labels` no longer keeps the full n×K distance matrix plus several temporary copies of it. Each block of distances is reduced right away to the closest centroid and its distance; its buffer is taken from GraphBLAS without a copy and updated in place. Samples are split into row blocks only when the n×K distances would take more memory than X itself (with a 256 MB minimum), so distance memory never exceeds the size of the data. Each block has to be copied out of X, and splitting unconditionally into 256 MB blocks made Url with Lloyd's method 5% slower than the original (551.2 s vs 525.2 s, 6.9 GB peak). With the size-relative rule, Url at K=100 is not split, and it splits into 3 and 5 blocks at K=500 and K=1000.
- **True dense format for dense centroids.** The original densified centroids with `centroids(mask=~centroids.S) << 0`. This leaves GraphBLAS in a sparse format that stores every entry together with its column index, about 3.3× the memory of the dense format. It also forces the later products onto sparse kernels. Densifying now builds a dense-format matrix directly, in `_to_full`. This affects the density check and Elkan's "always dense" step.
- **One copy for the transposed centroids.** The export/import round trip used to convert centroids to column format made up to four copies. Materializing `Cᵀ` directly makes one copy, and the following product is faster.
- **No squared copy of X.** `squared_row_norms` now multiplies by a constant vector of 2s under the `plus_pow` semiring. That vector takes O(1) memory, so `X ∘ X` is never materialized.
- **Centroid shift computed in place.** The old centroids are no longer needed after the update, so `old − new` overwrites them instead of allocating another K×d matrix.
- **Cheaper density count.** `count_nonzero` replaces `centroids.V.new()`, which built a K×d boolean matrix.
- **Feature variance without dense length-d arrays.** It is now computed from the sum of squared row norms and the sparse column sums. Before, it used dense length-d arrays plus a squared copy of X.
- **Explicit release of large temporaries.** In this python-graphblas version, Matrix and Vector objects are in reference cycles, so `del` does not free them; only the garbage collector does. Large temporaries (transposed centroids, row blocks of X, replaced centroids) are now freed with `.clear()` as soon as they are no longer needed.
- **GPU path.** The argmin and min are now computed on the device, and only n labels and n distances are copied back. Before, two n×K host arrays were allocated and the full n×K matrix was transferred. The host copy of X is also released once it has been moved to the device.

### Time

- **BLAS for C·Cᵀ in Elkan.** For dense centroids, GraphBLAS computed C·Cᵀ as K² separate long dot products, which is limited by memory bandwidth (about 8 s per iteration on EURLEX labels). It now uses BLAS on the centroid buffer, which is borrowed from GraphBLAS without a copy.
- **Elkan inner loop.** The loop no longer expands each centroid's products into a dense length-n vector. It reads values only at the candidate samples.
- **Shared intermediate results.** Squared row norms of X are computed once and shared by the initialization, the feature variance and Elkan's method; before, they were computed twice. Elkan's feature variance was also computed twice and is now computed once.
- **Initialization.** k-means++ records the chosen sample ids and extracts all centroids at once, instead of assigning one row at a time. The uniform probability vector is a NumPy array instead of a Python list of n floats.

### Bug fixes

- **Crash when K exceeds the number of distinct points.** The original raised `InvalidValue` in this case. It now reduces K with a warning, as the code intended.
- **Elkan labels overflow for K > 32,768.** Labels were stored as int16. They now use int16 when K ≤ 32,768 and int32 otherwise.

### Behaviour changes

- `predict_labels` now returns `(labels, closest_distances)` instead of `(labels, full n×K distance matrix)`. It is reachable through `from sparsekmeans import *`, but none of the package's documented functions use its second return value.
- The "Total samples" line in `fit` now prints only when `verbose=True`.
- `kmeans_predict` no longer modifies a user-supplied sparse centroid matrix in place when it converts it to dense.
- BLAS's C·Cᵀ in Elkan uses the BLAS library's own thread pool, not `n_threads`.

## Verification

- **Bit-exactness against the original.** Two changes alter floating-point summation order: the squared row norms and the feature variance formula. With those two temporarily swapped back to the original versions, every other change was checked against the original code. The check covered random sparse data, random dense data and an EURLEX subset, with Lloyd and Elkan, several values of K, 3 seeds each, and both single-block and forced multi-block distance computation. Labels and `predict` output were bit-identical in every case. Centroids differed only at about 1e-18, which the original code also shows between two runs of itself, because SuiteSparse's parallel summation order is not deterministic.
- **Effect of the two summation-order changes.** These two changes alter values by at most about 3e-14 relative. Inertia was still identical in every benchmark above.
- **GPU path.** This machine has no GPU, so the GPU code was run against a NumPy stand-in for CuPy. Labels from `fit`, `predict(use_gpu=True)` and `kmeans_predict(use_gpu=True)` matched the original in all tested cases and agreed with the CPU path. It still needs a run on real GPU hardware.

## Open items

- **K=500 not yet run.** The README also reports K=500 for both datasets; those runs were deferred.
- **GPU path untested on real hardware**, as above.
- **Single runs.** Each configuration was run once on a shared machine, so differences of a few percent are within noise.
- **Different hardware from the README.** These benchmarks ran on 2× AMD EPYC 7401 (Zen 1, 2.0 GHz, 8 NUMA nodes). The README's numbers come from an Intel Core i7-6900K at 3.2 GHz. The original code here is about 1.6× slower than the README on Amazon-670K. Only the original-vs-new comparisons on this machine are meaningful.
