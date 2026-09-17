# sparsekmeans Package

The sparsekmeans package provides an efficient implementation of the K-means clustering algorithm optimized for sparse data sets. It is designed to handle high-dimensional and sparse data commonly found in text mining, recommender systems, and bioinformatics. By leveraging appropriate storage format and sparse matrix multiplication operations, our package ensures significant speedup in running time while maintaining consistency for clustering results compared with scikit-learn. Besides, the design of the package allows users to easily extend and customize, making it suitable for research or integrating into large-scale machine learning systems.

Implementation details are in the paper ["SparseKmeans: Efficient K-means Clustering for Sparse Data"](https://www.csie.ntu.edu.tw/~cjlin/papers/sparse_kmeans/sparsekmeans-cikm.pdf), CIKM 2025.

## Installation

Use the following command to install sparsekmeans with python >= 3.10.

```bash
pip install sparsekmeans
```

GPU acceleration is also supported for CUDA 12 and 13. For example, users can install the GPU-enabled version with:

```bash
pip install "sparsekmeans[gpu-cu12]"
```

## Usage

We support two popular K-means algorithms: Lloyd's method and Elkan's method. While both require distance calculations between data points and centroids, Elkan's method reduces some computations by exploiting information from the previous iteration, specifically using the triangle inequality to maintain two distance-related bounds.

When the data set is small (e.g., the number of samples is less than 10,000), it is recommended to use Lloyd's. Lloyd's relies primarily on sparse matrix multiplications, whereas Elkan's mechanism to reduce distance calculations may cause non-negligible overhead for small problems. Conversely, for large data sets or the situation of a huge number of clusters (K), Elkan's method may significantly outperforms Lloyd's.

Use the following steps to cluster a data set:

```
from sparsekmeans import LloydKmeans, ElkanKmeans

kmeans = LloydKmeans(n_clusters=100)
kmeans = ElkanKmeans(n_clusters=100)
labels = kmeans.fit(X)
```

To use GPU acceleration for Lloyd's method, set `use_gpu=True`. The GPU-enabled installation is required; otherwise, an exception will be raised.

```
kmeans = LloydKmeans(n_clusters=100, use_gpu=True)
labels = kmeans.fit(X)
```

Users can specify the following parameters while creating the kmeans object.
```
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
```

To cluster additional data by using previously obtained centroids, use the following way.

```
test_labels = kmeans.predict(Xtest)

# To enable GPU acceleration, pass use_gpu=True:
test_labels = kmeans.predict(Xtest, use_gpu=True)
```

In case users have defined their own centroids and want to use them for clustering data, our package provides a function to do that:
```
from sparsekmeans import kmeans_predict

test_labels = kmeans_predict(Xtest, user_defined_centroids)

# To enable GPU acceleration, pass use_gpu=True:
test_labels = kmeans_predict(Xtest, user_defined_centroids, use_gpu=True)
```

## Example
We show an example to cluster a real-world data set.

```python
from sparsekmeans import LloydKmeans
from libsvm.svmutil import svm_read_problem
import io
import urllib.request
import bz2

with urllib.request.urlopen("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/news20_tfidf_train.svm.bz2") as r:
    with bz2.open(r, 'rt') as f:
        _, X = svm_read_problem(f, return_scipy=True)

kmeans = LloydKmeans(n_clusters=100)
labels = kmeans.fit(X)
```

## Note:
Sparse matrix should not contain duplicate entries. Duplicate entries may cause errors or incorrect behavior during sparse matrix multiplication.

## Benchmark

### CPU Benchmark

Running time comparison for **Lloyd's** algorithm


| Data sets    | K  |       sklearn          |  sparsekmeans |  Speedup |
|-------------:|------------:|---------------------:|-----:|-------------:|
| Wiki-500K   | 100 | 24,617.7    | 2,957.3        | 8.32x   |
|             | 500 | > $10^5$ | 26,104.9       | > 3.83x |
| Amazon-670K | 100 | 670.9       | 123.4         | 5.43x   | 
|             | 500 | 7,170.1     | 776.2         | 9.23x   |
| Url         | 100 | 799.9       | 163.5          | 4.89x   |
|             | 500 | 5,888.4     | 987.2          | 5.96x   |
| Amazon-3M   | 100 | 26,207.5    | 2,359.1        | 11.10x   |
|             | 500 | > $10^5$ | 39,346.1       | > 2.54x |

Running time comparison for **Elkan's** algorithm

| Data sets    | K  |      sklearn          |  sparsekmeans |  Speedup |
|-------------:|------------:|---------------------:|-----:|-------------:|
| Wiki-500K   | 100 | 4,042.7    | 2,382.1        | 1.69x   |
|             | 500 | 91,441.8 | 5,061.0       | 18.06x |
| Amazon-670K | 100 | 248.5       | 141.8         | 1.75x   | 
|             | 500 | 1,248.3     | 685.1         | 1.82x  |
| Url         | 100 | 719.8      | 296.0          | 2.43x   |
|             | 500 | 4,687.1     | 1,989.7          | 2.35x  |
| Amazon-3M   | 100 | 2,965.2    | 1,743.4        | 1.70x  |
|             | 500 | 13,340.0 | 5,517.9       | 2.41x|

### GPU Benchmark

Our current GPU implementation supports Lloyd's algorithm.

The GPU benchmark uses smaller data sets than the CPU benchmark because the larger data sets exceed the available GPU memory. Among them, Small URL is a subsampled version of the Url data set.


| Data sets    | K  | GPU (s) | CPU (s) | Speedup |
|-------------:|----:|--------:|--------:|--------:|
| EUR-Lex      | 100  | 2.28   | 4.23   | 1.86x   |
|              | 500  | 5.94   | 11.01  | 1.85x   |
| Wiki10-31K   | 100  | 18.46  | 69.88  | 3.79x   |
|              | 500  | 99.38  | 943.29 | 9.49x   |
| Small URL    | 100  | 32.18  | 57.13  | 1.78x   |
|              | 500  | 186.66 | 338.96 | 1.82x   |
|              | 1000 | 318.57 | 548.92 | 1.72x   |
