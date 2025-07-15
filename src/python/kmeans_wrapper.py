import numpy as np
import ctypes
from typing import Tuple, Optional, Literal
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class KMeansGPU:
    """
    GPU-accelerated K-means clustering with CUDA and OpenCL backends.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters to form
    max_iter : int, default=300
        Maximum number of iterations
    tol : float, default=1e-4
        Tolerance for convergence
    backend : {'cuda', 'opencl', 'cpu'}, default='cuda'
        Computation backend to use
    init : {'k-means++', 'random'}, default='k-means++'
        Method for initialization
    n_init : int, default=10
        Number of times to run k-means with different seeds
    random_state : int, optional
        Random state for reproducibility
    """
    
    def __init__(self, 
                 n_clusters: int,
                 max_iter: int = 300,
                 tol: float = 1e-4,
                 backend: Literal['cuda', 'opencl', 'cpu'] = 'cuda',
                 init: Literal['k-means++', 'random'] = 'k-means++',
                 n_init: int = 10,
                 random_state: Optional[int] = None):
        
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.backend = backend
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
        
        self.labels_ = None
        self.cluster_centers_ = None
        self.inertia_ = None
        self.n_iter_ = None
        
        # Load appropriate backend library
        self._load_backend()
    
    def _load_backend(self):
        """Load the compiled backend library."""
        lib_path = Path(__file__).parent.parent / 'build'
        
        if self.backend == 'cuda':
            lib_file = lib_path / 'libkmeans_cuda.so'
            if not lib_file.exists():
                raise RuntimeError(f"CUDA library not found at {lib_file}")
            self._lib = ctypes.CDLL(str(lib_file))
            self._setup_cuda_functions()
            
        elif self.backend == 'opencl':
            lib_file = lib_path / 'libkmeans_opencl.so'
            if not lib_file.exists():
                raise RuntimeError(f"OpenCL library not found at {lib_file}")
            self._lib = ctypes.CDLL(str(lib_file))
            self._setup_opencl_functions()
            
        elif self.backend == 'cpu':
            lib_file = lib_path / 'libkmeans_cpu.so'
            if not lib_file.exists():
                raise RuntimeError(f"CPU library not found at {lib_file}")
            self._lib = ctypes.CDLL(str(lib_file))
            self._setup_cpu_functions()
    
    def _setup_cuda_functions(self):
        """Setup CUDA function signatures."""
        # Define function signatures
        self._lib.kmeans_cuda_fit.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # data
            ctypes.c_int,                     # n_samples
            ctypes.c_int,                     # n_features
            ctypes.c_int,                     # n_clusters
            ctypes.c_int,                     # max_iter
            ctypes.c_float,                   # tolerance
            ctypes.POINTER(ctypes.c_int),     # labels (output)
            ctypes.POINTER(ctypes.c_float),   # centroids (output)
            ctypes.POINTER(ctypes.c_float),   # inertia (output)
            ctypes.POINTER(ctypes.c_int)      # n_iter (output)
        ]
        self._lib.kmeans_cuda_fit.restype = ctypes.c_int
    
    def fit(self, X: np.ndarray) -> 'KMeansGPU':
        """
        Compute k-means clustering.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
            
        Returns
        -------
        self : KMeansGPU
            Fitted estimator.
        """
        X = self._validate_data(X)
        n_samples, n_features = X.shape
        
        if n_samples < self.n_clusters:
            raise ValueError(f"n_samples={n_samples} should be >= n_clusters={self.n_clusters}")
        
        # Best run tracking
        best_labels = None
        best_centers = None
        best_inertia = float('inf')
        best_n_iter = 0
        
        # Multiple runs with different initializations
        for run in range(self.n_init):
            if self.random_state is not None:
                np.random.seed(self.random_state + run)
            
            # Prepare output arrays
            labels = np.zeros(n_samples, dtype=np.int32)
            centers = np.zeros((self.n_clusters, n_features), dtype=np.float32)
            inertia = np.zeros(1, dtype=np.float32)
            n_iter = np.zeros(1, dtype=np.int32)
            
            # Call backend implementation
            if self.backend == 'cuda':
                ret = self._lib.kmeans_cuda_fit(
                    X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    n_samples,
                    n_features,
                    self.n_clusters,
                    self.max_iter,
                    self.tol,
                    labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                    centers.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    inertia.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    n_iter.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
                )
                
                if ret != 0:
                    raise RuntimeError(f"CUDA K-means failed with error code {ret}")
            
            # Update best run if needed
            if inertia[0] < best_inertia:
                best_labels = labels.copy()
                best_centers = centers.copy()
                best_inertia = inertia[0]
                best_n_iter = n_iter[0]
        
        # Store results
        self.labels_ = best_labels
        self.cluster_centers_ = best_centers
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the closest cluster for each sample.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.
            
        Returns
        -------
        labels : array of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        X = self._validate_data(X)
        
        # Compute distances to all centers
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            distances[:, i] = np.linalg.norm(X - self.cluster_centers_[i], axis=1)
        
        return np.argmin(distances, axis=1)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Compute cluster centers and predict cluster index for each sample."""
        self.fit(X)
        return self.labels_
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X to cluster-distance space."""
        if self.cluster_centers_ is None:
            raise RuntimeError("Model must be fitted before transform")
        
        X = self._validate_data(X)
        distances = np.zeros((X.shape[0], self.n_clusters))
        
        for i in range(self.n_clusters):
            distances[:, i] = np.linalg.norm(X - self.cluster_centers_[i], axis=1)
        
        return distances
    
    def _validate_data(self, X: np.ndarray) -> np.ndarray:
        """Validate and convert input data."""
        X = np.asarray(X, dtype=np.float32, order='C')
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")
        return X