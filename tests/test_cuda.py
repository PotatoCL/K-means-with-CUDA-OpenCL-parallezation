import pytest
import numpy as np
from src.python.kmeans_wrapper import KMeansGPU
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

class TestKMeansCUDA:
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        X, y = make_blobs(n_samples=1000, n_features=10, 
                         centers=5, random_state=42)
        return X.astype(np.float32), y
    
    def test_cuda_available(self):
        """Test if CUDA is available."""
        try:
            import pycuda.driver as cuda
            cuda.init()
            assert cuda.Device.count() > 0
        except Exception as e:
            pytest.skip(f"CUDA not available: {e}")
    
    def test_basic_clustering(self, sample_data):
        """Test basic clustering functionality."""
        X, y_true = sample_data
        
        kmeans = KMeansGPU(n_clusters=5, backend='cuda', random_state=42)
        y_pred = kmeans.fit_predict(X)
        
        # Check basic properties
        assert len(y_pred) == len(X)
        assert len(np.unique(y_pred)) == 5
        assert kmeans.cluster_centers_.shape == (5, 10)
        
        # Check clustering quality
        ari = adjusted_rand_score(y_true, y_pred)
        assert ari > 0.8  # Should achieve good clustering on synthetic data
    
    def test_convergence(self, sample_data):
        """Test that algorithm converges."""
        X, _ = sample_data
        
        kmeans = KMeansGPU(n_clusters=5, backend='cuda', max_iter=300)
        kmeans.fit(X)
        
        assert kmeans.n_iter_ < 300  # Should converge before max iterations
    
    def test_deterministic_results(self, sample_data):
        """Test reproducibility with fixed random state."""
        X, _ = sample_data
        
        kmeans1 = KMeansGPU(n_clusters=5, backend='cuda', random_state=42)
        kmeans2 = KMeansGPU(n_clusters=5, backend='cuda', random_state=42)
        
        labels1 = kmeans1.fit_predict(X)
        labels2 = kmeans2.fit_predict(X)
        
        assert np.array_equal(labels1, labels2)
        assert np.allclose(kmeans1.cluster_centers_, kmeans2.cluster_centers_)
    
    def test_large_dataset(self):
        """Test performance on larger dataset."""
        X, y = make_blobs(n_samples=100_000, n_features=50, 
                         centers=10, random_state=42)
        X = X.astype(np.float32)
        
        kmeans = KMeansGPU(n_clusters=10, backend='cuda', max_iter=100)
        labels = kmeans.fit_predict(X)
        
        assert len(labels) == 100_000
        assert len(np.unique(labels)) == 10
    
    @pytest.mark.parametrize("n_clusters", [2, 5, 10, 20])
    def test_different_cluster_counts(self, n_clusters):
        """Test with different numbers of clusters."""
        X, _ = make_blobs(n_samples=5000, n_features=20, 
                         centers=n_clusters, random_state=42)
        X = X.astype(np.float32)
        
        kmeans = KMeansGPU(n_clusters=n_clusters, backend='cuda')
        labels = kmeans.fit_predict(X)
        
        assert len(np.unique(labels)) == n_clusters
    
    def test_predict_new_data(self, sample_data):
        """Test prediction on new data."""
        X_train, _ = sample_data
        X_test, _ = make_blobs(n_samples=200, n_features=10, 
                              centers=5, random_state=123)
        X_test = X_test.astype(np.float32)
        
        kmeans = KMeansGPU(n_clusters=5, backend='cuda')
        kmeans.fit(X_train)
        
        labels_train = kmeans.labels_
        labels_test = kmeans.predict(X_test)
        
        assert len(labels_test) == 200
        assert labels_test.min() >= 0
        assert labels_test.max() < 5