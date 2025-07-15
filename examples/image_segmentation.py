import numpy as np
import time
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

from src.python.kmeans_wrapper import KMeansGPU
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, adjusted_rand_score

class KMeansBenchmark:
    """Comprehensive benchmark suite for K-means implementations."""
    
    def __init__(self, output_dir: str = "benchmarks/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    def generate_dataset(self, 
                        n_samples: int, 
                        n_features: int, 
                        n_clusters: int,
                        cluster_std: float = 1.0,
                        random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic dataset for benchmarking."""
        X, y = make_blobs(n_samples=n_samples,
                         n_features=n_features,
                         centers=n_clusters,
                         cluster_std=cluster_std,
                         random_state=random_state)
        return X.astype(np.float32), y
    
    def benchmark_implementation(self,
                               X: np.ndarray,
                               n_clusters: int,
                               backend: str,
                               n_runs: int = 5) -> Dict:
        """Benchmark a single implementation."""
        times = []
        silhouette_scores = []
        
        for _ in range(n_runs):
            if backend == 'sklearn':
                model = SklearnKMeans(n_clusters=n_clusters, 
                                     max_iter=300,
                                     n_init=10,
                                     random_state=42)
            else:
                model = KMeansGPU(n_clusters=n_clusters,
                                 backend=backend,
                                 max_iter=300,
                                 n_init=10,
                                 random_state=42)
            
            start_time = time.time()
            labels = model.fit_predict(X)
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            # Calculate metrics
            if len(np.unique(labels)) > 1:
                silhouette = silhouette_score(X, labels, sample_size=min(10000, len(X)))
                silhouette_scores.append(silhouette)
        
        return {
            'backend': backend,
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'mean_silhouette': np.mean(silhouette_scores) if silhouette_scores else 0,
            'times': times
        }
    
    def run_scaling_benchmark(self):
        """Run benchmark across different dataset sizes."""
        sample_sizes = [10_000, 50_000, 100_000, 500_000, 1_000_000]
        n_features = 50
        n_clusters = 10
        
        backends = ['cpu', 'cuda', 'opencl', 'sklearn']
        
        for n_samples in sample_sizes:
            print(f"\nBenchmarking with {n_samples:,} samples...")
            X, y = self.generate_dataset(n_samples, n_features, n_clusters)
            
            for backend in backends:
                try:
                    print(f"  Testing {backend}...")
                    result = self.benchmark_implementation(X, n_clusters, backend)
                    result.update({
                        'n_samples': n_samples,
                        'n_features': n_features,
                        'n_clusters': n_clusters
                    })
                    self.results.append(result)
                except Exception as e:
                    print(f"    Failed: {e}")
    
    def run_dimension_benchmark(self):
        """Run benchmark across different feature dimensions."""
        n_samples = 100_000
        feature_sizes = [10, 50, 100, 200, 500]
        n_clusters = 10
        
        backends = ['cpu', 'cuda', 'opencl', 'sklearn']
        
        for n_features in feature_sizes:
            print(f"\nBenchmarking with {n_features} features...")
            X, y = self.generate_dataset(n_samples, n_features, n_clusters)
            
            for backend in backends:
                try:
                    print(f"  Testing {backend}...")
                    result = self.benchmark_implementation(X, n_clusters, backend)
                    result.update({
                        'n_samples': n_samples,
                        'n_features': n_features,
                        'n_clusters': n_clusters,
                        'benchmark_type': 'dimension'
                    })
                    self.results.append(result)
                except Exception as e:
                    print(f"    Failed: {e}")
    
    def run_cluster_benchmark(self):
        """Run benchmark across different numbers of clusters."""
        n_samples = 100_000
        n_features = 50
        cluster_sizes = [5, 10, 20, 50, 100]
        
        backends = ['cpu', 'cuda', 'opencl', 'sklearn']
        
        for n_clusters in cluster_sizes:
            print(f"\nBenchmarking with {n_clusters} clusters...")
            X, y = self.generate_dataset(n_samples, n_features, n_clusters)
            
            for backend in backends:
                try:
                    print(f"  Testing {backend}...")
                    result = self.benchmark_implementation(X, n_clusters, backend)
                    result.update({
                        'n_samples': n_samples,
                        'n_features': n_features,
                        'n_clusters': n_clusters,
                        'benchmark_type': 'clusters'
                    })
                    self.results.append(result)
                except Exception as e:
                    print(f"    Failed: {e}")
    
    def save_results(self):
        """Save benchmark results."""
        df = pd.DataFrame(self.results)
        
        # Save as CSV
        csv_path = self.output_dir / 'benchmark_results.csv'
        df.to_csv(csv_path, index=False)
        
        # Save as JSON
        json_path = self.output_dir / 'benchmark_results.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to {self.output_dir}")
        return df
    
    def plot_results(self):
        """Generate visualization plots for benchmark results."""
        df = pd.DataFrame(self.results)
        
        # Scaling plot
        scaling_df = df[df['benchmark_type'].isna() | (df['benchmark_type'] == 'scaling')]
        
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Execution time vs dataset size
        plt.subplot(2, 2, 1)
        for backend in scaling_df['backend'].unique():
            backend_df = scaling_df[scaling_df['backend'] == backend]
            plt.plot(backend_df['n_samples'], backend_df['mean_time'], 
                    marker='o', label=backend, linewidth=2, markersize=8)
        
        plt.xlabel('Number of Samples')
        plt.ylabel('Execution Time (seconds)')
        plt.title('K-means Scaling Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.yscale('log')
        
        # Plot 2: Speedup relative to CPU
        plt.subplot(2, 2, 2)
        cpu_times = scaling_df[scaling_df['backend'] == 'cpu'].set_index('n_samples')['mean_time']
        
        for backend in ['cuda', 'opencl']:
            backend_df = scaling_df[scaling_df['backend'] == backend].set_index('n_samples')
            speedup = cpu_times / backend_df['mean_time']
            plt.plot(speedup.index, speedup.values, 
                    marker='o', label=f'{backend} speedup', linewidth=2, markersize=8)
        
        plt.xlabel('Number of Samples')
        plt.ylabel('Speedup over CPU')
        plt.title('GPU Speedup Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        
        # Plot 3: Dimension scaling
        plt.subplot(2, 2, 3)
        dim_df = df[df['benchmark_type'] == 'dimension']
        for backend in dim_df['backend'].unique():
            backend_df = dim_df[dim_df['backend'] == backend]
            plt.plot(backend_df['n_features'], backend_df['mean_time'],
                    marker='s', label=backend, linewidth=2, markersize=8)
        
        plt.xlabel('Number of Features')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Performance vs Dimensionality')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Cluster scaling
        plt.subplot(2, 2, 4)
        cluster_df = df[df['benchmark_type'] == 'clusters']
        for backend in cluster_df['backend'].unique():
            backend_df = cluster_df[cluster_df['backend'] == backend]
            plt.plot(backend_df['n_clusters'], backend_df['mean_time'],
                    marker='^', label=backend, linewidth=2, markersize=8)
        
        plt.xlabel('Number of Clusters')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Performance vs Number of Clusters')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'benchmark_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """Generate a comprehensive benchmark report."""
        df = pd.DataFrame(self.results)
        
        report = []
        report.append("# K-means GPU Acceleration Benchmark Report\n")
        report.append(f"Generated on: {pd.Timestamp.now()}\n")
        
        # Overall speedup summary
        report.append("## Overall Speedup Summary\n")
        
        cpu_baseline = df[df['backend'] == 'cpu'].groupby('n_samples')['mean_time'].mean()
        
        for backend in ['cuda', 'opencl']:
            backend_times = df[df['backend'] == backend].groupby('n_samples')['mean_time'].mean()
            speedups = cpu_baseline / backend_times
            avg_speedup = speedups.mean()
            max_speedup = speedups.max()
            
            report.append(f"### {backend.upper()} Performance\n")
            report.append(f"- Average speedup: {avg_speedup:.1f}x\n")
            report.append(f"- Maximum speedup: {max_speedup:.1f}x\n")
            report.append(f"- Best performance at: {speedups.idxmax():,} samples\n\n")
        
        # Detailed results table
        report.append("## Detailed Results\n\n")
        summary_df = df.groupby(['backend', 'n_samples']).agg({
            'mean_time': 'mean',
            'std_time': 'mean',
            'mean_silhouette': 'mean'
        }).round(3)
        
        report.append(summary_df.to_markdown())
        
        # Save report
        report_path = self.output_dir / 'benchmark_report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"\nReport saved to {report_path}")
