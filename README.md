# GPU-Accelerated K-means Clustering

A high-performance implementation of K-means clustering leveraging GPU parallelization through CUDA and OpenCL, achieving up to 30x speedup over traditional CPU implementations.

## Features

- **Multiple Backends**: CUDA, OpenCL, and optimized CPU implementations
- **Scalable Performance**: Handles datasets from 10K to 10M+ samples efficiently
- **Flexible Configuration**: Customizable distance metrics, initialization methods, and convergence criteria
- **Production Ready**: Comprehensive testing, benchmarking, and real-world examples
- **Easy Integration**: Python API compatible with scikit-learn

## Performance

![Benchmark Results](benchmarks/results/benchmark_plots.png)

Key performance metrics:
- **30x speedup** on large datasets (1M+ samples)
- **Linear scaling** with dataset size
- **Efficient memory usage** with optimized kernels

## Installation

### Prerequisites

- CUDA Toolkit 11.0+ (for CUDA backend)
- OpenCL 2.0+ (for OpenCL backend)
- Python 3.8+
- CMake 3.18+

### Build from Source

```bash
git clone https://github.com/yourusername/kmeans-gpu-clustering.git
cd kmeans-gpu-clustering

# Install Python dependencies
pip install -r requirements.txt

# Build C++ extensions
mkdir build && cd build
cmake ..
make -j4

# Install package
cd ..
pip install -e .