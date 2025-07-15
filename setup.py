from setuptools import setup, find_packages, Extension
from pathlib import Path
import subprocess
import sys

# Build C++ extensions
def build_extensions():
    build_dir = Path("build")
    build_dir.mkdir(exist_ok=True)
    
    # Run CMake
    subprocess.check_call(["cmake", ".."], cwd=build_dir)
    subprocess.check_call(["make", "-j4"], cwd=build_dir)

# Build before setup
build_extensions()

setup(
    name="kmeans-gpu",
    version="1.0.0",
    author="Your Name",
    description="GPU-accelerated K-means clustering with CUDA and OpenCL",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
    ],
    extras_require={
        "cuda": ["pycuda>=2021.1"],
        "opencl": ["pyopencl>=2021.2.0"],
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)