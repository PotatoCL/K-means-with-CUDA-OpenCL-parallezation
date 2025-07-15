#ifndef KMEANS_KERNELS_CUH
#define KMEANS_KERNELS_CUH

#include <cuda_runtime.h>

// Distance calculation kernel
__global__ void compute_distances(const float* data, const float* centroids,
                                 float* distances, int n_samples, int n_features,
                                 int n_clusters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_samples) {
        for (int c = 0; c < n_clusters; c++) {
            float dist = 0.0f;
            for (int f = 0; f < n_features; f++) {
                float diff = data[idx * n_features + f] - 
                            centroids[c * n_features + f];
                dist += diff * diff;
            }
            distances[idx * n_clusters + c] = sqrtf(dist);
        }
    }
}

// Assignment kernel
__global__ void assign_clusters(const float* distances, int* labels,
                               int n_samples, int n_clusters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_samples) {
        float min_dist = distances[idx * n_clusters];
        int best_cluster = 0;
        
        for (int c = 1; c < n_clusters; c++) {
            float dist = distances[idx * n_clusters + c];
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = c;
            }
        }
        labels[idx] = best_cluster;
    }
}

// Centroid update kernel using atomic operations
__global__ void update_centroids_atomic(const float* data, const int* labels,
                                       float* centroids, int* counts,
                                       int n_samples, int n_features,
                                       int n_clusters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_samples) {
        int cluster = labels[idx];
        atomicAdd(&counts[cluster], 1);
        
        for (int f = 0; f < n_features; f++) {
            atomicAdd(&centroids[cluster * n_features + f],
                     data[idx * n_features + f]);
        }
    }
}

// Finalize centroids
__global__ void finalize_centroids(float* centroids, const int* counts,
                                  int n_features, int n_clusters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int cluster = idx / n_features;
    int feature = idx % n_features;
    
    if (cluster < n_clusters && feature < n_features && counts[cluster] > 0) {
        centroids[idx] /= counts[cluster];
    }
}

#endif // KMEANS_KERNELS_CUH