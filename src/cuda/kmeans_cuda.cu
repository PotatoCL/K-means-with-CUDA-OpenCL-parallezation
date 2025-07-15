#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "kernels.cuh"

class KMeansCUDA {
private:
    int n_clusters;
    int max_iterations;
    float tolerance;
    int n_samples;
    int n_features;
    
    // Device pointers
    float *d_data;
    float *d_centroids;
    float *d_distances;
    int *d_labels;
    int *d_counts;
    
    // Host data
    std::vector<float> h_centroids;
    std::vector<int> h_labels;
    
    int threads_per_block = 256;
    
public:
    KMeansCUDA(int k, int max_iter = 300, float tol = 1e-4)
        : n_clusters(k), max_iterations(max_iter), tolerance(tol) {}
    
    ~KMeansCUDA() {
        cleanup();
    }
    
    void cleanup() {
        if (d_data) cudaFree(d_data);
        if (d_centroids) cudaFree(d_centroids);
        if (d_distances) cudaFree(d_distances);
        if (d_labels) cudaFree(d_labels);
        if (d_counts) cudaFree(d_counts);
    }
    
    void initialize_centroids_gpu(const std::vector<float>& data) {
        // K-means++ initialization on GPU
        curandState_t* states;
        cudaMalloc(&states, n_clusters * sizeof(curandState_t));
        
        // Initialize with first random centroid
        int first_idx = rand() % n_samples;
        cudaMemcpy(d_centroids, &d_data[first_idx * n_features],
                   n_features * sizeof(float), cudaMemcpyDeviceToDevice);
        
        // Implement weighted selection for remaining centroids
        for (int i = 1; i < n_clusters; i++) {
            // Compute distances to nearest centroid
            int blocks = (n_samples + threads_per_block - 1) / threads_per_block;
            compute_distances<<<blocks, threads_per_block>>>(
                d_data, d_centroids, d_distances, n_samples, n_features, i);
            
            // Select next centroid based on weighted probability
            // (Simplified version - in practice, use thrust for reduction)
            std::vector<float> h_distances(n_samples * i);
            cudaMemcpy(h_distances.data(), d_distances,
                      n_samples * i * sizeof(float), cudaMemcpyDeviceToHost);
            
            // Find minimum distances and select next centroid
            std::vector<float> min_distances(n_samples);
            for (int j = 0; j < n_samples; j++) {
                float min_dist = h_distances[j * i];
                for (int k = 1; k < i; k++) {
                    min_dist = std::min(min_dist, h_distances[j * i + k]);
                }
                min_distances[j] = min_dist * min_dist;
            }
            
            // Weighted selection
            float sum = 0.0f;
            for (float d : min_distances) sum += d;
            float target = ((float)rand() / RAND_MAX) * sum;
            float cumsum = 0.0f;
            int selected = 0;
            
            for (int j = 0; j < n_samples; j++) {
                cumsum += min_distances[j];
                if (cumsum >= target) {
                    selected = j;
                    break;
                }
            }
            
            cudaMemcpy(&d_centroids[i * n_features],
                      &d_data[selected * n_features],
                      n_features * sizeof(float), cudaMemcpyDeviceToDevice);
        }
        
        cudaFree(states);
    }
    
    void fit(const std::vector<float>& data, int samples, int features) {
        n_samples = samples;
        n_features = features;
        
        // Allocate device memory
        size_t data_size = n_samples * n_features * sizeof(float);
        size_t centroid_size = n_clusters * n_features * sizeof(float);
        size_t distance_size = n_samples * n_clusters * sizeof(float);
        size_t label_size = n_samples * sizeof(int);
        size_t count_size = n_clusters * sizeof(int);
        
        cudaMalloc(&d_data, data_size);
        cudaMalloc(&d_centroids, centroid_size);
        cudaMalloc(&d_distances, distance_size);
        cudaMalloc(&d_labels, label_size);
        cudaMalloc(&d_counts, count_size);
        
        // Copy data to device
        cudaMemcpy(d_data, data.data(), data_size, cudaMemcpyHostToDevice);
        
        // Initialize centroids
        initialize_centroids_gpu(data);
        
        // Setup for kernel launches
        int blocks_samples = (n_samples + threads_per_block - 1) / threads_per_block;
        int blocks_centroids = (n_clusters * n_features + threads_per_block - 1) / 
                              threads_per_block;
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        
        // Main K-means loop
        for (int iter = 0; iter < max_iterations; iter++) {
            // Compute distances
            compute_distances<<<blocks_samples, threads_per_block>>>(
                d_data, d_centroids, d_distances, n_samples, n_features, n_clusters);
            
            // Assign clusters
            assign_clusters<<<blocks_samples, threads_per_block>>>(
                d_distances, d_labels, n_samples, n_clusters);
            
            // Save old centroids for convergence check
            std::vector<float> old_centroids(n_clusters * n_features);
            cudaMemcpy(old_centroids.data(), d_centroids, centroid_size,
                      cudaMemcpyDeviceToHost);
            
            // Reset centroids and counts
            cudaMemset(d_centroids, 0, centroid_size);
            cudaMemset(d_counts, 0, count_size);
            
            // Update centroids
            update_centroids_atomic<<<blocks_samples, threads_per_block>>>(
                d_data, d_labels, d_centroids, d_counts,
                n_samples, n_features, n_clusters);
            
            // Finalize centroids
            finalize_centroids<<<blocks_centroids, threads_per_block>>>(
                d_centroids, d_counts, n_features, n_clusters);
            
            // Check convergence
            h_centroids.resize(n_clusters * n_features);
            cudaMemcpy(h_centroids.data(), d_centroids, centroid_size,
                      cudaMemcpyDeviceToHost);
            
            float max_change = 0.0f;
            for (int i = 0; i < n_clusters * n_features; i++) {
                float change = std::abs(h_centroids[i] - old_centroids[i]);
                max_change = std::max(max_change, change);
            }
            
            if (max_change < tolerance) {
                std::cout << "CUDA converged at iteration " << iter << std::endl;
                break;
            }
        }
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "CUDA K-means completed in " << milliseconds << " ms" << std::endl;
        
        // Copy results back
        h_labels.resize(n_samples);
        cudaMemcpy(h_labels.data(), d_labels, label_size, cudaMemcpyDeviceToHost);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    std::vector<int> get_labels() const { return h_labels; }
    std::vector<float> get_centroids() const { return h_centroids; }
};