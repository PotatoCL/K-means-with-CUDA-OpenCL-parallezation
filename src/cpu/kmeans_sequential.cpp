#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include <chrono>

class KMeansCPU {
private:
    int n_clusters;
    int max_iterations;
    float tolerance;
    std::vector<std::vector<float>> centroids;
    std::vector<int> labels;
    
public:
    KMeansCPU(int k, int max_iter = 300, float tol = 1e-4) 
        : n_clusters(k), max_iterations(max_iter), tolerance(tol) {}
    
    float euclidean_distance(const std::vector<float>& p1, 
                            const std::vector<float>& p2) {
        float sum = 0.0f;
        for (size_t i = 0; i < p1.size(); ++i) {
            float diff = p1[i] - p2[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }
    
    void initialize_centroids(const std::vector<std::vector<float>>& data) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, data.size() - 1);
        
        centroids.clear();
        std::vector<int> selected;
        
        // K-means++ initialization
        centroids.push_back(data[dis(gen)]);
        
        for (int i = 1; i < n_clusters; ++i) {
            std::vector<float> distances(data.size());
            float sum_distances = 0.0f;
            
            for (size_t j = 0; j < data.size(); ++j) {
                float min_dist = std::numeric_limits<float>::max();
                for (const auto& centroid : centroids) {
                    float dist = euclidean_distance(data[j], centroid);
                    min_dist = std::min(min_dist, dist);
                }
                distances[j] = min_dist * min_dist;
                sum_distances += distances[j];
            }
            
            // Weighted random selection
            std::uniform_real_distribution<float> prob_dis(0, sum_distances);
            float target = prob_dis(gen);
            float cumsum = 0.0f;
            
            for (size_t j = 0; j < data.size(); ++j) {
                cumsum += distances[j];
                if (cumsum >= target) {
                    centroids.push_back(data[j]);
                    break;
                }
            }
        }
    }
    
    void fit(const std::vector<std::vector<float>>& data) {
        auto start = std::chrono::high_resolution_clock::now();
        
        initialize_centroids(data);
        labels.resize(data.size());
        
        for (int iter = 0; iter < max_iterations; ++iter) {
            // Assignment step
            for (size_t i = 0; i < data.size(); ++i) {
                float min_dist = std::numeric_limits<float>::max();
                int best_cluster = 0;
                
                for (int j = 0; j < n_clusters; ++j) {
                    float dist = euclidean_distance(data[i], centroids[j]);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_cluster = j;
                    }
                }
                labels[i] = best_cluster;
            }
            
            // Update step
            std::vector<std::vector<float>> new_centroids(n_clusters, 
                std::vector<float>(data[0].size(), 0.0f));
            std::vector<int> counts(n_clusters, 0);
            
            for (size_t i = 0; i < data.size(); ++i) {
                int cluster = labels[i];
                for (size_t j = 0; j < data[i].size(); ++j) {
                    new_centroids[cluster][j] += data[i][j];
                }
                counts[cluster]++;
            }
            
            float max_change = 0.0f;
            for (int i = 0; i < n_clusters; ++i) {
                if (counts[i] > 0) {
                    for (size_t j = 0; j < new_centroids[i].size(); ++j) {
                        new_centroids[i][j] /= counts[i];
                    }
                    float change = euclidean_distance(centroids[i], new_centroids[i]);
                    max_change = std::max(max_change, change);
                }
            }
            
            centroids = new_centroids;
            
            if (max_change < tolerance) {
                std::cout << "Converged at iteration " << iter << std::endl;
                break;
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "CPU K-means completed in " << duration.count() << " ms" << std::endl;
    }
    
    std::vector<int> get_labels() const { return labels; }
    std::vector<std::vector<float>> get_centroids() const { return centroids; }
};