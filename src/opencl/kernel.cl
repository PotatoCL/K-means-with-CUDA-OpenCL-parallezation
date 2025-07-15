__kernel void compute_distances(__global const float* data,
                               __global const float* centroids,
                               __global float* distances,
                               const int n_features,
                               const int n_clusters) {
    int idx = get_global_id(0);
    int n_samples = get_global_size(0);
    
    for (int c = 0; c < n_clusters; c++) {
        float dist = 0.0f;
        for (int f = 0; f < n_features; f++) {
            float diff = data[idx * n_features + f] - 
                        centroids[c * n_features + f];
            dist += diff * diff;
        }
        distances[idx * n_clusters + c] = sqrt(dist);
    }
}

__kernel void assign_clusters(__global const float* distances,
                             __global int* labels,
                             const int n_clusters) {
    int idx = get_global_id(0);
    
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

__kernel void update_centroids(__global const float* data,
                              __global const int* labels,
                              __global float* centroids,
                              __global int* counts,
                              const int n_features,
                              const int n_clusters) {
    int idx = get_global_id(0);
    int cluster = labels[idx];
    
    atomic_inc(&counts[cluster]);
    
    for (int f = 0; f < n_features; f++) {
        // Note: atomic_add for floats might need custom implementation
        // depending on OpenCL version
        float val = data[idx * n_features + f];
        __global float* addr = &centroids[cluster * n_features + f];
        
        // Atomic add implementation for floats
        union {
            unsigned int intVal;
            float floatVal;
        } newVal, prevVal;
        
        do {
            prevVal.floatVal = *addr;
            newVal.floatVal = prevVal.floatVal + val;
        } while (atomic_cmpxchg((volatile __global unsigned int*)addr,
                                prevVal.intVal, newVal.intVal) != prevVal.intVal);
    }
}

__kernel void finalize_centroids(__global float* centroids,
                                __global const int* counts,
                                const int n_features) {
    int idx = get_global_id(0);
    int cluster = idx / n_features;
    int feature = idx % n_features;
    
    if (counts[cluster] > 0) {
        centroids[idx] /= counts[cluster];
    }
}