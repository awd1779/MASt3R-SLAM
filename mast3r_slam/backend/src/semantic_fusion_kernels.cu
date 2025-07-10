#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace {

template <typename scalar_t>
__global__ void semantic_fusion_kernel_impl(
    const scalar_t* __restrict__ X_world,      // [N, 3] 3D points in world
    const int* __restrict__ semantic_mask,     // [H, W] semantic mask
    const scalar_t* __restrict__ confidences,  // [MAX_INSTANCES] confidence scores
    const scalar_t* __restrict__ K,            // [3, 3] intrinsics
    const scalar_t* __restrict__ T_CW,         // [4, 4] camera-from-world
    int* __restrict__ out_labels,              // [N] output labels
    scalar_t* __restrict__ out_conf,           // [N] output confidences
    const int* __restrict__ existing_labels,   // [N] existing labels (or nullptr)
    const scalar_t* __restrict__ existing_conf,// [N] existing confidence (or nullptr)
    int N, int H, int W, int MAX_INSTANCES) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    // Load 3D point
    scalar_t x = X_world[idx * 3 + 0];
    scalar_t y = X_world[idx * 3 + 1];
    scalar_t z = X_world[idx * 3 + 2];
    
    // Transform to camera coordinates
    scalar_t x_cam = T_CW[0] * x + T_CW[1] * y + T_CW[2] * z + T_CW[3];
    scalar_t y_cam = T_CW[4] * x + T_CW[5] * y + T_CW[6] * z + T_CW[7];
    scalar_t z_cam = T_CW[8] * x + T_CW[9] * y + T_CW[10] * z + T_CW[11];
    
    // Check if point is in front of camera
    if (z_cam <= 0.1) {
        // Keep existing label or set to 0
        out_labels[idx] = existing_labels ? existing_labels[idx] : 0;
        out_conf[idx] = existing_conf ? existing_conf[idx] : 0.0f;
        return;
    }
    
    // Project to image plane
    scalar_t x_img = K[0] * x_cam + K[1] * y_cam + K[2] * z_cam;
    scalar_t y_img = K[3] * x_cam + K[4] * y_cam + K[5] * z_cam;
    scalar_t z_img = K[6] * x_cam + K[7] * y_cam + K[8] * z_cam;
    
    // Normalize by depth
    scalar_t inv_z = 1.0f / z_img;
    scalar_t px = x_img * inv_z;
    scalar_t py = y_img * inv_z;
    
    // Round to nearest pixel
    int px_int = __float2int_rn(px);
    int py_int = __float2int_rn(py);
    
    // Check bounds
    if (px_int < 0 || px_int >= W || py_int < 0 || py_int >= H) {
        // Keep existing label or set to 0
        out_labels[idx] = existing_labels ? existing_labels[idx] : 0;
        out_conf[idx] = existing_conf ? existing_conf[idx] : 0.0f;
        return;
    }
    
    // Sample semantic mask
    int label = semantic_mask[py_int * W + px_int];
    
    // Get confidence for this label
    scalar_t conf = 0.0f;
    if (label > 0 && label < MAX_INSTANCES) {
        conf = confidences[label];
    }
    
    // Update based on confidence
    if (existing_labels && existing_conf) {
        // Only update if new confidence is higher
        if (conf > existing_conf[idx]) {
            out_labels[idx] = label;
            out_conf[idx] = conf;
        } else {
            out_labels[idx] = existing_labels[idx];
            out_conf[idx] = existing_conf[idx];
        }
    } else {
        // No existing labels, just assign
        out_labels[idx] = label;
        out_conf[idx] = conf;
    }
}

} // namespace

torch::Tensor semantic_fusion_cuda_forward(
    torch::Tensor X_world,
    torch::Tensor semantic_mask,
    torch::Tensor confidences,
    torch::Tensor K,
    torch::Tensor T_CW,
    torch::optional<torch::Tensor> existing_labels,
    torch::optional<torch::Tensor> existing_conf) {
    
    const auto N = X_world.size(0);
    const auto H = semantic_mask.size(0);
    const auto W = semantic_mask.size(1);
    const auto MAX_INSTANCES = confidences.size(0);
    
    // Prepare outputs
    auto out_labels = torch::zeros({N}, torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(X_world.device()));
    auto out_conf = torch::zeros({N}, X_world.options());
    
    // Launch kernel
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(X_world.scalar_type(), "semantic_fusion_cuda", ([&] {
        semantic_fusion_kernel_impl<scalar_t><<<blocks, threads>>>(
            X_world.data_ptr<scalar_t>(),
            semantic_mask.data_ptr<int>(),
            confidences.data_ptr<scalar_t>(),
            K.data_ptr<scalar_t>(),
            T_CW.data_ptr<scalar_t>(),
            out_labels.data_ptr<int>(),
            out_conf.data_ptr<scalar_t>(),
            existing_labels.has_value() ? existing_labels.value().data_ptr<int>() : nullptr,
            existing_conf.has_value() ? existing_conf.value().data_ptr<scalar_t>() : nullptr,
            N, H, W, MAX_INSTANCES
        );
    }));
    
    // Synchronize
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA error: ", cudaGetErrorString(err));
    }
    
    return out_labels;
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("semantic_fusion_forward", &semantic_fusion_cuda_forward, "Semantic fusion forward (CUDA)");
}