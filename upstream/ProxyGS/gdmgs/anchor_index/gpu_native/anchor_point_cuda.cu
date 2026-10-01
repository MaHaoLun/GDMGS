#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

using I = int64_t;

static void require_cuda_tensor(
    const at::Tensor& value,
    at::ScalarType dtype,
    int dimensions,
    const at::Device& device,
    const char* name
) {
    TORCH_CHECK(
        value.device() == device && value.scalar_type() == dtype &&
        value.dim() == dimensions && value.is_contiguous(),
        name, " has the wrong device, dtype, rank, or contiguity"
    );
}

__device__ float ordered_transform(const float* matrix, const float* point, int row) {
    float value = __fmul_rn(matrix[row], point[0]);
    value = __fadd_rn(value, __fmul_rn(matrix[4 + row], point[1]));
    value = __fadd_rn(value, __fmul_rn(matrix[8 + row], point[2]));
    value = __fadd_rn(value, matrix[12 + row]);
    return value;
}

__global__ void pointwise_keep_kernel(
    const float* positions,
    const I* candidate_ids,
    I candidate_count,
    I anchor_count,
    const float* depth,
    I height,
    I width,
    const float* world_view,
    const float* full_projection,
    float margin,
    float minimum_camera_z,
    uint8_t* keep
) {
    const I slot = I(blockIdx.x) * blockDim.x + threadIdx.x;
    if (slot >= candidate_count) {
        return;
    }
    const I row_id = candidate_ids[slot];
    if (row_id < 0 || row_id >= anchor_count) {
        keep[slot] = 0;
        return;
    }
    const float* point = positions + row_id * 3;
    const float camera_z = ordered_transform(world_view, point, 2);
    if (!(camera_z > minimum_camera_z)) {
        keep[slot] = 0;
        return;
    }
    const float clip_x = ordered_transform(full_projection, point, 0);
    const float clip_y = ordered_transform(full_projection, point, 1);
    const float clip_w = ordered_transform(full_projection, point, 3);
    const float reciprocal_w = 1.0f / __fadd_rn(clip_w, 1.0e-7f);
    const float projected_x = __fmul_rn(clip_x, reciprocal_w);
    const float projected_y = __fmul_rn(clip_y, reciprocal_w);
    float pixel_x = __fadd_rn(projected_x, 1.0f);
    pixel_x = __fmul_rn(pixel_x, static_cast<float>(width));
    pixel_x = pixel_x / 2.0f;
    float pixel_y = __fadd_rn(projected_y, 1.0f);
    pixel_y = __fmul_rn(pixel_y, static_cast<float>(height));
    pixel_y = pixel_y / 2.0f;
    if (!isfinite(pixel_x) || !isfinite(pixel_y)) {
        keep[slot] = 1;
        return;
    }
    const I column = static_cast<I>(pixel_x);
    const I depth_row = static_cast<I>(pixel_y);
    if (column < 0 || column >= width || depth_row < 0 || depth_row >= height) {
        keep[slot] = 1;
        return;
    }
    const float sampled = depth[depth_row * width + column];
    if (!isfinite(sampled)) {
        keep[slot] = 1;
        return;
    }
    keep[slot] = camera_z <= __fadd_rn(sampled, margin) ? 1 : 0;
}

at::Tensor pointwise_keep(
    at::Tensor positions,
    at::Tensor candidate_ids,
    at::Tensor depth,
    at::Tensor world_view,
    at::Tensor full_projection,
    double margin,
    double minimum_camera_z
) {
    TORCH_CHECK(positions.is_cuda(), "positions must be CUDA");
    const at::Device device = positions.device();
    const c10::cuda::CUDAGuard guard(device);
    require_cuda_tensor(positions, at::kFloat, 2, device, "positions");
    require_cuda_tensor(candidate_ids, at::kLong, 1, device, "candidate_ids");
    require_cuda_tensor(depth, at::kFloat, 2, device, "depth");
    require_cuda_tensor(world_view, at::kFloat, 2, device, "world_view");
    require_cuda_tensor(full_projection, at::kFloat, 2, device, "full_projection");
    TORCH_CHECK(positions.size(1) == 3, "positions must have shape [N,3]");
    TORCH_CHECK(world_view.sizes() == at::IntArrayRef({4, 4}), "world_view must be [4,4]");
    TORCH_CHECK(full_projection.sizes() == at::IntArrayRef({4, 4}), "full_projection must be [4,4]");
    TORCH_CHECK(depth.size(0) > 0 && depth.size(1) > 0, "depth must be nonempty");
    TORCH_CHECK(std::isfinite(margin) && margin >= 0.0, "margin must be finite and nonnegative");
    TORCH_CHECK(
        std::isfinite(minimum_camera_z) && minimum_camera_z >= 0.0,
        "minimum_camera_z must be finite and nonnegative"
    );
    auto output = at::empty({candidate_ids.numel()}, candidate_ids.options().dtype(at::kByte));
    if (candidate_ids.numel()) {
        const auto stream = at::cuda::getCurrentCUDAStream();
        pointwise_keep_kernel<<<(candidate_ids.numel() + 255) / 256, 256, 0, stream>>>(
            positions.data_ptr<float>(),
            candidate_ids.data_ptr<I>(),
            candidate_ids.numel(),
            positions.size(0),
            depth.data_ptr<float>(),
            depth.size(0),
            depth.size(1),
            world_view.data_ptr<float>(),
            full_projection.data_ptr<float>(),
            static_cast<float>(margin),
            static_cast<float>(minimum_camera_z),
            output.data_ptr<uint8_t>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}
