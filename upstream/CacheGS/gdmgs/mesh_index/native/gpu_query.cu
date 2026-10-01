#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>
#include <limits>

namespace {
constexpr int kThreads = 128;

void check_tensor(const torch::Tensor& value, torch::ScalarType dtype,
                  int64_t dimensions, const char* name) {
    TORCH_CHECK(value.is_cuda(), name, " must be CUDA resident");
    TORCH_CHECK(value.scalar_type() == dtype && value.dim() == dimensions,
                name, " has an invalid dtype or shape");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
}

void check_planes(const torch::Tensor& planes, const torch::Device& device) {
    check_tensor(planes, torch::kFloat64, 2, "planes");
    TORCH_CHECK(planes.size(1) == 8 && (planes.size(0) == 5 || planes.size(0) == 6),
                "planes must have shape [5 or 6,8]");
    TORCH_CHECK(planes.device() == device, "camera and geometry devices differ");
}

__device__ bool triangle_relevant(const double* vertices, const int64_t* face,
                                 const double* planes, int plane_count) {
    for (int plane = 0; plane < plane_count; ++plane) {
        const double* p = planes + plane * 8;
        bool has_inside_vertex = false;
        for (int corner = 0; corner < 3; ++corner) {
            const double* vertex = vertices + face[corner] * 3;
            double value = p[3], magnitude = fabs(p[3]), uncertainty = p[7];
            for (int axis = 0; axis < 3; ++axis) {
                const double term = p[axis] * vertex[axis];
                value += term;
                magnitude += fabs(term);
                uncertainty += p[axis + 4] * fabs(vertex[axis]);
            }
            const double tolerance = 64 * DBL_EPSILON * (magnitude + 1) + uncertainty;
            // Finite inputs can overflow a dot product. Uncertain geometry
            // must remain a candidate rather than treating NaN as outside.
            if (!isfinite(value) || !isfinite(magnitude) || !isfinite(uncertainty)
                    || !isfinite(tolerance)) {
                has_inside_vertex = true;
                break;
            }
            if (value >= -tolerance) {
                has_inside_vertex = true;
                break;
            }
        }
        if (!has_inside_vertex) return false;
    }
    return true;
}

__global__ void leaf_mask_kernel(const double* bounds, int64_t leaf_count,
                                const double* planes, int plane_count, bool* output) {
    const int64_t leaf = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (leaf >= leaf_count) return;
    const double* box = bounds + leaf * 6;
    bool relevant = true;
    for (int plane = 0; plane < plane_count; ++plane) {
        const double* p = planes + plane * 8;
        double value = p[3], magnitude = fabs(p[3]), uncertainty = p[7];
        for (int axis = 0; axis < 3; ++axis) {
            const double term = p[axis] * (p[axis] >= 0 ? box[axis + 3] : box[axis]);
            value += term;
            magnitude += fabs(term);
            uncertainty += p[axis + 4] * fmax(fabs(box[axis]), fabs(box[axis + 3]));
        }
        const double tolerance = 64 * DBL_EPSILON * (magnitude + 1) + uncertainty;
        if (!isfinite(value) || !isfinite(magnitude) || !isfinite(uncertainty)
                || !isfinite(tolerance)) continue;
        if (value < -tolerance) {
            relevant = false;
            break;
        }
    }
    output[leaf] = relevant;
}

__global__ void triangle_mask_kernel(const double* vertices, const int64_t* faces,
                                    int64_t face_count, const double* planes,
                                    int plane_count, const bool* active_leaves,
                                    const int64_t* face_leaf, bool indexed,
                                    bool* output) {
    const int64_t face = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (face >= face_count) return;
    if (indexed && !active_leaves[face_leaf[face]]) {
        output[face] = false;
        return;
    }
    output[face] = triangle_relevant(vertices, faces + face * 3, planes, plane_count);
}

int launch_blocks(int64_t count) {
    const int64_t blocks = (count + kThreads - 1) / kThreads;
    TORCH_CHECK(blocks <= std::numeric_limits<int>::max(), "query exceeds CUDA grid capacity");
    return int(blocks);
}
}  // namespace

torch::Tensor leaf_mask(torch::Tensor bounds, torch::Tensor planes) {
    check_tensor(bounds, torch::kFloat64, 3, "leaf bounds");
    TORCH_CHECK(bounds.size(1) == 2 && bounds.size(2) == 3, "leaf bounds must be [L,2,3]");
    check_planes(planes, bounds.device());
    c10::cuda::CUDAGuard guard(bounds.device());
    auto output = torch::empty({bounds.size(0)}, bounds.options().dtype(torch::kBool));
    if (bounds.size(0)) {
        leaf_mask_kernel<<<launch_blocks(bounds.size(0)), kThreads, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
            bounds.data_ptr<double>(), bounds.size(0), planes.data_ptr<double>(),
            int(planes.size(0)), output.data_ptr<bool>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}

torch::Tensor triangle_mask(torch::Tensor vertices, torch::Tensor faces,
                            torch::Tensor planes, torch::Tensor active_leaves,
                            torch::Tensor face_leaf, bool indexed) {
    check_tensor(vertices, torch::kFloat64, 2, "vertices");
    check_tensor(faces, torch::kInt64, 2, "faces");
    check_tensor(active_leaves, torch::kBool, 1, "active leaves");
    check_tensor(face_leaf, torch::kInt64, 1, "face leaf mapping");
    TORCH_CHECK(vertices.size(1) == 3 && faces.size(1) == 3,
                "vertices/faces require three coordinates or indices");
    TORCH_CHECK(face_leaf.size(0) == faces.size(0), "face leaf mapping count differs");
    TORCH_CHECK(vertices.device() == faces.device() && vertices.device() == active_leaves.device()
                && vertices.device() == face_leaf.device(), "mesh query device mismatch");
    check_planes(planes, vertices.device());
    c10::cuda::CUDAGuard guard(vertices.device());
    auto output = torch::empty({faces.size(0)}, faces.options().dtype(torch::kBool));
    if (faces.size(0)) {
        triangle_mask_kernel<<<launch_blocks(faces.size(0)), kThreads, 0,
                               at::cuda::getCurrentCUDAStream()>>>(
            vertices.data_ptr<double>(), faces.data_ptr<int64_t>(), faces.size(0),
            planes.data_ptr<double>(), int(planes.size(0)), active_leaves.data_ptr<bool>(),
            face_leaf.data_ptr<int64_t>(), indexed, output.data_ptr<bool>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("leaf_mask", &leaf_mask);
    module.def("triangle_mask", &triangle_mask);
}
