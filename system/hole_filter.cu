#include <cmath>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

__global__ void hole_keep_kernel(const double* bounds, const int64_t* ids,
                                 const double* planes, const int* counts,
                                 int64_t n, int holes, int plane_stride, bool* keep) {
    int64_t slot = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (slot >= n) return;
    const double* b = bounds + ids[slot] * 7;
    bool result = true;
    for (int h = 0; h < holes; ++h) {
        bool inside = true;
        for (int p = 0; p < counts[h]; ++p) {
            const double* q = planes + (h * plane_stride + p) * 4;
            double minimum = q[3];
            for (int axis = 0; axis < 3; ++axis)
                minimum += q[axis] * (q[axis] >= 0.0 ? b[axis] : b[axis + 3])
                           - 3.0 * b[6] * fabs(q[axis]);
            if (!(minimum > 1e-8)) { inside = false; break; }
        }
        if (inside) { result = false; break; }
    }
    keep[slot] = result;
}

at::Tensor hole_keep(at::Tensor bounds, at::Tensor ids,
                     at::Tensor planes, at::Tensor counts) {
    TORCH_CHECK(bounds.is_cuda() && ids.is_cuda() && planes.is_cuda() && counts.is_cuda());
    TORCH_CHECK(bounds.is_contiguous() && ids.is_contiguous() && planes.is_contiguous() && counts.is_contiguous());
    TORCH_CHECK(bounds.scalar_type() == at::kDouble && ids.scalar_type() == at::kLong);
    TORCH_CHECK(planes.scalar_type() == at::kDouble && counts.scalar_type() == at::kInt);
    TORCH_CHECK(bounds.dim() == 2 && bounds.size(1) == 7 && ids.dim() == 1);
    TORCH_CHECK(planes.dim() == 3 && planes.size(1) > 0 && planes.size(2) == 4);
    TORCH_CHECK(counts.dim() == 1 && counts.size(0) == planes.size(0));
    c10::cuda::CUDAGuard guard(bounds.device());
    auto keep = at::empty({ids.numel()}, bounds.options().dtype(at::kBool));
    auto stream = at::cuda::getCurrentCUDAStream();
    int64_t n = ids.numel();
    if (n) hole_keep_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        bounds.data_ptr<double>(), ids.data_ptr<int64_t>(),
        planes.data_ptr<double>(), counts.data_ptr<int>(), n,
        int(planes.size(0)), int(planes.size(1)), keep.data_ptr<bool>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return keep;
}

__global__ void lod_kernel(const double* position,const double* radius2,const int32_t* levels,
                           const double* eye,double res2,int64_t n,bool* keep) {
    const int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n)return;
    const double dx=__dsub_rn(position[3*i],eye[0]);
    const double dy=__dsub_rn(position[3*i+1],eye[1]);
    const double dz=__dsub_rn(position[3*i+2],eye[2]);
    const double d2=__dmul_rn(__dadd_rn(__dadd_rn(__dmul_rn(dx,dx),__dmul_rn(dy,dy)),__dmul_rn(dz,dz)),res2);
    keep[i]=levels[i]==0 || d2<radius2[i] || (d2==radius2[i] && (levels[i]%2)==0);
}

at::Tensor lod_mask(at::Tensor position,at::Tensor radius2,at::Tensor levels,at::Tensor eye,double resolution) {
    TORCH_CHECK(position.is_cuda() && radius2.device()==position.device() && levels.device()==position.device() && eye.device()==position.device());
    TORCH_CHECK(position.is_contiguous() && radius2.is_contiguous() && levels.is_contiguous() && eye.is_contiguous());
    TORCH_CHECK(position.scalar_type()==at::kDouble && radius2.scalar_type()==at::kDouble && levels.scalar_type()==at::kInt && eye.scalar_type()==at::kDouble);
    TORCH_CHECK(position.dim()==2 && position.size(1)==3 && radius2.numel()==position.size(0) && levels.numel()==position.size(0) && eye.numel()==3);
    TORCH_CHECK(resolution>0 && std::isfinite(resolution));
    c10::cuda::CUDAGuard guard(position.device());
    auto result=at::empty({position.size(0)},position.options().dtype(at::kBool));
    if(position.size(0))lod_kernel<<<(position.size(0)+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
        position.data_ptr<double>(),radius2.data_ptr<double>(),levels.data_ptr<int32_t>(),eye.data_ptr<double>(),
        resolution*resolution,position.size(0),result.data_ptr<bool>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hole_keep", &hole_keep);
    m.def("lod_mask", &lod_mask);
}
