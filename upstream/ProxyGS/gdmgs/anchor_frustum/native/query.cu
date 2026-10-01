#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "bounds.h"
#include <vector>
// One CUDA block queries a coarse node (four spatial leaves), one warp per
// surviving leaf, and all lanes then test different anchor support bounds.
// The same CPU-built hierarchy is used; no per-anchor work for rejected nodes.
__global__ void query_kernel(const double* b,const double* n,const int64_t* ids,const double* c,int64_t N,int base,int leaf,bool dense,bool sorted,bool* out,int64_t* counts) {
    int task=blockIdx.x*blockDim.x+threadIdx.x;
    if(dense) {
        if(task==0) counts[1]=N;
        if(task<N) out[task]=!outside(b+7*task,c);
        return;
    }
    __shared__ int coarse_reject;
    if(threadIdx.x==0) {
        int coarse=max(1,(base+int(blockIdx.x)*4)/4);
        coarse_reject=outside(n+7*coarse,c);
        atomicAdd((unsigned long long*)counts,1ULL);
        if(coarse_reject) atomicAdd((unsigned long long*)(counts+2),1ULL);
    }
    __syncthreads();
    if(coarse_reject) return;
    int leaf_idx=blockIdx.x*4+threadIdx.x/32;
    if(leaf_idx>=base) return;
    int lane=threadIdx.x%32, reject=0;
    if(lane==0) {
        reject=outside(n+7*(base+leaf_idx),c);
        atomicAdd((unsigned long long*)counts,1ULL);
        if(reject) atomicAdd((unsigned long long*)(counts+2),1ULL);
    }
    reject=__shfl_sync(0xffffffff,reject,0);
    if(reject) return;
    int64_t begin=int64_t(leaf_idx)*leaf,end=min(N,begin+leaf);
    if(lane==0 && end>begin) atomicAdd((unsigned long long*)(counts+1),(unsigned long long)(end-begin));
    for(int64_t j=begin+lane;j<end;j+=32) {
        auto id=ids[j];out[id]=!outside(b+7*(sorted?j:id),c);
    }
}
std::vector<at::Tensor> gpu_query(at::Tensor b,at::Tensor n,at::Tensor ids,at::Tensor c,int64_t base,int64_t leaf,bool dense,bool sorted) {
    TORCH_CHECK(b.is_cuda() && n.device()==b.device() && ids.device()==b.device() && c.device()==b.device());
    TORCH_CHECK(b.scalar_type()==at::kDouble && n.scalar_type()==at::kDouble && c.scalar_type()==at::kDouble && ids.scalar_type()==at::kLong);
    TORCH_CHECK(b.is_contiguous() && n.is_contiguous() && ids.is_contiguous() && c.is_contiguous());
    TORCH_CHECK(b.dim()==2 && b.size(1)==7 && n.dim()==2 && n.size(1)==7 && c.numel()==23 && ids.numel()==b.size(0));
    TORCH_CHECK(base>0 && base<(1LL<<29) && leaf>0 && n.size(0)==2*base);
    c10::cuda::CUDAGuard guard(b.device());
    auto out=at::zeros({b.size(0)},b.options().dtype(at::kBool));
    auto counts=at::zeros({3},b.options().dtype(at::kLong));
    int blocks=dense?(b.size(0)+127)/128:(base+3)/4;
    if(blocks) query_kernel<<<blocks,128,0,at::cuda::getCurrentCUDAStream()>>>(b.data_ptr<double>(),n.data_ptr<double>(),ids.data_ptr<int64_t>(),c.data_ptr<double>(),b.size(0),base,leaf,dense,sorted,out.data_ptr<bool>(),counts.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out,counts};
}

__global__ void block_kernel(const double* n,const double* c,int64_t base,bool* out) {
    int64_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<base)out[i]=!outside(n+7*(base+i),c);
}
at::Tensor gpu_blocks(at::Tensor n,at::Tensor c,int64_t base) {
    TORCH_CHECK(n.is_cuda() && c.device()==n.device() && n.scalar_type()==at::kDouble && c.scalar_type()==at::kDouble);
    TORCH_CHECK(n.is_contiguous() && c.is_contiguous() && n.dim()==2 && n.size(1)==7 && n.size(0)==2*base && c.numel()==23 && base>0);
    c10::cuda::CUDAGuard guard(n.device());
    auto out=at::empty({base},n.options().dtype(at::kBool));
    block_kernel<<<(base+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(n.data_ptr<double>(),c.data_ptr<double>(),base,out.data_ptr<bool>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

__global__ void refine_kernel(const double* sorted,const int64_t* ids,const double* c,const bool* flags,int64_t N,int leaf,bool* out) {
    int64_t j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<N && flags[j/leaf])out[ids[j]]=!outside(sorted+7*j,c);
}
at::Tensor gpu_refine(at::Tensor sorted,at::Tensor ids,at::Tensor c,at::Tensor flags,int64_t leaf) {
    TORCH_CHECK(sorted.is_cuda() && ids.device()==sorted.device() && c.device()==sorted.device() && flags.device()==sorted.device());
    TORCH_CHECK(sorted.scalar_type()==at::kDouble && ids.scalar_type()==at::kLong && c.scalar_type()==at::kDouble && flags.scalar_type()==at::kBool);
    TORCH_CHECK(sorted.is_contiguous() && ids.is_contiguous() && c.is_contiguous() && flags.is_contiguous());
    TORCH_CHECK(sorted.dim()==2 && sorted.size(1)==7 && ids.numel()==sorted.size(0) && c.numel()==23 && leaf>0 && flags.numel()*leaf>=ids.numel());
    c10::cuda::CUDAGuard guard(sorted.device());
    auto out=at::zeros({ids.numel()},flags.options());
    if(ids.numel())refine_kernel<<<(ids.numel()+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(sorted.data_ptr<double>(),ids.data_ptr<int64_t>(),c.data_ptr<double>(),flags.data_ptr<bool>(),ids.numel(),leaf,out.data_ptr<bool>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
