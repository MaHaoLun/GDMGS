#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "classify.h"
#include "../native/bounds.h"
#include <vector>
__global__ void init_frontier(int64_t* q,int32_t* count) {q[0]=1;count[0]=1;}
__global__ void step_frontier(const double* nodes,const double* camera,const int64_t* current,
    const int32_t* ncurrent,int64_t* next,int32_t* nnext,uint8_t* state,int32_t* active,
    int level,int64_t base) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int length=ncurrent[0];
    if(i>=length)return;
    if(i==0)active[level]=length;
    int64_t node=current[i];
    int result=classify_six_planes(nodes+7*node,camera);
    state[node]=(uint8_t)result;
    if(result==2 && node<base) {
        int slot=atomicAdd(nnext,2);
        next[slot]=2*node;next[slot+1]=2*node+1;
    }
}
__global__ void materialize(const double* bounds,const double* camera,const int64_t* order,
    const uint8_t* state,int64_t slots,int64_t count,int64_t base,int leaf,bool* mask) {
    int64_t j=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(j>=slots)return;
    int64_t node=base+j/leaf,id=order[j];if(id<0 || id>=count)return;bool keep=true;
    while(node) {
        uint8_t s=state[node];
        if(s==0){keep=false;break;}
        if(s==1){keep=true;break;}
        if(node>=base && s==2){keep=!outside(bounds+7*id,camera);break;}
        node>>=1;
    }
    mask[id]=keep;
}
std::vector<at::Tensor> frontier_query(at::Tensor bounds,at::Tensor nodes,at::Tensor order,at::Tensor camera,int64_t base,int64_t leaf) {
    TORCH_CHECK(bounds.is_cuda() && nodes.device()==bounds.device() && order.device()==bounds.device() && camera.device()==bounds.device());
    TORCH_CHECK(bounds.scalar_type()==at::kDouble && nodes.scalar_type()==at::kDouble && camera.scalar_type()==at::kDouble && order.scalar_type()==at::kLong);
    TORCH_CHECK(bounds.is_contiguous() && nodes.is_contiguous() && order.is_contiguous() && camera.is_contiguous());
    TORCH_CHECK(bounds.dim()==2 && bounds.size(1)==7 && nodes.dim()==2 && nodes.size(1)==7 && (order.numel()==bounds.size(0) || order.numel()==base*leaf) && camera.numel()==23 && nodes.size(0)==2*base && base>0 && base<(1LL<<29) && leaf>0);
    c10::cuda::CUDAGuard guard(bounds.device());auto stream=at::cuda::getCurrentCUDAStream();
    auto q0=at::empty({base},order.options()),q1=at::empty({base},order.options());
    auto n0=at::zeros({1},order.options().dtype(at::kInt)),n1=at::zeros({1},order.options().dtype(at::kInt));
    auto state=at::full({2*base},255,bounds.options().dtype(at::kByte));
    int levels=0;for(int64_t x=base;x;x>>=1)++levels;
    auto active=at::zeros({levels},order.options().dtype(at::kInt));
    auto mask=at::zeros({bounds.size(0)},bounds.options().dtype(at::kBool));
    init_frontier<<<1,1,0,stream>>>(q0.data_ptr<int64_t>(),n0.data_ptr<int32_t>());
    for(int level=0;level<levels;++level) {
        int64_t capacity=std::min<int64_t>(int64_t(1)<<level,base);
        auto &cur=(level%2)?q1:q0,&ncur=(level%2)?n1:n0;
        auto &next=(level%2)?q0:q1,&nnext=(level%2)?n0:n1;
        cudaMemsetAsync(nnext.data_ptr<int32_t>(),0,sizeof(int32_t),stream);
        step_frontier<<<(capacity+127)/128,128,0,stream>>>(nodes.data_ptr<double>(),camera.data_ptr<double>(),cur.data_ptr<int64_t>(),ncur.data_ptr<int32_t>(),next.data_ptr<int64_t>(),nnext.data_ptr<int32_t>(),state.data_ptr<uint8_t>(),active.data_ptr<int32_t>(),level,base);
    }
    if(order.numel())materialize<<<(order.numel()+255)/256,256,0,stream>>>(bounds.data_ptr<double>(),camera.data_ptr<double>(),order.data_ptr<int64_t>(),state.data_ptr<uint8_t>(),order.numel(),bounds.size(0),base,leaf,mask.data_ptr<bool>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {mask,state,active};
}
