#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "classify.h"
#include "warp_classify.cuh"
#include "../native/bounds.h"
#include <vector>
__global__ void init_frontier(int64_t* q,int32_t* count) {q[0]=1;count[0]=1;}
__global__ void step_frontier(const double* nodes,const double* camera,const int64_t* current,
    const int32_t* ncurrent,int64_t* next,int32_t* nnext,uint8_t* state,int32_t* active,
    int level,int64_t base) {
    constexpr unsigned FULL=0xffffffff;
    int lane=threadIdx.x&31;
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    int length=ncurrent[0];
    if(warp>=length)return;
    if(warp==0 && lane==0)active[level]=length;
    int64_t node=current[warp];
    double e[6]={0,0,0,0,0,0};
    int pre=2;
    if(lane==0)pre=support_envelope(nodes+7*node,camera,e);
    pre=__shfl_sync(FULL,pre,0);
    int result=pre;
    if(pre==3) {
        double envelope[6];
        #pragma unroll
        for(int j=0;j<6;++j)envelope[j]=__shfl_sync(FULL,e[j],0);
        double minimum=0,maximum=0;
        if(lane<6)plane_interval(lane,envelope,camera,minimum,maximum);
        unsigned unknown=__ballot_sync(FULL,lane<6 && (!isfinite(minimum)||!isfinite(maximum)));
        unsigned outside_mask=__ballot_sync(FULL,lane<6 && maximum<0);
        unsigned inside_mask=__ballot_sync(FULL,lane<6 && minimum>=0);
        result=unknown?2:(outside_mask?0:((inside_mask&0x3f)==0x3f?1:2));
    }
    if(lane==0) {
        state[node]=(uint8_t)result;
        if(result==2 && node<base) {
            int slot=atomicAdd(nnext,2);
            next[slot]=2*node;next[slot+1]=2*node+1;
        }
    }
}
// Exactly one warp owns a spatial leaf. Lane 0 resolves ancestor state;
// lanes 0..31 then write the corresponding original-row masks in parallel.
__global__ void materialize(const double* bounds,const double* camera,const int64_t* order,
    const uint8_t* state,int64_t slots,int64_t count,int64_t base,int leaf,bool* mask) {
    constexpr unsigned FULL=0xffffffff;
    int lane=threadIdx.x&31;
    int64_t leaf_index=(int64_t(blockIdx.x)*blockDim.x+threadIdx.x)/32;
    if(leaf_index>=base)return;
    int resolved=2;
    if(lane==0) {
        int64_t node=base+leaf_index;
        while(node) {
            uint8_t s=state[node];
            if(s==0||s==1){resolved=s;break;}
            if(node>=base && s==2){resolved=3;break;}
            node>>=1;
        }
    }
    resolved=__shfl_sync(FULL,resolved,0);
    for(int slot=lane;slot<leaf;slot+=32) {
        int64_t j=leaf_index*leaf+slot;
        if(j>=slots)continue;
        int64_t id=order[j];if(id<0||id>=count)continue;
        mask[id]=(resolved==1)||(resolved==3 && !outside(bounds+7*id,camera));
    }
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
        step_frontier<<<(capacity+7)/8,256,0,stream>>>(nodes.data_ptr<double>(),camera.data_ptr<double>(),cur.data_ptr<int64_t>(),ncur.data_ptr<int32_t>(),next.data_ptr<int64_t>(),nnext.data_ptr<int32_t>(),state.data_ptr<uint8_t>(),active.data_ptr<int32_t>(),level,base);
    }
    if(order.numel())materialize<<<(base+7)/8,256,0,stream>>>(bounds.data_ptr<double>(),camera.data_ptr<double>(),order.data_ptr<int64_t>(),state.data_ptr<uint8_t>(),order.numel(),bounds.size(0),base,leaf,mask.data_ptr<bool>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {mask,state,active};
}
