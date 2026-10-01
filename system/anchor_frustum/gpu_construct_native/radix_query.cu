#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "../hierarchy_native_warp/warp_classify.cuh"
#include "../native/bounds.h"
#include <cstdint>
#include <vector>
__global__ void init_queue(int* q,int* count){q[0]=0;count[0]=1;}
__global__ void step_binary(const double* nodes,const double* camera,const int* left,const int* right,
 const int* axis,const double* plane,const double* eye,const int* current,const int* ncurrent,
 int* next,int* nnext,uint8_t* state,int* active,int level,int internal){
 constexpr unsigned FULL=0xffffffff;
 int lane=threadIdx.x&31,slot=(blockIdx.x*blockDim.x+threadIdx.x)/32;
 int length=ncurrent[0];if(slot>=length)return;
 if(slot==0&&lane==0)active[level]=length;
 int node=current[slot],pre=2;double e[6]={0,0,0,0,0,0};
 if(lane==0)pre=support_envelope(nodes+7*node,camera,e);
 pre=__shfl_sync(FULL,pre,0);int result=pre;
 if(pre==3){
   double v[6];
   #pragma unroll
   for(int k=0;k<6;++k)v[k]=__shfl_sync(FULL,e[k],0);
   double mn=0,mx=0;if(lane<6)plane_interval(lane,v,camera,mn,mx);
   unsigned bad=__ballot_sync(FULL,lane<6&&(!isfinite(mn)||!isfinite(mx)));
   unsigned reject=__ballot_sync(FULL,lane<6&&mx<0);
   unsigned inside=__ballot_sync(FULL,lane<6&&mn>=0);
   result=bad?2:(reject?0:((inside&0x3f)==0x3f?1:2));
 }
 if(lane==0){
   state[node]=(uint8_t)result;
   if(result==2&&node<internal){
     int a=left[node],b=right[node];
     if(axis!=nullptr){int dim=axis[node];if(dim>=0&&eye[dim]>plane[node]){int t=a;a=b;b=t;}}
     int dest=atomicAdd(nnext,2);next[dest]=a;next[dest+1]=b;
   }
 }
}
__global__ void gather_binary(const double* bounds,const double* camera,const int64_t* order,const int* parent,
 const uint8_t* states,int64_t n,int leaf,int internal,bool* mask){
 int64_t j=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(j>=n)return;
 int64_t id=order[j];int node=internal+j/leaf;
 bool keep=true;
 while(node>=0){
   uint8_t s=states[node];
   if(s==0){keep=false;break;}
   if(s==1){keep=true;break;}
   if(node>=internal&&s==2){keep=!outside(bounds+7*id,camera);break;}
   node=parent[node];
 }
 mask[id]=keep;
}
std::vector<at::Tensor> query_radix(at::Tensor bounds,at::Tensor nodes,at::Tensor left,at::Tensor right,
 at::Tensor parent,at::Tensor sorted_ids,at::Tensor camera,at::Tensor axis,at::Tensor plane,
 int64_t leaf,bool kd,at::Tensor eye){
 TORCH_CHECK(bounds.is_cuda()&&bounds.scalar_type()==at::kDouble&&bounds.dim()==2&&bounds.size(1)==7&&leaf>0);
 auto d=bounds.device();c10::cuda::CUDAGuard guard(d);
 for(auto x:{nodes,plane,camera,eye})TORCH_CHECK(x.device()==d&&x.scalar_type()==at::kDouble&&x.is_contiguous());
 for(auto x:{left,right,parent,axis})TORCH_CHECK(x.device()==d&&x.scalar_type()==at::kInt&&x.is_contiguous());
 TORCH_CHECK(sorted_ids.device()==d&&sorted_ids.scalar_type()==at::kLong&&sorted_ids.is_contiguous());
 int64_t N=bounds.size(0),L=(N+leaf-1)/leaf,I=L-1;
 TORCH_CHECK(N>0&&sorted_ids.numel()==N&&nodes.size(0)==2*L-1&&left.numel()==I&&right.numel()==I&&parent.numel()==2*L-1&&axis.numel()==I&&plane.numel()==I&&camera.numel()==23&&eye.numel()==3);
 auto ints=bounds.options().dtype(at::kInt);auto queue0=at::empty({L},ints),queue1=at::empty({L},ints),n0=at::zeros({1},ints),n1=at::zeros({1},ints);
 auto states=at::full({2*L-1},255,bounds.options().dtype(at::kByte)),active=at::zeros({64},ints),mask=at::zeros({N},bounds.options().dtype(at::kBool));
 auto stream=at::cuda::getCurrentCUDAStream();init_queue<<<1,1,0,stream>>>(queue0.data_ptr<int>(),n0.data_ptr<int>());
 for(int depth=0;depth<64;++depth){
   int64_t cap=std::min<int64_t>(int64_t(1)<<std::min(depth,30),L);
   auto &q=(depth&1)?queue1:queue0,&n=(depth&1)?n1:n0;
   auto &next=(depth&1)?queue0:queue1,&nn=(depth&1)?n0:n1;
   cudaMemsetAsync(nn.data_ptr<int>(),0,sizeof(int),stream);
   step_binary<<<(cap+7)/8,256,0,stream>>>(nodes.data_ptr<double>(),camera.data_ptr<double>(),left.data_ptr<int>(),right.data_ptr<int>(),kd?axis.data_ptr<int>():nullptr,plane.data_ptr<double>(),eye.data_ptr<double>(),q.data_ptr<int>(),n.data_ptr<int>(),next.data_ptr<int>(),nn.data_ptr<int>(),states.data_ptr<uint8_t>(),active.data_ptr<int>(),depth,I);
 }
 gather_binary<<<(N+255)/256,256,0,stream>>>(bounds.data_ptr<double>(),camera.data_ptr<double>(),sorted_ids.data_ptr<int64_t>(),parent.data_ptr<int>(),states.data_ptr<uint8_t>(),N,leaf,I,mask.data_ptr<bool>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {mask,states,active};
}

__global__ void dense_mask_kernel(const double* bounds,const double* camera,int64_t n,bool* mask){
 int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
 if(i<n)mask[i]=!outside(bounds+7*i,camera);
}
at::Tensor gpu_dense_mask(at::Tensor bounds,at::Tensor camera){
 TORCH_CHECK(bounds.is_cuda()&&bounds.scalar_type()==at::kDouble&&camera.device()==bounds.device()&&camera.scalar_type()==at::kDouble&&bounds.is_contiguous()&&camera.is_contiguous()&&bounds.dim()==2&&bounds.size(1)==7&&camera.numel()==23);
 c10::cuda::CUDAGuard guard(bounds.device());auto out=at::empty({bounds.size(0)},bounds.options().dtype(at::kBool));
 if(bounds.size(0))dense_mask_kernel<<<(bounds.size(0)+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(bounds.data_ptr<double>(),camera.data_ptr<double>(),bounds.size(0),out.data_ptr<bool>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
