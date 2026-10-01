#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "../hierarchy_native_warp/warp_classify.cuh"
#include "../native/bounds.h"
#include <cstdint>
#include <vector>
__device__ void atomicMinDouble(double* address,double value){
 auto raw=(unsigned long long*)address;unsigned long long old=*raw,expected;
 while(value<__longlong_as_double(old)){expected=old;old=atomicCAS(raw,expected,__double_as_longlong(value));if(old==expected)break;}
}
__device__ void atomicMaxDouble(double* address,double value){
 auto raw=(unsigned long long*)address;unsigned long long old=*raw,expected;
 while(value>__longlong_as_double(old)){expected=old;old=atomicCAS(raw,expected,__double_as_longlong(value));if(old==expected)break;}
}
__global__ void init_oct(double* bounds,int64_t m){
 int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=m)return;
 for(int a=0;a<3;++a){bounds[7*i+a]=INFINITY;bounds[7*i+a+3]=-INFINITY;}bounds[7*i+6]=0;
}
__device__ int find_key(const int64_t* keys,int m,int64_t value){
 int lo=0,hi=m;while(lo<hi){int mid=(lo+hi)/2;if(keys[mid]<value)lo=mid+1;else hi=mid;}return lo<m&&keys[lo]==value?lo:-1;
}
__global__ void link_oct(const int64_t* keys,int m,int* parent,int* children,int* child_count){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=m)return;
 int depth=int((uint64_t)keys[i]>>32);if(!depth)return;
 uint32_t prefix=(uint32_t)keys[i];int64_t pk=(int64_t(depth-1)<<32)|(prefix>>3);
 int p=find_key(keys,m,pk);if(p<0)return;parent[i]=p;
 children[8*p+(prefix&7u)]=i;atomicAdd(child_count+p,1);
}
__global__ void fill_oct_leaves(const double* anchor_bounds,const int64_t* sorted_ids,const int64_t* sorted_leaf_cell,int64_t n,double* nodes){
 int64_t j=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(j>=n)return;
 int64_t id=sorted_ids[j],cell=sorted_leaf_cell[j];const double* src=anchor_bounds+7*id;double* dest=nodes+7*cell;
 for(int k=0;k<3;++k){atomicMinDouble(dest+k,src[k]);atomicMaxDouble(dest+k+3,src[k+3]);}atomicMaxDouble(dest+6,src[6]);
}
__global__ void reduce_oct(const int64_t* keys,int m,const int* parent,const int* children,const int* child_count,int* arrivals,double* bounds){
 int node=blockIdx.x*blockDim.x+threadIdx.x;if(node>=m||int((uint64_t)keys[node]>>32)!=10)return;
 while(true){
  __threadfence();int p=parent[node];if(p<0)break;
  int old=atomicAdd(arrivals+p,1);if(old!=child_count[p]-1)break;
  double lo[3]={INFINITY,INFINITY,INFINITY},hi[3]={-INFINITY,-INFINITY,-INFINITY},s=0;
  for(int k=0;k<8;++k){int ch=children[8*p+k];if(ch<0)continue;const double* b=bounds+7*ch;
    for(int a=0;a<3;++a){lo[a]=fmin(lo[a],b[a]);hi[a]=fmax(hi[a],b[a+3]);}s=fmax(s,b[6]);}
  double* dst=bounds+7*p;for(int a=0;a<3;++a){dst[a]=lo[a];dst[a+3]=hi[a];}dst[6]=s;
  node=p;
 }
}
std::vector<at::Tensor> build_octree(at::Tensor bounds,at::Tensor sorted_ids,at::Tensor node_keys,at::Tensor sorted_leaf_cell){
 TORCH_CHECK(bounds.is_cuda()&&bounds.scalar_type()==at::kDouble&&bounds.dim()==2&&bounds.size(1)==7);
 auto d=bounds.device();c10::cuda::CUDAGuard guard(d);
 for(auto x:{sorted_ids,node_keys,sorted_leaf_cell})TORCH_CHECK(x.device()==d&&x.scalar_type()==at::kLong&&x.is_contiguous());
 int64_t n=bounds.size(0),m=node_keys.numel();TORCH_CHECK(n>0&&m>0&&m<(1LL<<30)&&sorted_ids.numel()==n&&sorted_leaf_cell.numel()==n);
 auto intopt=bounds.options().dtype(at::kInt);auto nodes=at::empty({m,7},bounds.options());
 auto parent=at::full({m},-1,intopt),children=at::full({m,8},-1,intopt);
 auto child_count=at::zeros({m},intopt),arrivals=at::zeros({m},intopt);
 auto stream=at::cuda::getCurrentCUDAStream();
 init_oct<<<(m+255)/256,256,0,stream>>>(nodes.data_ptr<double>(),m);
 link_oct<<<(m+255)/256,256,0,stream>>>(node_keys.data_ptr<int64_t>(),m,parent.data_ptr<int>(),children.data_ptr<int>(),child_count.data_ptr<int>());
 fill_oct_leaves<<<(n+255)/256,256,0,stream>>>(bounds.data_ptr<double>(),sorted_ids.data_ptr<int64_t>(),sorted_leaf_cell.data_ptr<int64_t>(),n,nodes.data_ptr<double>());
 reduce_oct<<<(m+255)/256,256,0,stream>>>(node_keys.data_ptr<int64_t>(),m,parent.data_ptr<int>(),children.data_ptr<int>(),child_count.data_ptr<int>(),arrivals.data_ptr<int>(),nodes.data_ptr<double>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {nodes,parent,children,child_count};
}
__global__ void init_q(int* q,int* count){q[0]=0;count[0]=1;}
__global__ void step_oct(const double* nodes,const double* camera,const int* children,const int* child_count,
 const int* current,const int* ncurrent,int* next,int* nnext,uint8_t* states,int* active,int level){
 constexpr unsigned FULL=0xffffffff;int lane=threadIdx.x&31,slot=(blockIdx.x*blockDim.x+threadIdx.x)/32;
 int length=ncurrent[0];if(slot>=length)return;
 if(slot==0&&lane==0)active[level]=length;
 int node=current[slot],pre=2;double e[6]={0,0,0,0,0,0};
 if(lane==0)pre=support_envelope(nodes+7*node,camera,e);
 pre=__shfl_sync(FULL,pre,0);int result=pre;
 if(pre==3){double v[6];
  #pragma unroll
  for(int k=0;k<6;++k)v[k]=__shfl_sync(FULL,e[k],0);
  double mn=0,mx=0;if(lane<6)plane_interval(lane,v,camera,mn,mx);
  unsigned bad=__ballot_sync(FULL,lane<6&&(!isfinite(mn)||!isfinite(mx)));
  unsigned reject=__ballot_sync(FULL,lane<6&&mx<0);
  unsigned inside=__ballot_sync(FULL,lane<6&&mn>=0);
  result=bad?2:(reject?0:((inside&0x3f)==0x3f?1:2));
 }
 if(lane==0){states[node]=result;
  if(result==2 && child_count[node]>0){int dest=atomicAdd(nnext,child_count[node]);int offset=0;
    for(int k=0;k<8;++k){int ch=children[8*node+k];if(ch>=0)next[dest+offset++]=ch;}}
 }
}
__global__ void gather_oct(const double* bounds,const double* camera,const int64_t* leaf_for_original,
 const int* parent,const int* child_count,const uint8_t* states,int64_t n,bool* mask){
 int64_t id=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(id>=n)return;
 int node=(int)leaf_for_original[id];bool keep=true;
 while(node>=0){uint8_t s=states[node];
   if(s==0){keep=false;break;}if(s==1){keep=true;break;}
   if(child_count[node]==0&&s==2){keep=!outside(bounds+7*id,camera);break;}node=parent[node];
 }
 mask[id]=keep;
}
std::vector<at::Tensor> query_octree(at::Tensor bounds,at::Tensor nodes,at::Tensor parent,at::Tensor children,
 at::Tensor child_count,at::Tensor leaf_for_original,at::Tensor camera){
 TORCH_CHECK(bounds.is_cuda()&&bounds.scalar_type()==at::kDouble&&bounds.dim()==2&&bounds.size(1)==7);
 auto d=bounds.device();c10::cuda::CUDAGuard guard(d);int64_t n=bounds.size(0),m=nodes.size(0);
 for(auto x:{nodes,camera})TORCH_CHECK(x.device()==d&&x.scalar_type()==at::kDouble&&x.is_contiguous());
 for(auto x:{parent,children,child_count})TORCH_CHECK(x.device()==d&&x.scalar_type()==at::kInt&&x.is_contiguous());
 TORCH_CHECK(leaf_for_original.device()==d&&leaf_for_original.scalar_type()==at::kLong&&leaf_for_original.is_contiguous()&&leaf_for_original.numel()==n&&camera.numel()==23);
 TORCH_CHECK(nodes.dim()==2&&nodes.size(1)==7&&children.dim()==2&&children.size(0)==m&&children.size(1)==8&&parent.numel()==m&&child_count.numel()==m);
 auto opts=bounds.options().dtype(at::kInt);auto q0=at::empty({m},opts),q1=at::empty({m},opts),n0=at::zeros({1},opts),n1=at::zeros({1},opts);
 auto states=at::full({m},255,bounds.options().dtype(at::kByte)),active=at::zeros({11},opts),mask=at::zeros({n},bounds.options().dtype(at::kBool));
 auto stream=at::cuda::getCurrentCUDAStream();init_q<<<1,1,0,stream>>>(q0.data_ptr<int>(),n0.data_ptr<int>());
 int64_t capacity=1;
 for(int depth=0;depth<11;++depth){
  auto &q=(depth&1)?q1:q0,&ncur=(depth&1)?n1:n0,&next=(depth&1)?q0:q1,&nnext=(depth&1)?n0:n1;
  cudaMemsetAsync(nnext.data_ptr<int>(),0,sizeof(int),stream);
  step_oct<<<(capacity+7)/8,256,0,stream>>>(nodes.data_ptr<double>(),camera.data_ptr<double>(),children.data_ptr<int>(),child_count.data_ptr<int>(),q.data_ptr<int>(),ncur.data_ptr<int>(),next.data_ptr<int>(),nnext.data_ptr<int>(),states.data_ptr<uint8_t>(),active.data_ptr<int>(),depth);
  capacity=std::min<int64_t>(m,capacity*8);
 }
 gather_oct<<<(n+255)/256,256,0,stream>>>(bounds.data_ptr<double>(),camera.data_ptr<double>(),leaf_for_original.data_ptr<int64_t>(),parent.data_ptr<int>(),child_count.data_ptr<int>(),states.data_ptr<uint8_t>(),n,mask.data_ptr<bool>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {mask,states,active};
}
