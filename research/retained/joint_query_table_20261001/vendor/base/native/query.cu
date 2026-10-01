#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <cstdint>
#include <vector>
#include "bounds.h"
#include "warp_classify.cuh"
#include "triangle_predicate.cuh"
__device__ int anchor_class(const double* b,const double* c,int lane){
 const unsigned FULL=0xffffffff;
 int pre=2;double e[6]={};
 if(lane==0)pre=b[0]>b[3]?0:support_envelope(b,c,e);
 pre=__shfl_sync(FULL,pre,0);if(pre!=3)return pre;
 double v[6];for(int k=0;k<6;++k)v[k]=__shfl_sync(FULL,e[k],0);
 double mn=0,mx=0;if(lane<6)plane_interval(lane,v,c,mn,mx);
 unsigned bad=__ballot_sync(FULL,lane<6&&(!isfinite(mn)||!isfinite(mx)));
 unsigned out=__ballot_sync(FULL,lane<6&&mx<0);
 unsigned in=__ballot_sync(FULL,lane<6&&mn>=0);
 return bad?2:(out?0:((in&63)==63?1:2));
}
__device__ int mesh_class(const double* b,const double* ps,int np,int lane){
 if(b[0]>b[3])return 0;
 double mn=0,mx=0,err=0;
 if(lane<np){const double* p=ps+8*lane;mn=mx=p[3];double mag=fabs(p[3]);err=p[7];
  for(int a=0;a<3;++a){double x=p[a]*b[a],y=p[a]*b[a+3];mn+=fmin(x,y);mx+=fmax(x,y);mag+=fmax(fabs(x),fabs(y));err+=p[a+4]*fmax(fabs(b[a]),fabs(b[a+3]));}
  err+=64*DBL_EPSILON*(mag+1);
 }
 unsigned bad=__ballot_sync(0xffffffff,lane<np&&(!isfinite(mn)||!isfinite(mx)||!isfinite(err)));
 unsigned out=__ballot_sync(0xffffffff,lane<np&&mx < -err);
 unsigned in=__ballot_sync(0xffffffff,lane<np&&mn >= err);
 return bad?2:(out?0:((in&((1u<<np)-1))==((1u<<np)-1)?1:2));
}
__global__ void init(int* q,int* n){q[0]=0;n[0]=1;}
__global__ void step(const double* an,const double* mn,const double* cp,const double* planes,int np,
 const int* left,const int* right,const int* parent,const int* axis,const double* split,const double* eye,
 const int* q,const int* n,int* next,int* nn,uint8_t* states,int* active,int level,int internal,int mode){
 int lane=threadIdx.x&31,slot=(blockIdx.x*blockDim.x+threadIdx.x)/32;
 if(slot>=n[0])return;int node=q[slot],p=parent[node];
 int a=(mode&1)?2:0,m=(mode&2)?2:0;
 if(p>=0){if(a==2&&states[2*p]!=2)a=states[2*p];if(m==2&&states[2*p+1]!=2)m=states[2*p+1];}
 if(a==2)a=anchor_class(an+7*node,cp,lane);
 if(m==2)m=mesh_class(mn+7*node,planes,np,lane);
 if(lane==0){
  if(slot==0)active[level]=n[0];states[2*node]=a;states[2*node+1]=m;
  if((a==2||m==2)&&node<internal){
   int l=left[node],r=right[node],d=axis[node];if(d>=0&&eye[d]>split[node]){int t=l;l=r;r=t;}
   int pos=atomicAdd(nn,2);next[pos]=l;next[pos+1]=r;
  }
 }
}
__global__ void gather(const double* ab,const double* cp,const double* vs,const int64_t* fs,
 const double* planes,int np,const int64_t* order,const int* parent,const uint8_t* states,
 int64_t total,int64_t na,int internal,int mode,bool* am,bool* mm){
 int64_t slot=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(slot>=total)return;
 int64_t id=order[slot];bool isa=id<na;if(!(mode&(isa?1:2)))return;
 int type=isa?0:1,node=internal+slot/32;bool keep=true;
 while(node>=0){uint8_t s=states[2*node+type];
  if(s==0){keep=false;break;}if(s==1){keep=true;break;}
  if(node>=internal&&s==2){keep=isa?!outside(ab+7*id,cp):triangle_relevant(vs,fs+3*(id-na),planes,np);break;}
  node=parent[node];
 }
 if(isa)am[id]=keep;else mm[id-na]=keep;
}
std::vector<at::Tensor> joint_query(at::Tensor ab,at::Tensor vs,at::Tensor fs,at::Tensor an,at::Tensor mn,
 at::Tensor left,at::Tensor right,at::Tensor parent,at::Tensor order,at::Tensor cp,at::Tensor planes,
 at::Tensor axis,at::Tensor split,at::Tensor eye,int64_t mode,int64_t levels){
 auto d=ab.device();c10::cuda::CUDAGuard guard(d);
 TORCH_CHECK(ab.is_cuda()&&mode>=1&&mode<=3&&levels>0&&levels<=64);
 for(auto x:{ab,vs,an,mn,cp,planes,split,eye})TORCH_CHECK(x.device()==d&&x.scalar_type()==at::kDouble&&x.is_contiguous());
 for(auto x:{left,right,parent,axis})TORCH_CHECK(x.device()==d&&x.scalar_type()==at::kInt&&x.is_contiguous());
 for(auto x:{fs,order})TORCH_CHECK(x.device()==d&&x.scalar_type()==at::kLong&&x.is_contiguous());
 int64_t na=ab.size(0),nm=fs.size(0),n=na+nm,L=(n+31)/32,I=L-1;
 TORCH_CHECK(n>0&&order.numel()==n&&an.size(0)==2*L-1&&mn.sizes()==an.sizes()&&an.size(1)==7&&ab.size(1)==7&&vs.size(1)==3&&fs.size(1)==3&&left.numel()==I&&right.numel()==I&&parent.numel()==2*L-1&&axis.numel()==I&&split.numel()==I&&cp.numel()==23&&planes.size(1)==8&&planes.size(0)>=5&&planes.size(0)<=6&&eye.numel()==3);
 auto ints=ab.options().dtype(at::kInt);
 auto q0=at::empty({L},ints),q1=at::empty({L},ints),n0=at::zeros({1},ints),n1=at::zeros({1},ints);
 auto states=at::full({2*L-1,2},255,ints.dtype(at::kByte)),active=at::zeros({levels},ints);
 auto am=at::zeros({(mode&1)?na:0},ints.dtype(at::kBool)),mm=at::zeros({(mode&2)?nm:0},ints.dtype(at::kBool));
 auto stream=at::cuda::getCurrentCUDAStream();init<<<1,1,0,stream>>>(q0.data_ptr<int>(),n0.data_ptr<int>());
 for(int level=0;level<levels;++level){
  int64_t cap=std::min<int64_t>(int64_t(1)<<std::min(level,30),L);
  auto &q=(level&1)?q1:q0,&nq=(level&1)?n1:n0,&next=(level&1)?q0:q1,&nn=(level&1)?n0:n1;
  cudaMemsetAsync(nn.data_ptr<int>(),0,sizeof(int),stream);
  step<<<(cap+7)/8,256,0,stream>>>(an.data_ptr<double>(),mn.data_ptr<double>(),cp.data_ptr<double>(),planes.data_ptr<double>(),planes.size(0),left.data_ptr<int>(),right.data_ptr<int>(),parent.data_ptr<int>(),axis.data_ptr<int>(),split.data_ptr<double>(),eye.data_ptr<double>(),q.data_ptr<int>(),nq.data_ptr<int>(),next.data_ptr<int>(),nn.data_ptr<int>(),states.data_ptr<uint8_t>(),active.data_ptr<int>(),level,I,mode);
 }
 gather<<<(n+255)/256,256,0,stream>>>(ab.data_ptr<double>(),cp.data_ptr<double>(),vs.data_ptr<double>(),fs.data_ptr<int64_t>(),planes.data_ptr<double>(),planes.size(0),order.data_ptr<int64_t>(),parent.data_ptr<int>(),states.data_ptr<uint8_t>(),n,na,I,mode,am.data_ptr<bool>(),mm.data_ptr<bool>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {am,mm,states,active};
}
