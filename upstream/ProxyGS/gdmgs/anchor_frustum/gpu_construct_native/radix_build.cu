#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <vector>
static void check(const at::Tensor& t,at::ScalarType dt,const at::Device& d){TORCH_CHECK(t.device()==d&&t.scalar_type()==dt&&t.is_contiguous());}
__global__ void codes_kernel(const double* bounds,const double* lo,const double* hi,int64_t n,int64_t* out){
 int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
 double center[3];unsigned q[3];
 for(int a=0;a<3;++a){center[a]=(bounds[7*i+a]+bounds[7*i+3+a])*0.5;
   double ext=fmax(hi[a]-lo[a],1e-20);
   q[a]=(unsigned)fmin(1023.,fmax(0.,(center[a]-lo[a])/ext*1023.));}
 uint64_t code=0;
 for(int bit=0;bit<10;++bit)for(int axis=0;axis<3;++axis)code|=uint64_t((q[axis]>>bit)&1)<<(3*bit+axis);
 out[i]=int64_t((code<<32)|uint64_t(i));
}
at::Tensor morton_codes(at::Tensor bounds,at::Tensor lo,at::Tensor hi){
 TORCH_CHECK(bounds.is_cuda()&&bounds.dim()==2&&bounds.size(1)==7&&lo.numel()==3&&hi.numel()==3);
 c10::cuda::CUDAGuard guard(bounds.device());check(bounds,at::kDouble,bounds.device());check(lo,at::kDouble,bounds.device());check(hi,at::kDouble,bounds.device());
 TORCH_CHECK(bounds.size(0)<(1LL<<32));auto out=at::empty({bounds.size(0)},bounds.options().dtype(at::kLong));
 if(bounds.size(0))codes_kernel<<<(bounds.size(0)+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(bounds.data_ptr<double>(),lo.data_ptr<double>(),hi.data_ptr<double>(),bounds.size(0),out.data_ptr<int64_t>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
__device__ int delta(const int64_t* keys,int n,int i,int j){
 if(j<0||j>=n)return -1;
 unsigned long long a=(unsigned long long)keys[i],b=(unsigned long long)keys[j];
 return a==b?64:__clzll(a^b);
}
__global__ void topology(const int64_t* keys,int n,int* left,int* right,int* parent,int* axis,double* plane,const double* scene_lo,const double* scene_hi){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n-1)return;
 int d=(delta(keys,n,i,i+1)>delta(keys,n,i,i-1))?1:-1;
 int dmin=delta(keys,n,i,i-d),lmax=2;
 while(delta(keys,n,i,i+lmax*d)>dmin)lmax*=2;
 int l=0;for(int t=lmax/2;t>=1;t>>=1)if(delta(keys,n,i,i+(l+t)*d)>dmin)l+=t;
 int j=i+l*d,first=min(i,j),last=max(i,j),common=delta(keys,n,first,last);
 int low=first,high=last-1;
 while(low<high){int mid=(low+high+1)/2;if(delta(keys,n,first,mid)>common)low=mid;else high=mid-1;}
 int split=low,internal=n-1;
 int lc=(split==first)?internal+split:split;
 int rc=(split+1==last)?internal+split+1:split+1;
 left[i]=lc;right[i]=rc;parent[lc]=i;parent[rc]=i;
 int bit=63-common;
 if(bit>=32 && bit<=61){
   int ax=(bit-32)%3,rank=(bit-32)/3;
   unsigned code=(unsigned)(((uint64_t)keys[first])>>32),quant=0;
   for(int b=0;b<10;++b)quant|=((code>>(3*b+ax))&1u)<<b;
   unsigned boundary=((quant>>(rank+1))<<(rank+1))+(1u<<rank);
   axis[i]=ax;plane[i]=scene_lo[ax]+(double(boundary)/1023.)*(scene_hi[ax]-scene_lo[ax]);
 } else {axis[i]=-1;plane[i]=0;}

}
__global__ void leaf_bound(const double* src,const int64_t* sorted_ids,int64_t n,int leaf,int nleaves,double* dst){
 int leaf_id=blockIdx.x*blockDim.x+threadIdx.x;if(leaf_id>=nleaves)return;
 double lo[3]={INFINITY,INFINITY,INFINITY},hi[3]={-INFINITY,-INFINITY,-INFINITY},s=0;
 for(int64_t slot=int64_t(leaf_id)*leaf;slot<n&&slot<int64_t(leaf_id+1)*leaf;++slot){
   int64_t id=sorted_ids[slot];const double* p=src+7*id;
   for(int k=0;k<3;++k){lo[k]=fmin(lo[k],p[k]);hi[k]=fmax(hi[k],p[k+3]);}s=fmax(s,p[6]);
 }
 double* target=dst+7*(nleaves-1+leaf_id);
 for(int k=0;k<3;++k){target[k]=lo[k];target[k+3]=hi[k];}target[6]=s;
}
__global__ void bubble_bounds(int nleaves,const int* left,const int* right,const int* parent,int* arrivals,double* bounds){
 int leaf=blockIdx.x*blockDim.x+threadIdx.x;if(leaf>=nleaves||nleaves<=1)return;
 int node=nleaves-1+leaf;
 while(true){
   __threadfence();int p=parent[node];if(p<0)break;
   int old=atomicAdd(arrivals+p,1);if(old==0)break;
   const double* a=bounds+7*left[p],*b=bounds+7*right[p];double* out=bounds+7*p;
   for(int k=0;k<3;++k){out[k]=fmin(a[k],b[k]);out[k+3]=fmax(a[k+3],b[k+3]);}out[6]=fmax(a[6],b[6]);node=p;
 }
}
std::vector<at::Tensor> build_radix_bvh(at::Tensor bounds,at::Tensor sorted_ids,at::Tensor leaf_keys,at::Tensor scene_lo,at::Tensor scene_hi,int64_t leaf){
 TORCH_CHECK(bounds.is_cuda()&&bounds.dim()==2&&bounds.size(1)==7&&leaf>0);
 c10::cuda::CUDAGuard guard(bounds.device());auto d=bounds.device();check(bounds,at::kDouble,d);check(sorted_ids,at::kLong,d);check(leaf_keys,at::kLong,d);check(scene_lo,at::kDouble,d);check(scene_hi,at::kDouble,d);TORCH_CHECK(scene_lo.numel()==3&&scene_hi.numel()==3);
 int64_t n=bounds.size(0),L=leaf_keys.numel();TORCH_CHECK(n>0&&L==(n+leaf-1)/leaf&&L<(1LL<<30)&&sorted_ids.numel()==n);
 auto intopt=bounds.options().dtype(at::kInt);auto left=at::full({L-1},-1,intopt),right=at::full({L-1},-1,intopt),parent=at::full({2*L-1},-1,intopt);
 auto axis=at::full({L-1},-1,intopt);auto plane=at::zeros({L-1},bounds.options());
 auto nodes=at::empty({2*L-1,7},bounds.options());auto arrivals=at::zeros({L-1},intopt);auto stream=at::cuda::getCurrentCUDAStream();
 if(L>1)topology<<<(L-1+255)/256,256,0,stream>>>(leaf_keys.data_ptr<int64_t>(),L,left.data_ptr<int>(),right.data_ptr<int>(),parent.data_ptr<int>(),axis.data_ptr<int>(),plane.data_ptr<double>(),scene_lo.data_ptr<double>(),scene_hi.data_ptr<double>());
 leaf_bound<<<(L+255)/256,256,0,stream>>>(bounds.data_ptr<double>(),sorted_ids.data_ptr<int64_t>(),n,leaf,L,nodes.data_ptr<double>());
 if(L>1)bubble_bounds<<<(L+255)/256,256,0,stream>>>(L,left.data_ptr<int>(),right.data_ptr<int>(),parent.data_ptr<int>(),arrivals.data_ptr<int>(),nodes.data_ptr<double>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {nodes,left,right,parent,axis,plane};
}
