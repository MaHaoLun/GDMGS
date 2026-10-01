#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
using I = int64_t;
enum Counter { VISITED_NODES, EMPTY_NODES, CERTIFIED_NODES, ANCHOR_CHECKS,
    INTERVAL_KEPT_CHECKS, CERTIFIED_ANCHORS, UNBOUNDED, NEAR_PLANE, UNKNOWN_CELLS,
    DEPTH_FAIL, OUTSIDE, EARLY_UNKNOWN, RECT_QUERIES, MAXIMAL_NODES, COUNTER_COUNT };
struct Camera {
    double w[16],d[4],pad_x,pad_y,near_z,margin,rotation_norm;
    I h,width,lx,ly;
};
__device__ double mn(double a,double b){return b<a?b:a;}
__device__ double mx(double a,double b){return a<b?b:a;}
__device__ void count(I* counters,int slot){atomicAdd(reinterpret_cast<unsigned long long*>(counters+slot),1ULL);}
static void tensor(const at::Tensor& t,at::ScalarType type,int dims,const at::Device& dev,const char* name){
    TORCH_CHECK(t.device()==dev&&t.scalar_type()==type&&t.dim()==dims&&t.is_contiguous(),name," has wrong device/dtype/shape/contiguity");
}
static Camera camera(at::Tensor table,at::Tensor w2c,const std::vector<double>& domain,
                     const std::vector<I>& size,double pad,double near,double margin){
    TORCH_CHECK(table.is_cuda()&&table.is_contiguous()&&table.dim()==4&&
                (table.scalar_type()==at::kFloat||table.scalar_type()==at::kDouble),"table requires CUDA float32/64 [LY,LX,H,W]");
    TORCH_CHECK(w2c.device().is_cpu()&&w2c.scalar_type()==at::kDouble&&w2c.is_contiguous()&&w2c.sizes()==at::IntArrayRef({4,4}),"w2c requires CPU float64 [4,4]");
    TORCH_CHECK(domain.size()==4&&size.size()==2&&size[0]>0&&size[1]>0,"invalid domain/image size");
    Camera c{};for(int i=0;i<16;++i){c.w[i]=w2c.data_ptr<double>()[i];TORCH_CHECK(std::isfinite(c.w[i]),"nonfinite w2c");}
    TORCH_CHECK(c.w[12]==0&&c.w[13]==0&&c.w[14]==0&&c.w[15]==1,"w2c must be affine");
    for(int i=0;i<4;++i){c.d[i]=domain[i];TORCH_CHECK(std::isfinite(c.d[i]),"nonfinite domain");}
    TORCH_CHECK(c.d[0]<c.d[1]&&c.d[2]<c.d[3]&&std::isfinite(c.d[1]-c.d[0])&&std::isfinite(c.d[3]-c.d[2]),"invalid angular domain");
    TORCH_CHECK(std::isfinite(pad)&&pad>=0&&std::isfinite(near)&&near>0&&std::isfinite(margin)&&margin>=0,"invalid support settings");
    c.ly=table.size(0);c.lx=table.size(1);c.h=table.size(2);c.width=table.size(3);
    TORCH_CHECK(c.h>0&&c.width>0&&c.h<=INT32_MAX&&c.width<=INT32_MAX,"invalid table extent");
    TORCH_CHECK(c.ly==I(std::floor(std::log2(double(c.h))))+1&&c.lx==I(std::floor(std::log2(double(c.width))))+1,"invalid sparse table levels");
    c.pad_x=pad*(c.d[1]-c.d[0])/size[0];c.pad_y=pad*(c.d[3]-c.d[2])/size[1];c.near_z=near;c.margin=margin;
    double norm_sq=0;for(int j=0;j<3;++j){double row=0;for(int k=0;k<3;++k){double dot=0;for(int l=0;l<3;++l)dot+=c.w[j*4+l]*c.w[k*4+l];row+=std::abs(dot);}norm_sq=std::max(norm_sq,row);}
    c.rotation_norm=std::sqrt(norm_sq)*(1+64*std::numeric_limits<float>::epsilon());return c;
}
__device__ bool project_box(const double* box,const Camera& c,double& xmin,double& xmax,
                            double& ymin,double& ymax,double& zmin){
    xmin=ymin=zmin=INFINITY;xmax=ymax=-INFINITY;
    for(int corner=0;corner<8;++corner){
        double p[3]={box[(corner&1)?3:0],box[(corner&2)?4:1],box[(corner&4)?5:2]},qlo[3],qhi[3];
        for(int j=0;j<3;++j){double q=c.w[j*4+3],scale=fabs(q);for(int k=0;k<3;++k){q+=c.w[j*4+k]*p[k];scale+=fabs(c.w[j*4+k]*p[k]);}
            double error=(64*0x1p-23+16*0x1p-52)*mx(1.0,scale);qlo[j]=q-error;qhi[j]=q+error;}
        if(!isfinite(qlo[2])||!isfinite(qhi[2])||qlo[2]<=c.near_z)return false;zmin=mn(zmin,qlo[2]);
        for(int k=0;k<2;++k){double low=qlo[k]/(qlo[k]>=0?qhi[2]:qlo[2]);double high=qhi[k]/(qhi[k]>=0?qlo[2]:qhi[2]);
            if(!isfinite(low)||!isfinite(high))return false;if(k==0){xmin=mn(xmin,low);xmax=mx(xmax,high);}else{ymin=mn(ymin,low);ymax=mx(ymax,high);}}
    }return true;
}
template<typename T> __device__ double rectangle_max(const T* table,const Camera& c,I x0,I x1,I y0,I y1){
    unsigned int nx=unsigned(x1-x0+1),ny=unsigned(y1-y0+1);I kx=31-__clz(nx),ky=31-__clz(ny);
    I xb=x1-(I(1)<<kx)+1,yb=y1-(I(1)<<ky)+1,base=(ky*c.lx+kx)*c.h*c.width;
    double a=double(table[base+y0*c.width+x0]),b=double(table[base+y0*c.width+xb]);
    double e=double(table[base+yb*c.width+x0]),f=double(table[base+yb*c.width+xb]);return mx(mx(a,b),mx(e,f));
}
template<typename T> __device__ bool certify(const double* box,const double* centers,double radius,
                                            bool known,const T* table,const Camera& c,I* counts){
    if(!known){count(counts,UNBOUNDED);return false;}
    double midpoint[3],q[3];for(int j=0;j<3;++j)midpoint[j]=centers[j]*.5+centers[j+3]*.5;
    for(int j=0;j<3;++j)q[j]=c.w[j*4]*midpoint[0]+c.w[j*4+1]*midpoint[1]+c.w[j*4+2]*midpoint[2]+c.w[j*4+3];
    if(isfinite(q[2])&&q[2]>c.near_z){double x=q[0]/q[2],y=q[1]/q[2];if(x>=c.d[0]&&x<=c.d[1]&&y>=c.d[2]&&y<=c.d[3]){
        I ix=min(c.width-1,I((x-c.d[0])/(c.d[1]-c.d[0])*c.width)),iy=min(c.h-1,I((y-c.d[2])/(c.d[3]-c.d[2])*c.h));
        if(!isfinite(double(table[iy*c.width+ix]))){count(counts,EARLY_UNKNOWN);count(counts,UNKNOWN_CELLS);return false;}}}
    double xl,xh,yl,yh,zl,cxl,cxh,cyl,cyh,czl;
    if(!project_box(box,c,xl,xh,yl,yh,zl)||!project_box(centers,c,cxl,cxh,cyl,cyh,czl)){count(counts,NEAR_PLANE);return false;}
    double xlimit=mx(fabs(c.d[0]),fabs(c.d[1]))+.15*(c.d[1]-c.d[0]),ylimit=mx(fabs(c.d[2]),fabs(c.d[3]))+.15*(c.d[3]-c.d[2]);
    double xr=mn(mx(fabs(cxl),fabs(cxh)),xlimit),yr=mn(mx(fabs(cyl),fabs(cyh)),ylimit),dx=radius*c.rotation_norm*sqrt(1+xr*xr)/czl,dy=radius*c.rotation_norm*sqrt(1+yr*yr)/czl;
    xl=nextafter(mn(xl,cxl-dx)-c.pad_x,-INFINITY);xh=nextafter(mx(xh,cxh+dx)+c.pad_x,INFINITY);
    yl=nextafter(mn(yl,cyl-dy)-c.pad_y,-INFINITY);yh=nextafter(mx(yh,cyh+dy)+c.pad_y,INFINITY);
    if(xh<c.d[0]||xl>c.d[1]||yh<c.d[2]||yl>c.d[3]){count(counts,OUTSIDE);return false;}
    xl=mx(xl,c.d[0]);xh=mn(xh,c.d[1]);yl=mx(yl,c.d[2]);yh=mn(yh,c.d[3]);
    I x0=max(I(0),min(c.width-1,I(floor((xl-c.d[0])/(c.d[1]-c.d[0])*c.width)))),x1=max(I(0),min(c.width-1,I(floor((xh-c.d[0])/(c.d[1]-c.d[0])*c.width))));
    I y0=max(I(0),min(c.h-1,I(floor((yl-c.d[2])/(c.d[3]-c.d[2])*c.h)))),y1=max(I(0),min(c.h-1,I(floor((yh-c.d[2])/(c.d[3]-c.d[2])*c.h))));
    count(counts,RECT_QUERIES);double depth=rectangle_max(table,c,x0,x1,y0,y1);if(!isfinite(depth)){count(counts,UNKNOWN_CELLS);return false;}
    double error=128*0x1p-23*mx(1.0,mx(fabs(zl),fabs(depth)));
    if(depth+c.margin+error<nextafter(zl,-INFINITY))return true;count(counts,DEPTH_FAIL);return false;
}
template<typename T> __global__ void horizontal(const T* prev,T* out,I h,I w,I offset){
    I id=I(blockIdx.x)*blockDim.x+threadIdx.x;if(id>=h*w)return;I x=id%w;out[id]=x+offset<w?T(mx(double(prev[id]),double(prev[id+offset]))):T(INFINITY);
}
template<typename T> __global__ void vertical(const T* prev,T* out,I levels,I h,I w,I offset){
    I id=I(blockIdx.x)*blockDim.x+threadIdx.x;if(id>=levels*h*w)return;I y=(id/w)%h;out[id]=y+offset<h?T(mx(double(prev[id]),double(prev[id+offset*w]))):T(INFINITY);
}
at::Tensor anchor_sparse_max(at::Tensor depth){
    TORCH_CHECK(depth.is_cuda()&&depth.is_contiguous()&&depth.dim()==2&&(depth.scalar_type()==at::kFloat||depth.scalar_type()==at::kDouble),"depth requires CUDA contiguous float32/64 [H,W]");
    const c10::cuda::CUDAGuard guard(depth.device());I h=depth.size(0),w=depth.size(1);TORCH_CHECK(h>0&&w>0&&h<=INT32_MAX&&w<=INT32_MAX,"invalid depth extent");
    I ly=I(std::floor(std::log2(double(h))))+1,lx=I(std::floor(std::log2(double(w))))+1;auto out=at::empty({ly,lx,h,w},depth.options());auto stream=at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemcpyAsync(out.data_ptr(),depth.data_ptr(),depth.nbytes(),cudaMemcpyDeviceToDevice,stream));
    AT_DISPATCH_FLOATING_TYPES(depth.scalar_type(),"anchor_sparse_max",[&]{auto* p=out.data_ptr<scalar_t>();for(I k=1;k<lx;++k)horizontal<<<(h*w+255)/256,256,0,stream>>>(p+(k-1)*h*w,p+k*h*w,h,w,I(1)<<(k-1));
        for(I k=1;k<ly;++k)vertical<<<(lx*h*w+255)/256,256,0,stream>>>(p+(k-1)*lx*h*w,p+k*lx*h*w,lx,h,w,I(1)<<(k-1));});C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
template<typename T> __global__ void anchors_kernel(const double* bounds,const double* centers,const double* radii,const uint8_t* known,
        const I* ids,I count_ids,const bool* removed_rank,const I* rank,const T* table,Camera c,uint8_t* removed,I* counts){
    I slot=I(blockIdx.x)*blockDim.x+threadIdx.x;if(slot>=count_ids)return;I row=ids[slot];
    if(removed_rank&&removed_rank[rank[row]]){removed[slot]=1;count(counts,INTERVAL_KEPT_CHECKS);count(counts,CERTIFIED_ANCHORS);return;}
    count(counts,ANCHOR_CHECKS);bool hidden=certify(bounds+row*6,centers+row*6,radii[row],known[row],table,c,counts);removed[slot]=uint8_t(hidden);if(hidden)count(counts,CERTIFIED_ANCHORS);
}
static void support_inputs(at::Tensor bounds,at::Tensor centers,at::Tensor radii,at::Tensor known,at::Tensor counts,const at::Device& device){
    tensor(bounds,at::kDouble,2,device,"bounds");tensor(centers,at::kDouble,2,device,"centers");tensor(radii,at::kDouble,1,device,"radii");tensor(known,at::kByte,1,device,"known");tensor(counts,at::kLong,1,device,"counts");
    TORCH_CHECK(bounds.size(1)==6&&centers.sizes()==bounds.sizes()&&radii.numel()==bounds.size(0)&&known.numel()==bounds.size(0)&&counts.numel()==COUNTER_COUNT,"support dimensions disagree");
}
at::Tensor anchor_gpu_certify(at::Tensor bounds,at::Tensor centers,at::Tensor radii,at::Tensor known,at::Tensor ids,at::Tensor removed_rank,
        at::Tensor rank,at::Tensor table,at::Tensor w2c,std::vector<double> domain,std::vector<I> size,double pad,double near,double margin,at::Tensor counts){
    Camera c=camera(table,w2c,domain,size,pad,near,margin);auto device=table.device();const c10::cuda::CUDAGuard guard(device);support_inputs(bounds,centers,radii,known,counts,device);
    tensor(ids,at::kLong,1,device,"ids");tensor(rank,at::kLong,1,device,"rank");tensor(removed_rank,at::kBool,1,device,"removed_rank");
    TORCH_CHECK(rank.numel()==bounds.size(0)&&(removed_rank.numel()==0||removed_rank.numel()==rank.numel()),"row binding size mismatch");
    auto output=at::empty({ids.numel()},ids.options().dtype(at::kByte));auto stream=at::cuda::getCurrentCUDAStream();
    if(ids.numel())AT_DISPATCH_FLOATING_TYPES(table.scalar_type(),"anchor_gpu_certify",[&]{anchors_kernel<<<(ids.numel()+255)/256,256,0,stream>>>(bounds.data_ptr<double>(),centers.data_ptr<double>(),radii.data_ptr<double>(),known.data_ptr<uint8_t>(),ids.data_ptr<I>(),ids.numel(),removed_rank.numel()?removed_rank.data_ptr<bool>():nullptr,rank.data_ptr<I>(),table.data_ptr<scalar_t>(),c,output.data_ptr<uint8_t>(),counts.data_ptr<I>());});
    C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}
template<typename T> __global__ void nodes_kernel(const double* bounds,const double* centers,const double* radii,const uint8_t* known,
    const I* intervals,const I* prefix,I n,const T* table,Camera c,uint8_t* flags,I* counts){
    I id=I(blockIdx.x)*blockDim.x+threadIdx.x;if(id>=n)return;count(counts,VISITED_NODES);flags[id]=0;
    if(prefix[intervals[id*2]]==prefix[intervals[id*2+1]]){count(counts,EMPTY_NODES);return;}
    bool hidden=certify(bounds+id*6,centers+id*6,radii[id],known[id],table,c,counts);flags[id]=uint8_t(hidden);if(hidden)count(counts,CERTIFIED_NODES);
}
at::Tensor node_gpu_certify(at::Tensor bounds,at::Tensor centers,at::Tensor radii,at::Tensor known,at::Tensor intervals,at::Tensor prefix,
    at::Tensor table,at::Tensor w2c,std::vector<double> domain,std::vector<I> size,double pad,double near,double margin,at::Tensor counts){
    Camera c=camera(table,w2c,domain,size,pad,near,margin);auto device=table.device();const c10::cuda::CUDAGuard guard(device);support_inputs(bounds,centers,radii,known,counts,device);
    tensor(intervals,at::kLong,2,device,"intervals");tensor(prefix,at::kLong,1,device,"fov prefix");TORCH_CHECK(intervals.size(0)==bounds.size(0)&&intervals.size(1)==2,"node interval mismatch");
    auto flags=at::empty({bounds.size(0)},known.options());auto stream=at::cuda::getCurrentCUDAStream();
    if(bounds.size(0))AT_DISPATCH_FLOATING_TYPES(table.scalar_type(),"node_gpu_certify",[&]{nodes_kernel<<<(bounds.size(0)+255)/256,256,0,stream>>>(bounds.data_ptr<double>(),centers.data_ptr<double>(),radii.data_ptr<double>(),known.data_ptr<uint8_t>(),intervals.data_ptr<I>(),prefix.data_ptr<I>(),bounds.size(0),table.data_ptr<scalar_t>(),c,flags.data_ptr<uint8_t>(),counts.data_ptr<I>());});
    C10_CUDA_KERNEL_LAUNCH_CHECK();return flags;
}
__global__ void maximal_kernel(const uint8_t* flags,const I* parents,const I* intervals,I n,int32_t* delta,I* counts){
    I id=I(blockIdx.x)*blockDim.x+threadIdx.x;if(id>=n||!flags[id])return;I p=parents[id];while(p>=0){if(flags[p])return;p=parents[p];}
    atomicAdd(delta+intervals[id*2],1);atomicAdd(delta+intervals[id*2+1],-1);count(counts,MAXIMAL_NODES);
}
at::Tensor anchor_maximal_intervals(at::Tensor flags,at::Tensor parents,at::Tensor intervals,I n,at::Tensor counts){
    TORCH_CHECK(flags.is_cuda(),"flags requires CUDA");auto device=flags.device();const c10::cuda::CUDAGuard guard(device);
    tensor(flags,at::kByte,1,device,"node flags");tensor(parents,at::kLong,1,device,"parents");tensor(intervals,at::kLong,2,device,"intervals");tensor(counts,at::kLong,1,device,"counts");
    TORCH_CHECK(n>=0&&parents.numel()==flags.numel()&&intervals.size(0)==flags.numel()&&intervals.size(1)==2&&counts.numel()==COUNTER_COUNT,"invalid interval dimensions");
    auto delta=at::zeros({n+1},flags.options().dtype(at::kInt));if(flags.numel())maximal_kernel<<<(flags.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(flags.data_ptr<uint8_t>(),parents.data_ptr<I>(),intervals.data_ptr<I>(),flags.numel(),delta.data_ptr<int32_t>(),counts.data_ptr<I>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return delta;
}
