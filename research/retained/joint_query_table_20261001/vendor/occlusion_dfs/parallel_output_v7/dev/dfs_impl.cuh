// Snapshot depth is immutable during traversal. Empty coverage remains +inf.
__device__ bool hidden(const double* b,const double* c,const float* hz,const int64_t* meta,int nl){
 if(!hz || nl==0 || b[0]>b[3])return false;
 double e[6];if(support_envelope(b,c,e)!=3)return false;
 double xl=1e300,xh=-1e300,yl=1e300,yh=-1e300;
 for(int i=0;i<2;i++)for(int j=0;j<2;j++){
  double z=e[4+j],x=e[i]/z*c[16]+c[18],y=e[2+i]/z*c[17]+c[19];
  xl=fmin(xl,x);xh=fmax(xh,x);yl=fmin(yl,y);yh=fmax(yh,y);
 }
 if(!isfinite(xl)||!isfinite(xh)||!isfinite(yl)||!isfinite(yh))return false;
 // Off-screen portions cannot affect this image; near-plane crossing stays kept.
 int x0=(int)floor(fmax(0.,fmin(c[20]-1,xl))),x1=(int)ceil(fmax(0.,fmin(c[20]-1,xh)));
 int y0=(int)floor(fmax(0.,fmin(c[21]-1,yl))),y1=(int)ceil(fmax(0.,fmin(c[21]-1,yh)));
 if(xh<0||yh<0||xl>=c[20]||yl>=c[21])return false;
 int level=0;while(level+1<nl && (1<<(level+3))<max(x1-x0+1,y1-y0+1))level++;
 int tile=1<<(level+3),w=meta[3*level+1];int64_t off=meta[3*level];
 float farthest=0;
 for(int y=y0/tile;y<=y1/tile;y++)for(int x=x0/tile;x<=x1/tile;x++){
  float z=hz[off+int64_t(y)*w+x];if(!isfinite(z))return false;farthest=fmaxf(farthest,z);
 }
 double nearest=e[4]-3*b[6]*sqrt(c[22]);
 if(!isfinite(nearest)||nearest<=.01)return false;
 return nearest>double(farthest)+2e-5*(1+fabs(nearest));
}
__device__ double near_mesh(const double* b,const double* cp){
 if(b[0]>b[3])return INFINITY;double z=cp[11];
 for(int k=0;k<3;k++)z+=fmin(cp[8+k]*b[k],cp[8+k]*b[k+3]);return z;
}
__global__ void spans_kernel(const int* parent,int* lo,int* hi,int leaves){
 int leaf=blockIdx.x*blockDim.x+threadIdx.x;if(leaf>=leaves)return;
 for(int node=leaves-1+leaf;node>=0;node=parent[node]){atomicMin(lo+node,leaf*32);atomicMax(hi+node,(leaf+1)*32);}
}
__global__ void dfs_kernel(const double* ab,const double* vs,const int64_t* fs,
 const double* an,const double* mn,const double* cp,const double* planes,int np,
 const int* left,const int* right,const int* parent,const int64_t* order,const int* lo,const int* hi,
 const int* roots,int nr,const uint8_t* prefix,const float* hz,const int64_t* meta,int nl,
 int internal,int64_t total,int64_t na,bool* am,bool* mm,int64_t* counts,int64_t* chunk_lengths,uint8_t* range_states){
 int lane=threadIdx.x&31,warp=threadIdx.x/32,task=(blockIdx.x*blockDim.x+threadIdx.x)/32;
 if(task>=nr)return;
 __shared__ int sn[1][64],sa[1][64],sm[1][64];
 int root=roots[task];int top=1,node=root,p=parent[node],aa=p<0?2:prefix[2*p],mmode=p<0?2:prefix[2*p+1];
 if(lane==0){sn[warp][0]=node;sa[warp][0]=aa;sm[warp][0]=mmode;}__syncwarp();
 unsigned long long visits=0,culled=0,highwater=0,overflow=0,anchor_tests=0,mesh_tests=0;
 while(top){
  --top;node=sn[warp][top];aa=sa[warp][top];mmode=sm[warp][top];
  if(node==root&&prefix[2*node]!=255)aa=prefix[2*node];
  else if(aa==2)aa=anchor_class(an+7*node,cp,lane);
  if(node==root&&prefix[2*node+1]!=255)mmode=prefix[2*node+1];
  else if(mmode==2)mmode=mesh_class(mn+7*node,planes,np,lane);
  int h=0;if(lane==0 && aa && nl)h=hidden(an+7*node,cp,hz,meta,nl);
  h=__shfl_sync(0xffffffff,h,0);if(h)aa=0;
  if(lane==0){visits++;culled+=h;highwater=max(highwater,(unsigned long long)(top+1));}
  if(!aa&&!mmode)continue;
  if(node>=internal || (aa!=2 && mmode!=2)){
   int64_t finish=min(int64_t(hi[node]),total),length=finish-lo[node];
   if(length<=256){
    // Small terminals use the already resident warp; large terminals keep
    // the independent massively parallel output path.
    for(int64_t slot=lo[node]+lane;slot<finish;slot+=32){
     int64_t id=order[slot];
     if(id<na){if(aa){bool keep=aa==1||!outside(ab+7*id,cp);if(keep&&nl)keep=!hidden(ab+7*id,cp,hz,meta,nl);am[id]=keep;if(aa==2)anchor_tests++;}}
     else if(mmode){mm[id-na]=mmode==1||triangle_relevant(vs,fs+3*(id-na),planes,np);if(mmode==2)mesh_tests++;}
    }
   }else if(lane==0){
    chunk_lengths[node]=(length+255)/256;
    range_states[2*node]=aa;range_states[2*node+1]=mmode;
   }
  }else{
   int l=left[node],r=right[node];if(near_mesh(mn+7*r,cp)<near_mesh(mn+7*l,cp)){int t=l;l=r;r=t;}
   if(top+2>64){if(lane==0)overflow++;break;}
   if(lane==0){sn[warp][top]=r;sa[warp][top]=aa;sm[warp][top]=mmode;sn[warp][top+1]=l;sa[warp][top+1]=aa;sm[warp][top+1]=mmode;}
   top+=2;__syncwarp();
  }
 }
 for(int offset=16;offset;offset>>=1){
  anchor_tests+=__shfl_down_sync(0xffffffff,anchor_tests,offset);
  mesh_tests+=__shfl_down_sync(0xffffffff,mesh_tests,offset);
 }
 if(lane==0){
  if(anchor_tests)atomicAdd((unsigned long long*)(counts+2),anchor_tests);
  if(mesh_tests)atomicAdd((unsigned long long*)(counts+3),mesh_tests);
  atomicAdd((unsigned long long*)(counts+0),visits);
  if(culled)atomicAdd((unsigned long long*)(counts+1),culled);
  atomicMax((unsigned long long*)(counts+4),highwater);
  if(overflow)atomicAdd((unsigned long long*)(counts+5),overflow);
 }
}
// Populate the direct chunk-to-terminal map in parallel (one warp per node).
// Its entries are disjoint and the consumer reads only the emitted prefix.
__global__ void map_chunk_owners(const int64_t* prefix,int nodes,int* owner){
 int lane=threadIdx.x&31,node=(blockIdx.x*blockDim.x+threadIdx.x)/32;
 if(node>=nodes)return;
 int64_t begin=node?prefix[node-1]:0,end=prefix[node];
 for(int64_t chunk=begin+lane;chunk<end;chunk+=32)owner[chunk]=node;
}
// Independent 256-object chunks expose parallelism even when the whole tree
// terminates at one Keep node. GPU prefix sums allocate chunks without a host
// count read or an O(number of objects) emission loop in the DFS warp.
__global__ void parallel_output(const double* ab,const double* vs,const int64_t* fs,
 const double* cp,const double* planes,int np,const int64_t* order,const int* lo,const int* hi,
 const int64_t* chunk_prefix,const int* owner,const uint8_t* range_states,int nodes,int64_t total,int64_t na,
 const float* hz,const int64_t* meta,int nl,bool* am,bool* mm,int64_t* counts){
 int64_t chunks=chunk_prefix[nodes-1];
 if(blockIdx.x==0&&threadIdx.x==0){counts[6]=chunks;counts[7]=min(chunks,int64_t(gridDim.x));}
 __shared__ int64_t begin,end;__shared__ int aa,mmode;
 __shared__ unsigned long long warp_anchor[8],warp_mesh[8];
 unsigned long long anchor_tests=0,mesh_tests=0;
 for(int64_t chunk=blockIdx.x;chunk<chunks;chunk+=gridDim.x){
  if(threadIdx.x==0){
   int l=owner[chunk];
   int64_t prior=l?chunk_prefix[l-1]:0;
   begin=int64_t(lo[l])+(chunk-prior)*256;end=min(int64_t(hi[l]),total);
   aa=range_states[2*l];mmode=range_states[2*l+1];
  }
  __syncthreads();
  int64_t slot=begin+threadIdx.x;
  if(slot<end){
   int64_t id=order[slot];
   if(id<na){
    if(aa){bool keep=aa==1||!outside(ab+7*id,cp);if(keep&&nl)keep=!hidden(ab+7*id,cp,hz,meta,nl);am[id]=keep;
     if(aa==2)anchor_tests++;}
   }else if(mmode){
    mm[id-na]=mmode==1||triangle_relevant(vs,fs+3*(id-na),planes,np);
    if(mmode==2)mesh_tests++;
   }
  }
  __syncthreads();
 }
 // Exact integer diagnostic reduction: two global additions per output block,
 // instead of one contended atomic for each leaf primitive.
 int lane=threadIdx.x&31,warp=threadIdx.x/32;
 for(int offset=16;offset;offset>>=1){
  anchor_tests+=__shfl_down_sync(0xffffffff,anchor_tests,offset);
  mesh_tests+=__shfl_down_sync(0xffffffff,mesh_tests,offset);
 }
 if(lane==0){warp_anchor[warp]=anchor_tests;warp_mesh[warp]=mesh_tests;}
 __syncthreads();
 if(threadIdx.x==0){
  unsigned long long a=0,m=0;for(int w=0;w<8;w++){a+=warp_anchor[w];m+=warp_mesh[w];}
  if(a)atomicAdd((unsigned long long*)(counts+2),a);
  if(m)atomicAdd((unsigned long long*)(counts+3),m);
 }
}
// Emit early terminal prefix subtrees as independent DFS tasks too.
__global__ void roots_kernel(const uint8_t* states,const int* parent,int n,int cut,bool* flags){
 int node=blockIdx.x*blockDim.x+threadIdx.x;if(node>=n)return;
 int p=parent[node];int a=states[2*node],m=states[2*node+1];
 bool unvisited=a==255&&m==255;
 bool terminal=!unvisited&&(node>=n/2||(a!=2&&m!=2))&&(a||m);
 flags[node]=(unvisited&&(p<0||states[2*p]==2||states[2*p+1]==2))||terminal;
}
__global__ void filter_kernel(const double* ab,const double* cp,const float* hz,const int64_t* meta,int nl,const int64_t* ids,int64_t n,bool* keep){
 int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)keep[i]=!hidden(ab+7*ids[i],cp,hz,meta,nl);
}
std::vector<at::Tensor> prefix_query(at::Tensor an,at::Tensor mn,at::Tensor left,at::Tensor right,at::Tensor parent,
 at::Tensor cp,at::Tensor planes,at::Tensor axis,at::Tensor split,at::Tensor eye,int64_t cut){
 c10::cuda::CUDAGuard guard(an.device());auto stream=at::cuda::getCurrentCUDAStream();
 TORCH_CHECK(cut>=0&&cut<=20);int L=parent.numel()/2+1,I=L-1;
 auto ints=an.options().dtype(at::kInt);auto q0=at::empty({L},ints),q1=at::empty({L},ints),n0=at::zeros({1},ints),n1=at::zeros({1},ints);
 auto states=at::full({2*L-1,2},255,ints.dtype(at::kByte)),active=at::zeros({cut},ints);
 init<<<1,1,0,stream>>>(q0.data_ptr<int>(),n0.data_ptr<int>());
 for(int level=0;level<cut;level++){
  auto &q=(level&1)?q1:q0,&nq=(level&1)?n1:n0,&next=(level&1)?q0:q1,&nn=(level&1)?n0:n1;
  cudaMemsetAsync(nn.data_ptr<int>(),0,sizeof(int),stream);int cap=std::min(1<<level,L);
  step<<<(cap+7)/8,256,0,stream>>>(an.data_ptr<double>(),mn.data_ptr<double>(),cp.data_ptr<double>(),planes.data_ptr<double>(),planes.size(0),left.data_ptr<int>(),right.data_ptr<int>(),parent.data_ptr<int>(),axis.data_ptr<int>(),split.data_ptr<double>(),eye.data_ptr<double>(),q.data_ptr<int>(),nq.data_ptr<int>(),next.data_ptr<int>(),nn.data_ptr<int>(),states.data_ptr<uint8_t>(),active.data_ptr<int>(),level,I,3);
 }
 auto flags=at::empty({2*L-1},ints.dtype(at::kBool));
 roots_kernel<<<(2*L+254)/256,256,0,stream>>>(states.data_ptr<uint8_t>(),parent.data_ptr<int>(),2*L-1,cut,flags.data_ptr<bool>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {flags,states,active};
}
std::vector<at::Tensor> make_spans(at::Tensor parent){
 c10::cuda::CUDAGuard guard(parent.device());int L=parent.numel()/2+1;
 auto lo=at::full_like(parent,INT_MAX),hi=at::zeros_like(parent);
 spans_kernel<<<(L+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(parent.data_ptr<int>(),lo.data_ptr<int>(),hi.data_ptr<int>(),L);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {lo,hi};
}
std::vector<at::Tensor> run_dfs(at::Tensor ab,at::Tensor vs,at::Tensor fs,at::Tensor an,at::Tensor mn,
 at::Tensor left,at::Tensor right,at::Tensor parent,at::Tensor order,at::Tensor cp,at::Tensor planes,
 at::Tensor lo,at::Tensor hi,at::Tensor roots,at::Tensor prefix,at::Tensor hz,at::Tensor meta,at::Tensor am){
 c10::cuda::CUDAGuard guard(ab.device());auto mm=at::zeros({fs.size(0)},ab.options().dtype(at::kBool));auto counts=at::zeros({8},ab.options().dtype(at::kLong));
 auto lengths=at::zeros({parent.numel()},ab.options().dtype(at::kLong));
 auto range_states=at::empty({parent.numel(),2},ab.options().dtype(at::kByte));
 int nr=roots.numel();if(nr)dfs_kernel<<<nr,32,0,at::cuda::getCurrentCUDAStream()>>>(ab.data_ptr<double>(),vs.data_ptr<double>(),fs.data_ptr<int64_t>(),an.data_ptr<double>(),mn.data_ptr<double>(),cp.data_ptr<double>(),planes.data_ptr<double>(),planes.size(0),left.data_ptr<int>(),right.data_ptr<int>(),parent.data_ptr<int>(),order.data_ptr<int64_t>(),lo.data_ptr<int>(),hi.data_ptr<int>(),roots.data_ptr<int>(),nr,prefix.data_ptr<uint8_t>(),hz.numel()?hz.data_ptr<float>():nullptr,meta.data_ptr<int64_t>(),meta.size(0),parent.numel()/2,order.numel(),ab.size(0),am.data_ptr<bool>(),mm.data_ptr<bool>(),counts.data_ptr<int64_t>(),lengths.data_ptr<int64_t>(),range_states.data_ptr<uint8_t>());
 auto chunk_offsets=at::cumsum(lengths,0,at::kLong);
 // A few resident work blocks per SM. Each keeps taking grid-stride chunks;
 // scheduling depends on emitted output work, not on the number of DFS roots.
 int64_t max_chunks=(order.numel()+255)/256+parent.numel()/2+1;
 auto owner=at::empty({max_chunks},parent.options());
 map_chunk_owners<<<(parent.numel()+7)/8,256,0,at::cuda::getCurrentCUDAStream()>>>(chunk_offsets.data_ptr<int64_t>(),parent.numel(),owner.data_ptr<int>());
 int blocks=std::min<int64_t>(parent.numel()/2+1,4*at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
 parallel_output<<<blocks,256,0,at::cuda::getCurrentCUDAStream()>>>(ab.data_ptr<double>(),vs.data_ptr<double>(),fs.data_ptr<int64_t>(),cp.data_ptr<double>(),planes.data_ptr<double>(),planes.size(0),order.data_ptr<int64_t>(),lo.data_ptr<int>(),hi.data_ptr<int>(),chunk_offsets.data_ptr<int64_t>(),owner.data_ptr<int>(),range_states.data_ptr<uint8_t>(),parent.numel(),order.numel(),ab.size(0),hz.numel()?hz.data_ptr<float>():nullptr,meta.data_ptr<int64_t>(),meta.size(0),am.data_ptr<bool>(),mm.data_ptr<bool>(),counts.data_ptr<int64_t>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {mm,counts};
}
at::Tensor filter_ids(at::Tensor ab,at::Tensor cp,at::Tensor hz,at::Tensor meta,at::Tensor ids){
 c10::cuda::CUDAGuard guard(ab.device());auto keep=at::empty({ids.numel()},ab.options().dtype(at::kBool));
 if(ids.numel())filter_kernel<<<(ids.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(ab.data_ptr<double>(),cp.data_ptr<double>(),hz.data_ptr<float>(),meta.data_ptr<int64_t>(),meta.size(0),ids.data_ptr<int64_t>(),ids.numel(),keep.data_ptr<bool>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return keep;
}
