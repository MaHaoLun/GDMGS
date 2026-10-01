#include <torch/extension.h>
#include "bounds.h"
#include <vector>
#include <thread>
std::vector<at::Tensor> gpu_query(at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t,int64_t,bool,bool);
at::Tensor gpu_blocks(at::Tensor,at::Tensor,int64_t);
at::Tensor gpu_refine(at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t);
std::vector<at::Tensor> cpu_query(at::Tensor bounds,at::Tensor nodes,at::Tensor order,at::Tensor camera,int64_t base,int64_t leaf,bool dense) {
    TORCH_CHECK(!bounds.is_cuda() && !nodes.is_cuda() && !order.is_cuda() && !camera.is_cuda());
    TORCH_CHECK(bounds.scalar_type()==at::kDouble && nodes.scalar_type()==at::kDouble && camera.scalar_type()==at::kDouble && order.scalar_type()==at::kLong);
    TORCH_CHECK(bounds.is_contiguous() && nodes.is_contiguous() && order.is_contiguous() && camera.is_contiguous());
    TORCH_CHECK(bounds.dim()==2 && bounds.size(1)==7 && nodes.dim()==2 && nodes.size(1)==7 && camera.numel()==23 && order.numel()==bounds.size(0));
    TORCH_CHECK(base>0 && leaf>0 && nodes.size(0)==2*base);
    auto mask=at::zeros({bounds.size(0)},bounds.options().dtype(at::kBool));
    auto counts=at::zeros({3},bounds.options().dtype(at::kLong));
    auto out=mask.data_ptr<bool>();auto ct=counts.data_ptr<int64_t>();
    auto b=bounds.data_ptr<double>();auto n=nodes.data_ptr<double>();auto c=camera.data_ptr<double>();auto ids=order.data_ptr<int64_t>();
    if(dense) {for(int64_t i=0;i<bounds.size(0);++i) {out[i]=!outside(b+7*i,c);++ct[1];}}
    else {
        std::vector<int64_t> stack{1};
        while(!stack.empty()) {
            auto k=stack.back();stack.pop_back();++ct[0];
            if(outside(n+7*k,c)) {++ct[2];continue;}
            if(k<base) {stack.push_back(2*k+1);stack.push_back(2*k);}
            else for(int64_t j=(k-base)*leaf;j<std::min((k-base+1)*leaf,bounds.size(0));++j) {
                auto id=ids[j];out[id]=!outside(b+7*id,c);++ct[1];
            }
        }
    }
    return {mask,counts};
}
// Transfer only one byte per spatial leaf. GPU retains the original-row
// -> leaf mapping and constructs ordered IDs without a CPU ID roundtrip.
at::Tensor cpu_blocks(at::Tensor nodes,at::Tensor camera,int64_t base,int64_t threads) {
    TORCH_CHECK(!nodes.is_cuda() && !camera.is_cuda() && nodes.scalar_type()==at::kDouble && camera.scalar_type()==at::kDouble);
    TORCH_CHECK(nodes.is_contiguous() && camera.is_contiguous() && nodes.dim()==2 && nodes.size(1)==7 && nodes.size(0)==2*base && camera.numel()==23);
    TORCH_CHECK(base>0 && threads>=1 && threads<=32);
    auto out=at::empty({base},nodes.options().dtype(at::kBool).pinned_memory(true));
    auto mask=out.data_ptr<bool>();auto n=nodes.data_ptr<double>();auto c=camera.data_ptr<double>();
    auto worker=[&](int64_t t){for(int64_t i=t*base/threads;i<(t+1)*base/threads;++i) mask[i]=!outside(n+7*(base+i),c);};
    if(threads==1) worker(0);
    else {
        std::vector<std::thread> workers;
        for(int64_t t=0;t<threads;++t)workers.emplace_back(worker,t);
        for(auto& w:workers)w.join();
    }
    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("cpu_query",&cpu_query);m.def("gpu_query",&gpu_query);
    m.def("gpu_refine",&gpu_refine);m.def("cpu_blocks",&cpu_blocks);m.def("gpu_blocks",&gpu_blocks);
}
