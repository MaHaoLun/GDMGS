#include <torch/extension.h>
#include "classify.h"
#include <vector>
std::vector<at::Tensor> frontier_query(at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t,int64_t);
std::vector<at::Tensor> cpu_classify(at::Tensor nodes,at::Tensor camera) {
    TORCH_CHECK(!nodes.is_cuda() && !camera.is_cuda() && nodes.is_contiguous() && camera.is_contiguous());
    TORCH_CHECK(nodes.scalar_type()==at::kDouble && camera.scalar_type()==at::kDouble && nodes.dim()==2 && nodes.size(1)==7 && camera.numel()==23);
    auto states=at::empty({nodes.size(0)},nodes.options().dtype(at::kByte));
    auto p=states.data_ptr<uint8_t>();auto n=nodes.data_ptr<double>();auto c=camera.data_ptr<double>();
    for(int64_t i=0;i<nodes.size(0);++i)p[i]=classify_six_planes(n+7*i,c);
    return {states};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("frontier_query",&frontier_query);m.def("cpu_classify",&cpu_classify);}
