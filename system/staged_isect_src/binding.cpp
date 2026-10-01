#include <torch/extension.h>
namespace gsplat {
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> staged_count(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,uint32_t,uint32_t,uint32_t);
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> staged_finish(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,int64_t,uint32_t,uint32_t,uint32_t,bool,bool);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
 m.def("count_async",&gsplat::staged_count);
 m.def("finish",&gsplat::staged_finish);
}
