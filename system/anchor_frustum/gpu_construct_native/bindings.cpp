#include <torch/extension.h>
#include <vector>
at::Tensor morton_codes(at::Tensor,at::Tensor,at::Tensor);
std::vector<at::Tensor> build_radix_bvh(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t);
std::vector<at::Tensor> build_octree(at::Tensor,at::Tensor,at::Tensor,at::Tensor);
std::vector<at::Tensor> query_radix(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t,bool,at::Tensor);
at::Tensor gpu_dense_mask(at::Tensor,at::Tensor);
std::vector<at::Tensor> query_octree(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
 m.def("morton_codes",&morton_codes);
 m.def("gpu_dense_mask",&gpu_dense_mask);
 m.def("build_radix_bvh",&build_radix_bvh);
 m.def("build_octree",&build_octree);
 m.def("query_radix",&query_radix);
 m.def("query_octree",&query_octree);
}
