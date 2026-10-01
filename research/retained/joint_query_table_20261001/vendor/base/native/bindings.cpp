#include <torch/extension.h>
#include <vector>
using T=at::Tensor;
T morton_codes(T,T,T);
std::vector<T> build_radix_bvh(T,T,T,T,T,int64_t);
std::vector<T> joint_query(T,T,T,T,T,T,T,T,T,T,T,T,T,T,int64_t,int64_t);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
 m.def("morton_codes",&morton_codes);
 m.def("build_radix_bvh",&build_radix_bvh);
 m.def("query",&joint_query);
}
