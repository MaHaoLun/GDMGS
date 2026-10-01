#include <torch/extension.h>
#include <vector>
using T=at::Tensor;
std::vector<T> prefix_query(T,T,T,T,T,T,T,T,T,T,int64_t);
std::vector<T> make_spans(T);
std::vector<T> run_dfs(T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T);
T filter_ids(T,T,T,T,T);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
 m.def("prefix",&prefix_query);m.def("spans",&make_spans);m.def("dfs",&run_dfs);m.def("filter",&filter_ids);
}
