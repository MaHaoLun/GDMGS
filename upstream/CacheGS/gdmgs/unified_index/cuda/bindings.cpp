#include <torch/extension.h>

at::Tensor anchor_sparse_max(at::Tensor depth);
at::Tensor anchor_gpu_certify(at::Tensor bounds, at::Tensor centers, at::Tensor radii,
    at::Tensor known, at::Tensor ids, at::Tensor removed_rank, at::Tensor rank,
    at::Tensor table, at::Tensor w2c, std::vector<double> domain,
    std::vector<int64_t> image_size, double pad, double near, double margin, at::Tensor counts);
at::Tensor node_gpu_certify(at::Tensor bounds, at::Tensor centers, at::Tensor radii,
    at::Tensor known, at::Tensor intervals, at::Tensor fov_prefix, at::Tensor table,
    at::Tensor w2c, std::vector<double> domain, std::vector<int64_t> image_size,
    double pad, double near, double margin, at::Tensor counts);
at::Tensor anchor_maximal_intervals(at::Tensor flags, at::Tensor parents,
    at::Tensor intervals, int64_t anchor_count, at::Tensor counts);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_max", &anchor_sparse_max);
    m.def("certify_anchors", &anchor_gpu_certify);
    m.def("certify_nodes", &node_gpu_certify);
    m.def("maximal_intervals", &anchor_maximal_intervals);
}
