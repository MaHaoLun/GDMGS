#include <torch/extension.h>

at::Tensor pointwise_keep(
    at::Tensor positions,
    at::Tensor candidate_ids,
    at::Tensor depth,
    at::Tensor world_view,
    at::Tensor full_projection,
    double margin,
    double minimum_camera_z
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("pointwise_keep", &pointwise_keep);
}
