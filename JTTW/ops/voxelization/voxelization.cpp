#include <torch/extension.h>
#include "voxelization.hpp"

namespace voxelization
{

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hard_voxelize", &hard_voxelize, "hard voxelize");
    m.def("dynamic_voxelize", &dynamic_voxelize, "dynamic voxelize");
}

} // namespace voxelization
