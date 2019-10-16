#include <torch/extension.h>

void real_spherical_harmonics_cuda(
        torch::Tensor output_placeholder,
        torch::Tensor radii);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void real_spherical_harmonics(
        torch::Tensor output_placeholder,
        torch::Tensor radii){
    CHECK_INPUT(output_placeholder);
    CHECK_INPUT(radii);

    real_spherical_harmonics_cuda(output_placeholder, radii);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rsh", &real_spherical_harmonics, "Real Spherical Harmonics (CUDA)");
}