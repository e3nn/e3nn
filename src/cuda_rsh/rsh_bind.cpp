#include <torch/extension.h>

void real_spherical_harmonics_cuda(
        torch::Tensor Ys,
        torch::Tensor radii);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DTYPE(x) AT_ASSERTM(x.dtype() == torch::kFloat64 || x.dtype() == torch::kFloat32, #x " must be either float32 or float64")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_DTYPE(x);

void real_spherical_harmonics(
        torch::Tensor Ys,
        torch::Tensor radii){
    CHECK_INPUT(Ys);
    CHECK_INPUT(radii);
    AT_ASSERTM(Ys.dtype() == radii.dtype(), "output and input in rsh should have the same data type");

    real_spherical_harmonics_cuda(Ys, radii);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("real_spherical_harmonics", &real_spherical_harmonics, "Real Spherical Harmonics (CUDA)");
}
