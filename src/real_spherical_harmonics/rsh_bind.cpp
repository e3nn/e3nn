#include <torch/extension.h>

void real_spherical_harmonics_cuda(torch::Tensor output, torch::Tensor xyz);
void e3nn_normalization_cuda(torch::Tensor rsh);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DTYPE(x) TORCH_CHECK(x.dtype() == torch::kFloat64 || x.dtype() == torch::kFloat32, #x " must be either float32 or float64")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_DTYPE(x);

torch::Tensor real_spherical_harmonics(torch::Tensor xyz, uint32_t lmax){
    CHECK_INPUT(xyz);

    const uint32_t lm_size = (lmax + 1) * (lmax + 1);
    const uint32_t ab_size = xyz.size(0);

    torch::Tensor output = torch::empty({lm_size, ab_size}, xyz.options());
    real_spherical_harmonics_cuda(output, xyz);
    return output;
}

void e3nn_normalization(torch::Tensor rsh){
    CHECK_INPUT(rsh);

    e3nn_normalization_cuda(rsh);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("real_spherical_harmonics", &real_spherical_harmonics, "Real Spherical Harmonics (CUDA)");
  m.def("e3nn_normalization", &e3nn_normalization, "e3nn normalization (-1)^L of real spherical harmonics (CUDA)");
}
