#include <torch/extension.h>

void rsh_cuda(torch::Tensor output, torch::Tensor xyz);
void rsh_cpp(torch::Tensor output, torch::Tensor xyz);
void drsh_cuda(torch::Tensor output, torch::Tensor xyz);
void drsh_cpp(torch::Tensor output, torch::Tensor xyz);

void e3nn_normalization_cuda(torch::Tensor rsh);

// C++ interface

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DTYPE(x) TORCH_CHECK(x.dtype() == torch::kFloat64 || x.dtype() == torch::kFloat32, #x " must be either float32 or float64")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x); CHECK_DTYPE(x);

torch::Tensor rsh(torch::Tensor xyz, uint32_t lmax) {
    CHECK_INPUT(xyz);

    const uint32_t lm_size = (lmax + 1) * (lmax + 1);
    const uint32_t ab_size = xyz.size(0);

    torch::Tensor output = torch::empty({lm_size, ab_size}, xyz.options());

    if (xyz.device().is_cuda()) rsh_cuda(output, xyz);
    else  /*  device is cpu  */ rsh_cpp(output, xyz);

    return output;
}

torch::Tensor drsh(torch::Tensor xyz, uint32_t lmax) {
    CHECK_INPUT(xyz);

    const uint32_t lm_size = (lmax + 1) * (lmax + 1);
    const uint32_t ab_size = xyz.size(0);

    torch::Tensor output = torch::empty({lm_size, ab_size, 3}, xyz.options());

    if (xyz.device().is_cuda()) drsh_cuda(output, xyz);
    else  /*  device is cpu  */ drsh_cpp(output, xyz);

    return output;
}


void e3nn_normalization(torch::Tensor rsh) {
    CHECK_INPUT(rsh);

    e3nn_normalization_cuda(rsh);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rsh", &rsh, "Real Spherical Harmonics");
  m.def("drsh", &drsh, "Partial derivatives of Real Spherical Harmonics with respect to the Cartesian coordinates");
  m.def("e3nn_normalization", &e3nn_normalization, "e3nn normalization (-1)^L of real spherical harmonics (CUDA)");
}
