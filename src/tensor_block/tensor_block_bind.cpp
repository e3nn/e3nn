#include <torch/extension.h>

void repeat_m_cuda(
        torch::Tensor output,
        torch::Tensor input,
        torch::Tensor L_list,
        torch::Tensor mul_sizes,
        torch::Tensor output_base_offsets,
        torch::Tensor input_base_offsets);

void sum_m_cuda(
        torch::Tensor output,
        torch::Tensor input,
        torch::Tensor L_list,
        torch::Tensor mul_sizes,
        torch::Tensor output_base_offsets,
        torch::Tensor input_base_offsets);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT_DTYPE(x) TORCH_CHECK(x.dtype() == torch::kFloat64 || x.dtype() == torch::kFloat32, #x " must be either float32 or float64")
#define CHECK_INT_DTYPE(x) TORCH_CHECK(x.dtype() == torch::kInt32, #x " must be int32")

#define CHECK_FLOAT_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT_DTYPE(x);
#define CHECK_INT_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_INT_DTYPE(x);


torch::Tensor repeat_m(
        torch::Tensor input,
        torch::Tensor L_list,
        torch::Tensor mul_sizes,
        torch::Tensor output_base_offsets,
        torch::Tensor input_base_offsets
){
    CHECK_FLOAT_INPUT(input);
    CHECK_INT_INPUT(L_list);
    CHECK_INT_INPUT(mul_sizes);
    CHECK_INT_INPUT(output_base_offsets);
    CHECK_INT_INPUT(input_base_offsets);

    const uint32_t ab_size = (uint32_t) input.size(1);
    const uint32_t lvm_size = (uint32_t) output_base_offsets[output_base_offsets.size(0)-1].item<int32_t>();

    torch::Tensor output = torch::empty({lvm_size, ab_size}, input.options());
    repeat_m_cuda(output, input, L_list, mul_sizes, output_base_offsets, input_base_offsets);

    return output;
}


torch::Tensor sum_m(
        torch::Tensor input,
        torch::Tensor L_list,
        torch::Tensor mul_sizes,
        torch::Tensor output_base_offsets,
        torch::Tensor input_base_offsets
){
    CHECK_FLOAT_INPUT(input);
    CHECK_INT_INPUT(L_list);
    CHECK_INT_INPUT(mul_sizes);
    CHECK_INT_INPUT(output_base_offsets);
    CHECK_INT_INPUT(input_base_offsets);

    const uint32_t ab_size = (uint32_t) input.size(1);
    const uint32_t lv_size = (uint32_t) output_base_offsets[output_base_offsets.size(0)-1].item<int32_t>();

    torch::Tensor output = torch::empty({lv_size, ab_size}, input.options());
    sum_m_cuda(output, input, L_list, mul_sizes, output_base_offsets, input_base_offsets);

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("repeat_m", &repeat_m, "Replicate T_lv over m into T_lvm (CUDA)");
  m.def("sum_m", &sum_m, "Sum T_lvm over m -> T'_lv (CUDA)");
}
