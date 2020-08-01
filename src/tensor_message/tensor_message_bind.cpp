#include <torch/extension.h>

void forward_cuda(
        torch::Tensor output,
        torch::Tensor W,
        torch::Tensor C,
        torch::Tensor F,
        torch::Tensor Y,
        torch::Tensor R,
        torch::Tensor L_out_list,
        torch::Tensor L_in_list,
        torch::Tensor u_sizes,
        torch::Tensor v_sizes,
        torch::Tensor output_base_offsets,
        torch::Tensor C_offsets,
        torch::Tensor F_base_offsets,
        torch::Tensor R_base_offsets);

void backward_F_cuda(
        torch::Tensor output,
        torch::Tensor W,
        torch::Tensor C,
        torch::Tensor G,
        torch::Tensor Y,
        torch::Tensor R,
        torch::Tensor L_out_list,
        torch::Tensor L_in_list,
        torch::Tensor u_sizes,
        torch::Tensor v_sizes,
        torch::Tensor output_base_offsets,
        torch::Tensor C_offsets,
        torch::Tensor G_base_offsets,
        torch::Tensor R_base_offsets);

void backward_R_cuda(
        torch::Tensor output,
        torch::Tensor W,
        torch::Tensor C,
        torch::Tensor G,
        torch::Tensor F,
        torch::Tensor Y,
        torch::Tensor L_out_list,
        torch::Tensor L_in_list,
        torch::Tensor u_sizes,
        torch::Tensor v_sizes,
        torch::Tensor output_base_offsets,
        torch::Tensor C_offsets,
        torch::Tensor G_base_offsets,
        torch::Tensor F_base_offsets);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT_DTYPE(x) TORCH_CHECK(x.dtype() == torch::kFloat64 || x.dtype() == torch::kFloat32, #x " must be either float32 or float64")
#define CHECK_INT_DTYPE(x) TORCH_CHECK(x.dtype() == torch::kInt32, #x " must be int32")

#define CHECK_FLOAT_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT_DTYPE(x);
#define CHECK_INT_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_INT_DTYPE(x);


torch::Tensor forward(
        torch::Tensor W,
        torch::Tensor C,
        torch::Tensor F,
        torch::Tensor Y,
        torch::Tensor R,
        torch::Tensor L_out_list,
        torch::Tensor L_in_list,
        torch::Tensor u_sizes,
        torch::Tensor v_sizes,
        torch::Tensor output_base_offsets,
        torch::Tensor C_offsets,
        torch::Tensor F_base_offsets,
        torch::Tensor R_base_offsets
){
    CHECK_FLOAT_INPUT(W);
    CHECK_FLOAT_INPUT(C);
    CHECK_FLOAT_INPUT(F);
    CHECK_FLOAT_INPUT(Y);
    CHECK_FLOAT_INPUT(R);
    CHECK_INT_INPUT(L_out_list);
    CHECK_INT_INPUT(L_in_list);
    CHECK_INT_INPUT(u_sizes);
    CHECK_INT_INPUT(v_sizes);
    CHECK_INT_INPUT(output_base_offsets);
    CHECK_INT_INPUT(C_offsets);
    CHECK_INT_INPUT(F_base_offsets);
    CHECK_INT_INPUT(R_base_offsets);

    const uint32_t l_out_ui_size = (uint32_t) output_base_offsets[output_base_offsets.size(0)-1].item<int32_t>();
    const uint32_t ab_size = (uint32_t) F.size(1);

    torch::Tensor output = torch::empty({l_out_ui_size, ab_size}, W.options()); // |(l_out, u, i), (a, b)|

    forward_cuda(output, W, C, F, Y, R,
                L_out_list, L_in_list, u_sizes, v_sizes,
                output_base_offsets, C_offsets, F_base_offsets, R_base_offsets);
    return output;
}


torch::Tensor backward_F(
        torch::Tensor W,
        torch::Tensor C,
        torch::Tensor G,
        torch::Tensor Y,
        torch::Tensor R,
        torch::Tensor L_out_list,
        torch::Tensor L_in_list,
        torch::Tensor u_sizes,
        torch::Tensor v_sizes,
        torch::Tensor output_base_offsets,
        torch::Tensor C_offsets,
        torch::Tensor G_base_offsets,
        torch::Tensor R_base_offsets
){
    CHECK_FLOAT_INPUT(W);
    CHECK_FLOAT_INPUT(C);
    CHECK_FLOAT_INPUT(G);
    CHECK_FLOAT_INPUT(Y);
    CHECK_FLOAT_INPUT(R);
    CHECK_INT_INPUT(L_out_list);
    CHECK_INT_INPUT(L_in_list);
    CHECK_INT_INPUT(u_sizes);
    CHECK_INT_INPUT(v_sizes);
    CHECK_INT_INPUT(output_base_offsets);
    CHECK_INT_INPUT(C_offsets);
    CHECK_INT_INPUT(G_base_offsets);
    CHECK_INT_INPUT(R_base_offsets);

    const uint32_t lin_vj_size = (uint32_t) output_base_offsets[output_base_offsets.size(0)-1].item<int32_t>();
    const uint32_t ab_size = (uint32_t) G.size(1);

    torch::Tensor output = torch::empty({lin_vj_size, ab_size}, W.options());   // |(l_in, v, j), (a, b)|

    backward_F_cuda(output, W, C, G, Y, R,
                    L_out_list, L_in_list, u_sizes, v_sizes,
                    output_base_offsets, C_offsets, G_base_offsets, R_base_offsets);
    return output;
}


torch::Tensor backward_R(
        torch::Tensor W,
        torch::Tensor C,
        torch::Tensor G,
        torch::Tensor F,
        torch::Tensor Y,
        torch::Tensor L_out_list,
        torch::Tensor L_in_list,
        torch::Tensor u_sizes,
        torch::Tensor v_sizes,
        torch::Tensor output_base_offsets,
        torch::Tensor C_offsets,
        torch::Tensor G_base_offsets,
        torch::Tensor F_base_offsets
){
    CHECK_FLOAT_INPUT(W);
    CHECK_FLOAT_INPUT(C);
    CHECK_FLOAT_INPUT(G);
    CHECK_FLOAT_INPUT(F);
    CHECK_FLOAT_INPUT(Y);
    CHECK_INT_INPUT(L_out_list);
    CHECK_INT_INPUT(L_in_list);
    CHECK_INT_INPUT(u_sizes);
    CHECK_INT_INPUT(v_sizes);
    CHECK_INT_INPUT(output_base_offsets);
    CHECK_INT_INPUT(C_offsets);
    CHECK_INT_INPUT(G_base_offsets);
    CHECK_INT_INPUT(F_base_offsets);

    const uint32_t lout_lin_luv = (uint32_t) output_base_offsets[output_base_offsets.size(0)-1].item<int32_t>();
    const uint32_t ab_size      = (uint32_t) F.size(1);

    torch::Tensor output = torch::empty({lout_lin_luv, ab_size}, W.options());   // |(l_out, l_in, l, u, v), (a, b)|

    backward_R_cuda(output, W, C, G, F, Y,
                    L_out_list, L_in_list, u_sizes, v_sizes,
                    output_base_offsets, C_offsets, G_base_offsets, F_base_offsets);
    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Tensor message function: forward pass (CUDA)");
  m.def("backward_F", &backward_F, "Tensor message function: backward pass for features (CUDA)");
  m.def("backward_R", &backward_R, "Tensor message function: backward pass for Radial Basis Function outputs (CUDA)");
}