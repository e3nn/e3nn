#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>


__device__ constexpr uint32_t threads_repeat_m_per_batch_block()    { return 256; }
__device__ constexpr uint32_t threads_sum_m_per_batch_block()       { return 256; }


// Declarations of child kernels
template<typename T>
__global__ void repeat_m_child_cuda_kernel(
              T*        const __restrict__,
        const T*        const __restrict__,
		const uint32_t
);

template<typename T>
__global__ void sum_m_child_cuda_kernel(
              T*        const __restrict__,
        const T*        const __restrict__,
		const uint32_t,
		const uint32_t
);


//implementation
template<typename T>
__global__ void repeat_m_parent_cuda_kernel(
              T*        const __restrict__ output,
        const T*        const __restrict__ input,
        const uint32_t* const __restrict__ L_list,
        const uint32_t* const __restrict__ mul_sizes,
        const uint32_t* const __restrict__ output_base_offsets,
        const uint32_t* const __restrict__ input_base_offsets,
        const uint32_t  batch_size
){
    const uint32_t idx        = blockIdx.x;
    const uint32_t L          = L_list[idx];
    const uint32_t mul_size   = mul_sizes[idx];
    const uint32_t m_size     = 2 * L + 1;

    // add offsets
		  T* const __restrict__ output_l = output + output_base_offsets[idx] * batch_size;
	const T* const __restrict__ input_l  = input + input_base_offsets[idx] * batch_size;

    const uint32_t batch_num_blocks = (batch_size + threads_repeat_m_per_batch_block() - 1) / threads_repeat_m_per_batch_block();
    dim3 blocks(batch_num_blocks, mul_size, m_size);

    repeat_m_child_cuda_kernel<T><<<blocks, threads_repeat_m_per_batch_block()>>>(output_l, input_l, batch_size);
}


template<typename T>
__global__ void repeat_m_child_cuda_kernel(
              T*        const __restrict__ output_l,
        const T*        const __restrict__ input_l,
        const uint32_t  batch_size
){
    const uint32_t b      = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t mul    = blockIdx.y;
    const uint32_t m      = blockIdx.z;
    const uint32_t m_size = gridDim.z;
    
    // last warp may be not completely filled, in that case return early
    if (b >= batch_size) return;

    // T' = T_l
    // T'_vm,b = T'_v,b
    output_l[mul * m_size * batch_size + m * batch_size + b] = input_l[mul * batch_size + b];
}


template<typename T>
__global__ void sum_m_parent_cuda_kernel(
              T*        const __restrict__ output,
        const T*        const __restrict__ input,
        const uint32_t* const __restrict__ L_list,
        const uint32_t* const __restrict__ mul_sizes,
        const uint32_t* const __restrict__ output_base_offsets,
        const uint32_t* const __restrict__ input_base_offsets,
        const uint32_t  batch_size
){
    const uint32_t idx        = blockIdx.x;
    const uint32_t L          = L_list[idx];
    const uint32_t mul_size   = mul_sizes[idx];
    const uint32_t m_size     = 2 * L + 1;

    // add offsets
		  T* const __restrict__ output_l = output + output_base_offsets[idx] * batch_size;
	const T* const __restrict__ input_l  = input + input_base_offsets[idx] * batch_size;

    const uint32_t batch_num_blocks = (batch_size + threads_sum_m_per_batch_block() - 1) / threads_sum_m_per_batch_block();
    dim3 blocks(batch_num_blocks, mul_size);

    sum_m_child_cuda_kernel<T><<<blocks, threads_sum_m_per_batch_block()>>>(output_l, input_l, batch_size, m_size);
}


template<typename T>
__global__ void sum_m_child_cuda_kernel(
              T*        const __restrict__ output_l,
        const T*        const __restrict__ input_l,
        const uint32_t  batch_size,
        const uint32_t  m_size
){
    const uint32_t b      = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t mul    = blockIdx.y;

    // last warp may be not completely filled, in that case return early
    if (b >= batch_size) return;

    // T' = T_l
    // T'_v,ab = sum_m(T'_vm,ab)
    T entry = 0;
    for(uint32_t m = 0; m < m_size; ++m){
        entry += input_l[mul * m_size * batch_size + m * batch_size + b];
    }

    output_l[mul * batch_size + b] = entry;
}


void repeat_m_cuda(
        torch::Tensor output,
        torch::Tensor input,
        torch::Tensor L_list,
        torch::Tensor mul_sizes,
        torch::Tensor output_base_offsets,
        torch::Tensor input_base_offsets
){
    const uint32_t batch_size = input.size(1);

    const uint32_t* const __restrict__ L_list_ptr               = (uint32_t*) L_list.data_ptr();
    const uint32_t* const __restrict__ mul_sizes_ptr            = (uint32_t*) mul_sizes.data_ptr();
    const uint32_t* const __restrict__ output_base_offsets_ptr  = (uint32_t*) output_base_offsets.data_ptr();
    const uint32_t* const __restrict__ input_base_offsets_ptr   = (uint32_t*) input_base_offsets.data_ptr();

    dim3 blocks(L_list.size(0));

    if (output.dtype() == torch::kFloat64){
              double* const __restrict__ output_ptr = (double*) output.data_ptr();
        const double* const __restrict__ input_ptr  = (double*) input.data_ptr();
        repeat_m_parent_cuda_kernel<double><<<blocks, 1>>>(output_ptr, input_ptr, L_list_ptr, mul_sizes_ptr, output_base_offsets_ptr, input_base_offsets_ptr, batch_size);
    }
    else if (output.dtype() == torch::kFloat32){
              float* const __restrict__ output_ptr = (float*) output.data_ptr();
        const float* const __restrict__ input_ptr  = (float*) input.data_ptr();
        repeat_m_parent_cuda_kernel<float><<<blocks, 1>>>(output_ptr, input_ptr, L_list_ptr, mul_sizes_ptr, output_base_offsets_ptr, input_base_offsets_ptr, batch_size);
    }
}


void sum_m_cuda(
        torch::Tensor output,
        torch::Tensor input,
        torch::Tensor L_list,
        torch::Tensor mul_sizes,
        torch::Tensor output_base_offsets,
        torch::Tensor input_base_offsets
){
    const uint32_t batch_size = (uint32_t) input.size(1);

    const uint32_t* const __restrict__ L_list_ptr               = (uint32_t*) L_list.data_ptr();
    const uint32_t* const __restrict__ mul_sizes_ptr            = (uint32_t*) mul_sizes.data_ptr();
    const uint32_t* const __restrict__ output_base_offsets_ptr  = (uint32_t*) output_base_offsets.data_ptr();
    const uint32_t* const __restrict__ input_base_offsets_ptr   = (uint32_t*) input_base_offsets.data_ptr();

    dim3 blocks(L_list.size(0));

    if (output.dtype() == torch::kFloat64){
              double* const __restrict__ output_ptr = (double*) output.data_ptr();
        const double* const __restrict__ input_ptr  = (double*) input.data_ptr();
        sum_m_parent_cuda_kernel<double><<<blocks, 1>>>(output_ptr, input_ptr, L_list_ptr, mul_sizes_ptr, output_base_offsets_ptr, input_base_offsets_ptr, batch_size);
    }
    else if (output.dtype() == torch::kFloat32){
              float* const __restrict__ output_ptr = (float*) output.data_ptr();
        const float* const __restrict__ input_ptr  = (float*) input.data_ptr();
        sum_m_parent_cuda_kernel<float><<<blocks, 1>>>(output_ptr, input_ptr, L_list_ptr, mul_sizes_ptr, output_base_offsets_ptr, input_base_offsets_ptr, batch_size);
    }
}
