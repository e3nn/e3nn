#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

__device__ constexpr uint32_t threads_per_block_forward_parent_cuda_kernel()    { return 256; }
__device__ constexpr uint32_t threads_per_block_backward_R_parent_cuda_kernel() { return 256; }
__device__ constexpr uint32_t threads_per_block_backward_F_parent_cuda_kernel() { return 256; }


// Declarations of child kernels
template<typename T>
__global__ void forward_child_cuda_kernel(
              T*        const __restrict__,
        const T*        const __restrict__,
        const T*        const __restrict__,
        const T*        const __restrict__,
        const T*        const __restrict__,
        const T,
		const uint32_t,
		const uint32_t,
		const uint32_t,
		const uint32_t,
		const uint32_t,
		const uint32_t,
		const uint32_t
);

template<typename T>
__global__ void backward_F_child_cuda_kernel(
              T*        const __restrict__,
        const T*        const __restrict__,
        const T*        const __restrict__,
        const T*        const __restrict__,
        const T*        const __restrict__,
        const T,
		const uint32_t,
		const uint32_t,
		const uint32_t,
		const uint32_t,
		const uint32_t,
		const uint32_t,
		const uint32_t
);

template<typename T>
__global__ void backward_R_child_cuda_kernel(
			  T* 	    const __restrict__,
		const T* 	    const __restrict__,
		const T* 	    const __restrict__,
		const T* 	    const __restrict__,
		const T* 	    const __restrict__,
		const T,
		const uint32_t,
		const uint32_t,
		const uint32_t,
		const uint32_t,
		const uint32_t,
		const uint32_t
);


// Implementations
template<typename T>
__global__ void forward_parent_cuda_kernel(
              T*        const __restrict__ output,
        const T*        const __restrict__ W,
        const T*        const __restrict__ C,
        const T*        const __restrict__ F,
        const T*        const __restrict__ Y,
        const T*        const __restrict__ R,
        const uint32_t* const __restrict__ L_out_list,
        const uint32_t* const __restrict__ L_in_list,
        const uint32_t* const __restrict__ u_sizes,
        const uint32_t* const __restrict__ v_sizes,
        const uint32_t* const __restrict__ output_base_offsets,
        const uint32_t* const __restrict__ C_offsets,
        const uint32_t* const __restrict__ F_base_offsets,
        const uint32_t* const __restrict__ R_base_offsets,
        const uint32_t					   ab_size,
        const uint32_t 					   l_in_max_net_bound
){
    const uint32_t l_out_id  = blockIdx.x;
	const uint32_t l_in_id 	 = blockIdx.y;
	const uint32_t l_in_size = gridDim.y;

	const uint32_t l_out = L_out_list[l_out_id];
	const uint32_t l_in  = L_in_list[l_in_id];
	const uint32_t i_size = 2*l_out + 1;
	const uint32_t j_size = 2*l_in + 1;

	const uint32_t u_size = u_sizes[l_out_id]; 	// output multiplicity (for particular l_out)
	const uint32_t v_size = v_sizes[l_in_id]; 	// input multiplicity  (for particular l_in)

    /*
	  Expected order of indices:
	 	 output -> [l_out, u, i, a, b]
	 	 W 		-> [l_out, l_in]
	 	 C		-> [l_out, l_in, l, i, j, m]
	 	 F 		-> [l_in, v, j, a, b]
	 	 Y 		-> [l, m, a, b]
	 	 R      -> [l_out, l_in, l, u, v, a, b]
	 */
	// add offsets
		  T* const __restrict__ output_lout	    = output + (output_base_offsets[l_out_id] * ab_size);             // base offsets are the same as for gradients
	const T* const __restrict__ C_lout_lin		= C + C_offsets[l_out*l_in_max_net_bound + l_in];
	const T* const __restrict__ F_lin			= F + (F_base_offsets[l_in_id] * ab_size);
	const T* const __restrict__ R_lout_lin      = R + (R_base_offsets[l_out_id*l_in_size + l_in_id] * ab_size);

	const T W_lout_lin = W[l_out_id * l_in_size + l_in_id];

	const uint32_t l_min = abs((int32_t)l_out - (int32_t)l_in);
	const uint32_t l_max = l_out + l_in;

	const uint32_t threads_per_block = threads_per_block_forward_parent_cuda_kernel();
	const uint32_t uiab_size = u_size * i_size * ab_size;

	const uint32_t blocks = (uiab_size + threads_per_block - 1)/threads_per_block;

    forward_child_cuda_kernel<<<blocks, threads_per_block>>>(output_lout, C_lout_lin, F_lin, Y, R_lout_lin, W_lout_lin,
                                                             l_min, l_max, u_size, v_size, ab_size, i_size, j_size);
}


template<typename T>
__global__ void forward_child_cuda_kernel(
              T*        const __restrict__ output_lout,
        const T*        const __restrict__ C_lout_lin,
        const T*        const __restrict__ F_lin,
        const T*        const __restrict__ Y,
        const T*        const __restrict__ R_lout_lin,
        const T 					       W_lout_lin,
		const uint32_t					   l_min,
		const uint32_t					   l_max,
		const uint32_t					   u_size,
		const uint32_t					   v_size,
		const uint32_t					   ab_size,
		const uint32_t 					   i_size,
		const uint32_t 					   j_size
){
    const uint32_t uiab = threadIdx.x + blockIdx.x * blockDim.x;

    // last block can be incompletely filled, because uiab_size is not necessary divisible by set number of threads
	if (blockIdx.x == gridDim.x - 1 && uiab >= u_size * i_size * ab_size) return;

	// deduce individual indices
	const uint32_t u    = uiab / (i_size * ab_size);
	const uint32_t i    = (uiab - u * i_size * ab_size) / ab_size;
	const uint32_t ab   = uiab - u * i_size * ab_size - i * ab_size;

	T output_lout_uiab_addendum = 0;

    // TODO: cache multipliers to registers
    for(uint32_t l_f = l_min, l_id = 0; l_f <= l_max; ++l_f, ++l_id){
        for(uint32_t m = 0, m_size = 2*l_f + 1; m < m_size; ++m){
	        for(uint32_t j = 0; j < j_size; ++j){
	            for(uint32_t v = 0; v < v_size; ++v){
	                output_lout_uiab_addendum +=
	                    C_lout_lin[(l_f*l_f - l_min*l_min)*i_size*j_size + i*j_size*m_size + j*m_size + m] *
	                    F_lin[v*j_size*ab_size + j*ab_size + ab] *
	                    Y[(l_f*l_f + m)*ab_size + ab] *
	                    R_lout_lin[l_id*u_size*v_size*ab_size + u*v_size*ab_size + v*ab_size + ab];
	            }
	        }
	    }
	}

	atomicAdd(&output_lout[uiab], W_lout_lin * output_lout_uiab_addendum);
}


template<typename T>
__global__ void backward_F_parent_cuda_kernel(
              T*        const __restrict__ output,
        const T*        const __restrict__ W,
        const T*        const __restrict__ C,
        const T*        const __restrict__ G,
        const T*        const __restrict__ Y,
        const T*        const __restrict__ R,
        const uint32_t* const __restrict__ L_out_list,
        const uint32_t* const __restrict__ L_in_list,
        const uint32_t* const __restrict__ u_sizes,
        const uint32_t* const __restrict__ v_sizes,
        const uint32_t* const __restrict__ output_base_offsets,
        const uint32_t* const __restrict__ C_offsets,
        const uint32_t* const __restrict__ G_base_offsets,
        const uint32_t* const __restrict__ R_base_offsets,
        const uint32_t					   ab_size,
        const uint32_t 					   l_in_max_net_bound
){
    const uint32_t l_out_id  = blockIdx.x;
	const uint32_t l_in_id 	 = blockIdx.y;
	const uint32_t l_in_size = gridDim.y;

	const uint32_t l_out = L_out_list[l_out_id];
	const uint32_t l_in  = L_in_list[l_in_id];
	const uint32_t i_size = 2*l_out + 1;
	const uint32_t j_size = 2*l_in + 1;

	const uint32_t u_size = u_sizes[l_out_id]; 	// output multiplicity (for particular l_out)
	const uint32_t v_size = v_sizes[l_in_id]; 	// input multiplicity  (for particular l_in)

    /*
	  Expected order of indices:
	 	 output -> [l_in, v, j, a, b]
	 	 W 		-> [l_out, l_in]
	 	 C		-> [l_out, l_in, l, i, j, m]
	 	 G 		-> [l_out, u, i, a, b]
	 	 Y 		-> [l, m, a, b]
	 	 R      -> [l_out, l_in, l, u, v, a, b]
	 */
	// add offsets
		  T* const __restrict__ output_lin	    = output + (output_base_offsets[l_in_id] * ab_size);             // base offsets are the same as for features
	const T* const __restrict__ C_lout_lin		= C + C_offsets[l_out*l_in_max_net_bound + l_in];
	const T* const __restrict__ G_lout			= G + (G_base_offsets[l_out_id] * ab_size);
	const T* const __restrict__ R_lout_lin      = R + (R_base_offsets[l_out_id*l_in_size + l_in_id] * ab_size);

	const T W_lout_lin = W[l_out_id * l_in_size + l_in_id];

	const uint32_t l_min = abs((int32_t)l_out - (int32_t)l_in);
	const uint32_t l_max = l_out + l_in;

	const uint32_t threads_per_block = threads_per_block_backward_F_parent_cuda_kernel();
	const uint32_t vjab_size = v_size * j_size * ab_size;

	const uint32_t blocks = (vjab_size + threads_per_block - 1)/threads_per_block;

    backward_F_child_cuda_kernel<<<blocks, threads_per_block>>>(output_lin, C_lout_lin, G_lout, Y, R_lout_lin, W_lout_lin,
                                                                          l_min, l_max, u_size, v_size, ab_size, i_size, j_size);
}


template<typename T>
__global__ void backward_F_child_cuda_kernel(
              T*        const __restrict__ output_lin,
        const T*        const __restrict__ C_lout_lin,
        const T*        const __restrict__ G_lout,
        const T*        const __restrict__ Y,
        const T*        const __restrict__ R_lout_lin,
        const T 					       W_lout_lin,
		const uint32_t					   l_min,
		const uint32_t					   l_max,
		const uint32_t					   u_size,
		const uint32_t					   v_size,
		const uint32_t					   ab_size,
		const uint32_t 					   i_size,
		const uint32_t 					   j_size
){
    const uint32_t vjab = threadIdx.x + blockIdx.x * blockDim.x;

    // last block can be incompletely filled, because vjab_size is not necessary divisible by set number of threads
	if (blockIdx.x == gridDim.x - 1 && vjab >= v_size * j_size * ab_size) return;

	// deduce individual indices
	const uint32_t v	= vjab / (j_size * ab_size);
	const uint32_t j 	= (vjab - v * j_size * ab_size) / ab_size;
	const uint32_t ab   = vjab - v * j_size * ab_size - j * ab_size;

	T output_lin_v_j_ab_addendum = 0;

    // TODO: cache multipliers to registers
    for(uint32_t l_f = l_min, l_id = 0; l_f <= l_max; ++l_f, ++l_id){
        for(uint32_t m = 0, m_size = 2*l_f + 1; m < m_size; ++m){
    	    for(uint32_t i = 0; i < i_size; ++i){
	            for(uint32_t u = 0; u < u_size; ++u){
	                output_lin_v_j_ab_addendum +=
	                    C_lout_lin[(l_f*l_f - l_min*l_min)*i_size*j_size + i*j_size*m_size + j*m_size + m] *
	                    G_lout[u*i_size*ab_size + i*ab_size + ab] *
	                    Y[(l_f*l_f + m)*ab_size + ab] *
	                    R_lout_lin[l_id*u_size*v_size*ab_size + u*v_size*ab_size + v*ab_size + ab];
	            }
	        }
	    }
	}

	atomicAdd(&output_lin[vjab], W_lout_lin * output_lin_v_j_ab_addendum);
}


template<typename T>
__global__ void backward_R_parent_cuda_kernel(
			  T* 	    const __restrict__ output,		        // placeholder to store gradients
		const T* 	    const __restrict__ W,			        // normalization coefficients
		const T* 	    const __restrict__ C,			        // coupling coefficients
		const T* 	    const __restrict__ G,			        // gradients coming from next layer
		const T* 	    const __restrict__ F,			        // input features
		const T* 	    const __restrict__ Y,			        // spherical harmonics
		const uint32_t* const __restrict__ L_out_list,			// output rotational orders
		const uint32_t* const __restrict__ L_in_list,			// input rotational orders
		const uint32_t* const __restrict__ u_sizes,				// output multiplicities
		const uint32_t* const __restrict__ v_sizes,				// input multiplicities
		const uint32_t* const __restrict__ output_base_offsets,	// jump points for indexing output over l_out, l_in
		const uint32_t* const __restrict__ C_offsets,			// jump points for indexing C over l_out, l_in
		const uint32_t* const __restrict__ G_base_offsets,		// jump points for indexing G over l_out
		const uint32_t* const __restrict__ F_base_offsets, 		// jump points for indexing F over l_in
		const uint32_t					   ab_size,			    // total number of pairs point-neighbor
		const uint32_t 					   l_in_max_net_bound	// maximal value of l_in that is present in C (for selecting offset)
) {
	const uint32_t l_out_id  = blockIdx.x;
	const uint32_t l_in_id 	 = blockIdx.y;
	const uint32_t l_in_size = gridDim.y;

	const uint32_t l_out = L_out_list[l_out_id];
	const uint32_t l_in  = L_in_list[l_in_id];
	const uint32_t i_size = 2*l_out + 1;
	const uint32_t j_size = 2*l_in + 1;

	const uint32_t u_size = u_sizes[l_out_id]; 	// output multiplicity (for particular l_out)
	const uint32_t v_size = v_sizes[l_in_id]; 	// input multiplicity  (for particular l_in)

	/*
	  Expected order of indices:
	 	 output -> [l_out, l_in, l, u, v, a, b]
	 	 C		-> [l_out, l_in, l, i, j, m]
	 	 G 		-> [l_out, u, i, a, b]
	 	 F 		-> [l_in, v, j, a, b]
	 	 Y 		-> [l, m, a, b]
	 	 W 		-> [l_out, l_in]
	 */

	// add offsets
		  T* const __restrict__ output_lout_lin	= output + (output_base_offsets[l_out_id*l_in_size + l_in_id] * ab_size); // base offsets are the same as for R
	const T* const __restrict__ C_lout_lin		= C + C_offsets[l_out*l_in_max_net_bound + l_in];
	const T* const __restrict__ G_lout			= G + (G_base_offsets[l_out_id] * ab_size);
	const T* const __restrict__ F_lin			= F + (F_base_offsets[l_in_id] * ab_size);
	// no offsets for Y
	const T  W_lout_lin	= W[l_out_id * l_in_size + l_in_id];

	const uint32_t l_offset = abs((int32_t)l_out - (int32_t)l_in);

	const uint32_t threads_per_block = threads_per_block_backward_R_parent_cuda_kernel();
	const uint32_t uvab_size = u_size * v_size * ab_size;

	dim3 blocks((uvab_size + threads_per_block - 1)/threads_per_block, 2*min(l_out, l_in)+1);

	// TODO: for parity we will need to pass additional list with l filters, or maybe recreate get_l_filters_with_parity here
	backward_R_child_cuda_kernel<<<blocks, threads_per_block>>>(output_lout_lin, C_lout_lin, G_lout, F_lin, Y, W_lout_lin,
			                                                    l_offset, u_size, v_size, ab_size, i_size, j_size);
}


template<typename T>
__global__ void backward_R_child_cuda_kernel(
			  T* 	    const __restrict__ output_lout_lin,
		const T* 	    const __restrict__ C_lout_lin,
		const T* 	    const __restrict__ G_lout,
		const T* 	    const __restrict__ F_lin,
		const T* 	    const __restrict__ Y,
		const T 					       W_lout_lin,
		const uint32_t					   l_offset,
		const uint32_t					   u_size,
		const uint32_t					   v_size,
		const uint32_t					   ab_size,
		const uint32_t 					   i_size,
		const uint32_t 					   j_size
){
	const uint32_t uvab = threadIdx.x + blockIdx.x * blockDim.x;

	// last block can be incompletely filled, because uvab_size is not necessary divisible by set number of threads
	if (blockIdx.x == gridDim.x - 1 && uvab >= u_size * v_size * ab_size) return;

    const uint32_t l_id   = blockIdx.y;
	const uint32_t l_f    = l_id + l_offset;
	const uint32_t m_size = 2*l_f + 1;

	// deduce individual indices
	const uint32_t u	= uvab / (v_size * ab_size);
	const uint32_t v 	= (uvab - u * v_size * ab_size) / ab_size;
	const uint32_t ab   = uvab - u * v_size * ab_size - v * ab_size ;

	// add offsets
	const T* const __restrict__ C_lout_lin_l 	= C_lout_lin 	+ (i_size * j_size * (l_f*l_f - l_offset*l_offset)); 	// only valid L's, thus index is shifted
	const T* const __restrict__ G_lout_u		= G_lout 		+ (u * i_size * ab_size);
	const T* const __restrict__ F_lin_v		    = F_lin 		+ (v * j_size * ab_size);
	const T* const __restrict__ Y_l 			= Y 			+ (l_f * l_f * ab_size);							    // contains values without gaps along L

	// make additions (writes) to register
	T output_lout_lin_l_uvab = 0;

    uint32_t ijm = 0;
    T G_tmp, F_tmp;
	for (uint32_t i = 0; i < i_size; ++i){
	    G_tmp = G_lout_u[i*ab_size + ab];
		for (uint32_t j = 0; j < j_size; ++j){
		    F_tmp = F_lin_v[j*ab_size + ab];
			for (uint32_t m = 0; m < m_size; ++m, ++ijm){
				output_lout_lin_l_uvab += C_lout_lin_l[ijm] * G_tmp * F_tmp * Y_l[m*ab_size + ab];
			}
		}
	}

	// write normalized result to global memory
	output_lout_lin[l_id * u_size * v_size * ab_size + uvab] = W_lout_lin * output_lout_lin_l_uvab;
}



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
		torch::Tensor R_base_offsets
){
    const uint32_t ab_size              = (uint32_t) F.size(1);
    const uint32_t l_in_max_net_bound   = (uint32_t) C_offsets.size(1);

    const uint32_t* const __restrict__ L_out_list_ptr          = (uint32_t*) L_out_list.data_ptr();
    const uint32_t* const __restrict__ L_in_list_ptr           = (uint32_t*) L_in_list.data_ptr();
    const uint32_t* const __restrict__ u_sizes_ptr             = (uint32_t*) u_sizes.data_ptr();
    const uint32_t* const __restrict__ v_sizes_ptr             = (uint32_t*) v_sizes.data_ptr();
    const uint32_t* const __restrict__ output_base_offsets_ptr = (uint32_t*) output_base_offsets.data_ptr();
    const uint32_t* const __restrict__ C_offsets_ptr           = (uint32_t*) C_offsets.data_ptr();
    const uint32_t* const __restrict__ F_base_offsets_ptr      = (uint32_t*) F_base_offsets.data_ptr();
    const uint32_t* const __restrict__ R_base_offsets_ptr      = (uint32_t*) R_base_offsets.data_ptr();

    dim3 blocks(L_out_list.size(0), L_in_list.size(0));

    if (output.dtype() == torch::kFloat64){
              double* const __restrict__ output_ptr = (double*) output.data_ptr();
        const double* const __restrict__ W_ptr      = (double*) W.data_ptr();
        const double* const __restrict__ C_ptr      = (double*) C.data_ptr();
        const double* const __restrict__ F_ptr      = (double*) F.data_ptr();
        const double* const __restrict__ Y_ptr      = (double*) Y.data_ptr();
        const double* const __restrict__ R_ptr      = (double*) R.data_ptr();

        forward_parent_cuda_kernel<double><<<blocks, 1>>>(output_ptr, W_ptr, C_ptr, F_ptr, Y_ptr, R_ptr,
                                                                    L_out_list_ptr, L_in_list_ptr, u_sizes_ptr, v_sizes_ptr,
                                                                    output_base_offsets_ptr, C_offsets_ptr, F_base_offsets_ptr, R_base_offsets_ptr,
                                                                    ab_size, l_in_max_net_bound);
    }
    else if (output.dtype() == torch::kFloat32){
              float* const __restrict__ output_ptr = (float*) output.data_ptr();
        const float* const __restrict__ W_ptr      = (float*) W.data_ptr();
        const float* const __restrict__ C_ptr      = (float*) C.data_ptr();
        const float* const __restrict__ F_ptr      = (float*) F.data_ptr();
        const float* const __restrict__ Y_ptr      = (float*) Y.data_ptr();
        const float* const __restrict__ R_ptr      = (float*) R.data_ptr();

        forward_parent_cuda_kernel<float><<<blocks, 1>>>(output_ptr, W_ptr, C_ptr, F_ptr, Y_ptr, R_ptr,
                                                                   L_out_list_ptr, L_in_list_ptr, u_sizes_ptr, v_sizes_ptr,
                                                                   output_base_offsets_ptr, C_offsets_ptr, F_base_offsets_ptr, R_base_offsets_ptr,
                                                                   ab_size, l_in_max_net_bound);
    }
}


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
		torch::Tensor R_base_offsets
){
    const uint32_t ab_size              = (uint32_t) G.size(1);
    const uint32_t l_in_max_net_bound   = (uint32_t) C_offsets.size(1);

    const uint32_t* const __restrict__ L_out_list_ptr          = (uint32_t*) L_out_list.data_ptr();
    const uint32_t* const __restrict__ L_in_list_ptr           = (uint32_t*) L_in_list.data_ptr();
    const uint32_t* const __restrict__ u_sizes_ptr             = (uint32_t*) u_sizes.data_ptr();
    const uint32_t* const __restrict__ v_sizes_ptr             = (uint32_t*) v_sizes.data_ptr();
    const uint32_t* const __restrict__ output_base_offsets_ptr = (uint32_t*) output_base_offsets.data_ptr();
    const uint32_t* const __restrict__ C_offsets_ptr           = (uint32_t*) C_offsets.data_ptr();
    const uint32_t* const __restrict__ G_base_offsets_ptr      = (uint32_t*) G_base_offsets.data_ptr();
    const uint32_t* const __restrict__ R_base_offsets_ptr      = (uint32_t*) R_base_offsets.data_ptr();

    dim3 blocks(L_out_list.size(0), L_in_list.size(0));

    if (output.dtype() == torch::kFloat64){
              double* const __restrict__ output_ptr  = (double*) output.data_ptr();
        const double* const __restrict__ W_ptr       = (double*) W.data_ptr();
        const double* const __restrict__ C_ptr       = (double*) C.data_ptr();
        const double* const __restrict__ G_ptr       = (double*) G.data_ptr();
        const double* const __restrict__ Y_ptr       = (double*) Y.data_ptr();
        const double* const __restrict__ R_ptr       = (double*) R.data_ptr();

        backward_F_parent_cuda_kernel<double><<<blocks, 1>>>(output_ptr, W_ptr, C_ptr, G_ptr, Y_ptr, R_ptr,
                                                                       L_out_list_ptr, L_in_list_ptr, u_sizes_ptr, v_sizes_ptr,
                                                                       output_base_offsets_ptr, C_offsets_ptr, G_base_offsets_ptr, R_base_offsets_ptr,
                                                                       ab_size, l_in_max_net_bound);
    }
    else if (output.dtype() == torch::kFloat32){
              float* const __restrict__ output_ptr  = (float*) output.data_ptr();
        const float* const __restrict__ W_ptr       = (float*) W.data_ptr();
        const float* const __restrict__ C_ptr       = (float*) C.data_ptr();
        const float* const __restrict__ G_ptr       = (float*) G.data_ptr();
        const float* const __restrict__ Y_ptr       = (float*) Y.data_ptr();
        const float* const __restrict__ R_ptr       = (float*) R.data_ptr();

        backward_F_parent_cuda_kernel<float><<<blocks, 1>>>(output_ptr, W_ptr, C_ptr, G_ptr, Y_ptr, R_ptr,
                                                                      L_out_list_ptr, L_in_list_ptr, u_sizes_ptr, v_sizes_ptr,
                                                                      output_base_offsets_ptr, C_offsets_ptr, G_base_offsets_ptr, R_base_offsets_ptr,
                                                                      ab_size, l_in_max_net_bound);
    }
}


void backward_R_cuda(
        torch::Tensor output,		        // allocated in higher level wrapper
		torch::Tensor W,                    // layer specific
		torch::Tensor C,					// network wide (sampling is layer specific)
		torch::Tensor G,					// passed by pipeline backward pipeline
		torch::Tensor F,					// layer specific, stored in buffer for backward pass during forward pass
		torch::Tensor Y,					// object is network wide (sampling is layer specific)
		torch::Tensor L_out_list,			// layer specific
		torch::Tensor L_in_list,			// layer specific
		torch::Tensor u_sizes,				// layer specific
		torch::Tensor v_sizes,				// layer specific
		torch::Tensor output_base_offsets,	// network wide
		torch::Tensor C_offsets,			// network wide
		torch::Tensor G_base_offsets,		// layer specific
		torch::Tensor F_base_offsets 		// layer specific
) {
    const uint32_t ab_size            = (uint32_t) F.size(1);
    const uint32_t l_in_max_net_bound = (uint32_t) C_offsets.size(1);

    const uint32_t* const __restrict__ L_out_list_ptr          = (uint32_t*) L_out_list.data_ptr();
    const uint32_t* const __restrict__ L_in_list_ptr           = (uint32_t*) L_in_list.data_ptr();
    const uint32_t* const __restrict__ u_sizes_ptr             = (uint32_t*) u_sizes.data_ptr();
    const uint32_t* const __restrict__ v_sizes_ptr             = (uint32_t*) v_sizes.data_ptr();
    const uint32_t* const __restrict__ output_base_offsets_ptr = (uint32_t*) output_base_offsets.data_ptr();
    const uint32_t* const __restrict__ C_offsets_ptr           = (uint32_t*) C_offsets.data_ptr();
    const uint32_t* const __restrict__ G_base_offsets_ptr      = (uint32_t*) G_base_offsets.data_ptr();
    const uint32_t* const __restrict__ F_base_offsets_ptr      = (uint32_t*) F_base_offsets.data_ptr();

    dim3 blocks(L_out_list.size(0), L_in_list.size(0));

    if (output.dtype() == torch::kFloat64){
              double* const __restrict__ output_ptr = (double*) output.data_ptr();
        const double* const __restrict__ W_ptr      = (double*) W.data_ptr();
        const double* const __restrict__ C_ptr      = (double*) C.data_ptr();
        const double* const __restrict__ G_ptr      = (double*) G.data_ptr();
        const double* const __restrict__ F_ptr      = (double*) F.data_ptr();
        const double* const __restrict__ Y_ptr      = (double*) Y.data_ptr();

        backward_R_parent_cuda_kernel<double><<<blocks, 1>>>(output_ptr, W_ptr, C_ptr, G_ptr, F_ptr, Y_ptr,
                                                             L_out_list_ptr, L_in_list_ptr, u_sizes_ptr, v_sizes_ptr,
                                                             output_base_offsets_ptr, C_offsets_ptr, G_base_offsets_ptr, F_base_offsets_ptr,
                                                             ab_size, l_in_max_net_bound);
    }
    else if (output.dtype() == torch::kFloat32){
              float* const __restrict__ output_ptr = (float*) output.data_ptr();
        const float* const __restrict__ W_ptr      = (float*) W.data_ptr();
        const float* const __restrict__ C_ptr      = (float*) C.data_ptr();
        const float* const __restrict__ G_ptr      = (float*) G.data_ptr();
        const float* const __restrict__ F_ptr      = (float*) F.data_ptr();
        const float* const __restrict__ Y_ptr      = (float*) Y.data_ptr();

        backward_R_parent_cuda_kernel<float><<<blocks, 1>>>(output_ptr, W_ptr, C_ptr, G_ptr, F_ptr, Y_ptr,
                                                            L_out_list_ptr, L_in_list_ptr, u_sizes_ptr, v_sizes_ptr,
                                                            output_base_offsets_ptr, C_offsets_ptr, G_base_offsets_ptr, F_base_offsets_ptr,
                                                            ab_size, l_in_max_net_bound);
    }
}
