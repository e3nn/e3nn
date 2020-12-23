#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>


template<typename T>
__global__ void e3nn_normalization_cuda_kernel(T* const __restrict__ rsh, const size_t n_entries) {
	const size_t entry_pos = blockDim.x*blockIdx.x + threadIdx.x;                       // position of entry
	if (entry_pos >= n_entries) return;                                                 // terminate early if out-of-bound - last warp (of threads) can be partially filled

    // e3nn normalization: multiply by (-1)^L
    // L = floor(sqrt(idx))

    // +0.5f is needed to avoid numerical instability for perfect squares
    // _rn - round to the nearest
    // _rd - round down
    // rounding does NOT mean rounding to the integer in case of floats, it is rounding to the binary representation (as in finite precision)
    // x&1 == x%2
    if ( (__float2int_rd(__fsqrt_rn(__uint2float_rn(blockIdx.y) + 0.5f)) & 1) == 1) {
        rsh[blockIdx.y*n_entries + entry_pos] = -rsh[blockIdx.y*n_entries + entry_pos]; // compiler can optimize this part nicely, because of __restrict__
    }
}


void e3nn_normalization_cuda(torch::Tensor rsh) {
    const size_t lm_size    = rsh.size(0);
    const size_t n_entries  = rsh.size(1);

    const size_t threads_per_block = 32;
    dim3 numBlocks((n_entries + threads_per_block - 1)/threads_per_block, lm_size, 1);

    if (rsh.dtype() == torch::kFloat64) {
        e3nn_normalization_cuda_kernel<double><<<numBlocks, threads_per_block>>>((double*) rsh.data_ptr(), n_entries);
    }
    else {
        e3nn_normalization_cuda_kernel<float><<<numBlocks, threads_per_block>>>((float*) rsh.data_ptr(), n_entries);
    }
}


template<typename T>
void e3nn_normalization_cpp_kernel(T* const __restrict__ rsh, const size_t lm_size, const size_t n_entries) {
    size_t lm = 1, L = 1;
    while (lm < lm_size) {
        for (size_t m = 0; m < 2*L + 1; m++) {
            for (size_t entry_pos = 0; entry_pos < n_entries; entry_pos++) {
                rsh[lm*n_entries + entry_pos] = -rsh[lm*n_entries + entry_pos];
            }
            lm++;
        }
        lm += 2*L + 3; // 2*(L+2) - 1, -1 is to counter over-counting in prev loop
        L += 2;
    }
}


void e3nn_normalization_cpp(torch::Tensor rsh) {
    const size_t lm_size    = rsh.size(0);
    const size_t n_entries  = rsh.size(1);

    if (rsh.dtype() == torch::kFloat64) e3nn_normalization_cpp_kernel<double>((double*) rsh.data_ptr(), lm_size, n_entries);
    else /* torch::kFloat32 */          e3nn_normalization_cpp_kernel<float>((float*) rsh.data_ptr(), lm_size, n_entries);
}
