/*
    Calculates real spherical harmonics up to L=6 (inclusive) from Cartesian coordinates.
    Coordinates x, y, z are expected to form unit length vector.
*/

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

// 1./sqrt(pi) - on higher precision than actual execution of operation over double can provide
__device__ constexpr double RSQRT_PI() 		{ return 		0.564189583547756286948079451560772585844050629328998856844;}

/*
   Idea of the following list of constexpr functions is to compute coefficients once at compile time,
   and embed them as a direct access constants, same as if they were explicitly written (e.g. 5.5).
   Strictly speaking constexpr can, but not guarantee calculation of expression at compile time.
   Here all functions accept no parameters, so probability of fail scenario is extremely low.
   When in doubt, run:
        nvcc -ptx rsh_cuda.cu
   and check if constexpr computed in rsh_cuda.ptx file.

   Coefficients are either for a whole expression (e.g.: RSH_C11), or for monomial of z.
   Note: power marked for immediate multiplier: ..._z2 * z^2, but also (..._z2 * z^2 + ..._c) * z.
*/
__device__ constexpr double RSH_C00() 		{ return 		RSQRT_PI()/2.			  		;}

__device__ constexpr double RSH_C10() 		{ return 		RSQRT_PI()*sqrt(3.)/2.    		;}
__device__ constexpr double RSH_C11() 		{ return 		RSQRT_PI()*sqrt(3.)/2.    		;}

__device__ constexpr double RSH_C20_c() 	{ return 	   -RSQRT_PI()*sqrt(5.)/4.    		;}
__device__ constexpr double RSH_C20_z2() 	{ return 		RSQRT_PI()*sqrt(5.)*3./4.       ;}
__device__ constexpr double RSH_C21() 		{ return 		RSQRT_PI()*sqrt(15.)/2.   		;}
__device__ constexpr double RSH_C22() 		{ return 		RSQRT_PI()*sqrt(15.)/4.   		;}

__device__ constexpr double RSH_C30_c() 	{ return 	   -RSQRT_PI()*sqrt(7.)*3./4.  		;}
__device__ constexpr double RSH_C30_z2() 	{ return 		RSQRT_PI()*sqrt(7.)*5./4.  		;}
__device__ constexpr double RSH_C31_c() 	{ return 	   -RSQRT_PI()*sqrt(42.)/8.   		;}
__device__ constexpr double RSH_C31_z2() 	{ return 		RSQRT_PI()*sqrt(42.)*5./8. 		;}
__device__ constexpr double RSH_C32() 		{ return 		RSQRT_PI()*sqrt(105.)/4.  		;}
__device__ constexpr double RSH_C33() 		{ return 		RSQRT_PI()*sqrt(70.)/8.   		;}

__device__ constexpr double RSH_C40_c()		{ return 		RSQRT_PI()*9./16.          		;}
__device__ constexpr double RSH_C40_z2() 	{ return 	   -RSQRT_PI()*90./16.         		;}
__device__ constexpr double RSH_C40_z4() 	{ return 		RSQRT_PI()*105./16.        		;}
__device__ constexpr double RSH_C41_c() 	{ return 	   -RSQRT_PI()*sqrt(10.)*9./8.		;}
__device__ constexpr double RSH_C41_z2() 	{ return 		RSQRT_PI()*sqrt(10.)*21./8.		;}
__device__ constexpr double RSH_C42_c() 	{ return 	   -RSQRT_PI()*sqrt(5.)*3./8.  		;}
__device__ constexpr double RSH_C42_z2() 	{ return 		RSQRT_PI()*sqrt(5.)*21./8. 		;}
__device__ constexpr double RSH_C43() 		{ return 		RSQRT_PI()*sqrt(70.)*3./8. 		;}
__device__ constexpr double RSH_C44() 		{ return 		RSQRT_PI()*sqrt(35.)*3./16.		;}

__device__ constexpr double RSH_C50_c() 	{ return 		RSQRT_PI()*sqrt(11.)*15./16. 	;}
__device__ constexpr double RSH_C50_z2() 	{ return 	   -RSQRT_PI()*sqrt(11.)*70./16. 	;}
__device__ constexpr double RSH_C50_z4() 	{ return 		RSQRT_PI()*sqrt(11.)*63./16. 	;}
__device__ constexpr double RSH_C51_c() 	{ return 		RSQRT_PI()*sqrt(165.)/16.   	;}
__device__ constexpr double RSH_C51_z2() 	{ return 	   -RSQRT_PI()*sqrt(165.)*14./16.	;}
__device__ constexpr double RSH_C51_z4() 	{ return 		RSQRT_PI()*sqrt(165.)*21./16.	;}
__device__ constexpr double RSH_C52_c() 	{ return 	   -RSQRT_PI()*sqrt(1155.)/8.   	;}
__device__ constexpr double RSH_C52_z2() 	{ return 		RSQRT_PI()*sqrt(1155.)*3./8. 	;}
__device__ constexpr double RSH_C53_c() 	{ return 	   -RSQRT_PI()*sqrt(770.)/32.   	;}
__device__ constexpr double RSH_C53_z2() 	{ return 		RSQRT_PI()*sqrt(770.)*9./32. 	;}
__device__ constexpr double RSH_C54() 		{ return 		RSQRT_PI()*sqrt(385.)*3./16.  	;}
__device__ constexpr double RSH_C55() 		{ return 		RSQRT_PI()*sqrt(154.)*3./32. 	;}

__device__ constexpr double RSH_C60_c() 	{ return 	   -RSQRT_PI()*sqrt(13.)*5./32. 	;}
__device__ constexpr double RSH_C60_z2() 	{ return 		RSQRT_PI()*sqrt(13.)*105./32. 	;}
__device__ constexpr double RSH_C60_z4() 	{ return 	   -RSQRT_PI()*sqrt(13.)*315./32. 	;}
__device__ constexpr double RSH_C60_z6() 	{ return 		RSQRT_PI()*sqrt(13.)*231./32. 	;}
__device__ constexpr double RSH_C61_c() 	{ return 		RSQRT_PI()*sqrt(273.)*5./16.  	;}
__device__ constexpr double RSH_C61_z2() 	{ return 	   -RSQRT_PI()*sqrt(273.)*30./16.	;}
__device__ constexpr double RSH_C61_z4() 	{ return 		RSQRT_PI()*sqrt(273.)*33./16.	;}
__device__ constexpr double RSH_C62_c() 	{ return 		RSQRT_PI()*sqrt(2730.)/64.   	;}
__device__ constexpr double RSH_C62_z2() 	{ return 	   -RSQRT_PI()*sqrt(2730.)*18./64. 	;}
__device__ constexpr double RSH_C62_z4() 	{ return 		RSQRT_PI()*sqrt(2730.)*33./64. 	;}
__device__ constexpr double RSH_C63_c() 	{ return 	   -RSQRT_PI()*sqrt(2730.)*3./32.   ;}
__device__ constexpr double RSH_C63_z2() 	{ return 		RSQRT_PI()*sqrt(2730.)*11./32. 	;}
__device__ constexpr double RSH_C64_c() 	{ return 	   -RSQRT_PI()*sqrt(91.)*3./32.  	;}
__device__ constexpr double RSH_C64_z2() 	{ return 		RSQRT_PI()*sqrt(91.)*33./32.  	;}
__device__ constexpr double RSH_C65() 		{ return 		RSQRT_PI()*sqrt(2002.)*3./32. 	;}
__device__ constexpr double RSH_C66() 		{ return 		RSQRT_PI()*sqrt(6006.)/64.  	;}


/*
    Compressed sin^m(theta)*[exp^(-i*m*phi) - exp^(i*m*phi)] and sin^m(theta)*[exp^(-i*m*phi) + exp^(i*m*phi)].
    These are shared multipliers for multiple L.

    __forceinline__ forces body of the function to be substituted in the place of the call.
    It proportionally enlarges executable size, but on the other hand saves time otherwise required to resolve function call.
*/
__device__ __forceinline__ double f_phi_n6(const double x, const double y) { const double x2 = x*x, y2 = y*y; 					return x * y * (3.*x2 - y2) * (x2 - 3.*y2) * 2. ; }
__device__ __forceinline__ double f_phi_n5(const double x, const double y) { const double x2 = x*x, y2 = y*y; 					return y * (y2*y2 + 5.*x2 * (x2 - 2.*y2))	    ; }
__device__ __forceinline__ double f_phi_n4(const double x, const double y) { 													return x * y * (x + y) * (x - y) * 4.		    ; }
__device__ __forceinline__ double f_phi_n3(const double x, const double y) { 													return y * (3.*x*x - y*y)				        ; }
__device__ __forceinline__ double f_phi_n2(const double x, const double y) { 													return x * y * 2.							    ; }
__device__ __forceinline__ double f_phi_n1(const double x, const double y) { 													return y								        ; }

__device__ __forceinline__ double f_phi_p1(const double x, const double y) { 													return x								        ; }
__device__ __forceinline__ double f_phi_p2(const double x, const double y) { 													return (x + y) * (x - y)				        ; }
__device__ __forceinline__ double f_phi_p3(const double x, const double y) { 													return x * (x*x - 3.*y*y)				        ; }
__device__ __forceinline__ double f_phi_p4(const double x, const double y) { const double x2 = x*x, y2 = y*y; 					return x2 * (x2 - 6.*y2) + y2*y2			    ; }
__device__ __forceinline__ double f_phi_p5(const double x, const double y) { const double x2 = x*x, y2 = y*y; 					return x * (x2*x2 + 5.*y2 * (y2 - 2.*x2))	    ; }
__device__ __forceinline__ double f_phi_p6(const double x, const double y) { const double x2 = x*x, y2 = y*y, dx2y2 = x2 - y2; 	return dx2y2 * (dx2y2*dx2y2 - 12.*x2*y2)        ; }


/*
    Polynoms in z.
*/
__device__ __forceinline__ double p_c  (const double z, const double c0)                                                    {                           return c0                                   ; }
__device__ __forceinline__ double p_z  (const double z, const double c1)                                                    {                           return c1 * z                               ; }
__device__ __forceinline__ double p_z2 (const double z, const double c0, const double c2)                                   {                           return c0 + c2 * z * z                      ; }
__device__ __forceinline__ double p_z2z(const double z, const double c0, const double c2)                                   {                           return (c0 + c2 * z * z) * z                ; }
__device__ __forceinline__ double p_z4 (const double z, const double c0, const double c2, const double c4)                  { const double z2 = z*z;    return c0 + (c2 + c4 * z2) * z2             ; }
__device__ __forceinline__ double p_z4z(const double z, const double c0, const double c2, const double c4)                  { const double z2 = z*z;    return (c0 + (c2 + c4 * z2) * z2) * z       ; }
__device__ __forceinline__ double p_z6 (const double z, const double c0, const double c2, const double c4, const double c6) { const double z2 = z*z;    return c0 + (c2 + (c4 + c6 * z2) * z2) * z2 ; }


/* handle special case of xyz = (0., 0., 0.) with additional multiplier, either 0. or 1. */
__device__ __forceinline__ double special(const double x, const double y, const double z) { return (double) (x != 0. || y != 0. || z != 0.); }


/*
    Functions for specific (L, m). Product of common multiplier from m and polynomial in z.
    Cases (L, 0) with L even and > 0, have additional multiplier to treat special case of input (0, 0, 0) - return 0.
    Multiplier is either 1. if any of x, y, z is different from 0, or 0. otherwise.
    It is constructed as a multiplier in opposite to if-else statement in order to avoid branch divergence.
*/

__device__ double sh00 (const double x, const double y, const double z) { return RSH_C00()                                                                              ; }

__device__ double sh1n1(const double x, const double y, const double z) { return f_phi_n1(x, y) * p_c  (z, RSH_C11())                                                   ; }
__device__ double sh10 (const double x, const double y, const double z) { return 				  p_z  (z, RSH_C10())                                                   ; }
__device__ double sh1p1(const double x, const double y, const double z) { return f_phi_p1(x, y) * p_c  (z, RSH_C11())                                                   ; }

__device__ double sh2n2(const double x, const double y, const double z) { return f_phi_n2(x, y) * p_c  (z, RSH_C22())                                                   ; }
__device__ double sh2n1(const double x, const double y, const double z) { return f_phi_n1(x, y) * p_z  (z, RSH_C21())                                                   ; }
__device__ double sh20 (const double x, const double y, const double z) { return special(x,y,z) * p_z2 (z, RSH_C20_c(), RSH_C20_z2())                                   ; }
__device__ double sh2p1(const double x, const double y, const double z) { return f_phi_p1(x, y) * p_z  (z, RSH_C21())                                                   ; }
__device__ double sh2p2(const double x, const double y, const double z) { return f_phi_p2(x, y) * p_c  (z, RSH_C22())                                                   ; }

__device__ double sh3n3(const double x, const double y, const double z) { return f_phi_n3(x, y) * p_c  (z, RSH_C33())                                                   ; }
__device__ double sh3n2(const double x, const double y, const double z) { return f_phi_n2(x, y) * p_z  (z, RSH_C32())                                                   ; }
__device__ double sh3n1(const double x, const double y, const double z) { return f_phi_n1(x, y) * p_z2 (z, RSH_C31_c(), RSH_C31_z2())                                   ; }
__device__ double sh30 (const double x, const double y, const double z) { return 				  p_z2z(z, RSH_C30_c(), RSH_C30_z2())                                   ; }
__device__ double sh3p1(const double x, const double y, const double z) { return f_phi_p1(x, y) * p_z2 (z, RSH_C31_c(), RSH_C31_z2())                                   ; }
__device__ double sh3p2(const double x, const double y, const double z) { return f_phi_p2(x, y) * p_z  (z, RSH_C32())                                                   ; }
__device__ double sh3p3(const double x, const double y, const double z) { return f_phi_p3(x, y) * p_c  (z, RSH_C33())                                                   ; }

__device__ double sh4n4(const double x, const double y, const double z) { return f_phi_n4(x, y) * p_c  (z, RSH_C44())                                                   ; }
__device__ double sh4n3(const double x, const double y, const double z) { return f_phi_n3(x, y) * p_z  (z, RSH_C43())                                                   ; }
__device__ double sh4n2(const double x, const double y, const double z) { return f_phi_n2(x, y) * p_z2 (z, RSH_C42_c(), RSH_C42_z2())                                   ; }
__device__ double sh4n1(const double x, const double y, const double z) { return f_phi_n1(x, y) * p_z2z(z, RSH_C41_c(), RSH_C41_z2())                                   ; }
__device__ double sh40 (const double x, const double y, const double z) { return special(x,y,z) * p_z4 (z, RSH_C40_c(), RSH_C40_z2(), RSH_C40_z4())                     ; }
__device__ double sh4p1(const double x, const double y, const double z) { return f_phi_p1(x, y) * p_z2z(z, RSH_C41_c(), RSH_C41_z2())                                   ; }
__device__ double sh4p2(const double x, const double y, const double z) { return f_phi_p2(x, y) * p_z2 (z, RSH_C42_c(), RSH_C42_z2())                                   ; }
__device__ double sh4p3(const double x, const double y, const double z) { return f_phi_p3(x, y) * p_z  (z, RSH_C43())                                                   ; }
__device__ double sh4p4(const double x, const double y, const double z) { return f_phi_p4(x, y) * p_c  (z, RSH_C44())                                                   ; }

__device__ double sh5n5(const double x, const double y, const double z) { return f_phi_n5(x, y) * p_c  (z, RSH_C55())                                                   ; }
__device__ double sh5n4(const double x, const double y, const double z) { return f_phi_n4(x, y) * p_z  (z, RSH_C54())                                                   ; }
__device__ double sh5n3(const double x, const double y, const double z) { return f_phi_n3(x, y) * p_z2 (z, RSH_C53_c(), RSH_C53_z2())                                   ; }
__device__ double sh5n2(const double x, const double y, const double z) { return f_phi_n2(x, y) * p_z2z(z, RSH_C52_c(), RSH_C52_z2())                                   ; }
__device__ double sh5n1(const double x, const double y, const double z) { return f_phi_n1(x, y) * p_z4 (z, RSH_C51_c(), RSH_C51_z2(), RSH_C51_z4())                     ; }
__device__ double sh50 (const double x, const double y, const double z) { return 				  p_z4z(z, RSH_C50_c(), RSH_C50_z2(), RSH_C50_z4())                     ; }
__device__ double sh5p1(const double x, const double y, const double z) { return f_phi_p1(x, y) * p_z4 (z, RSH_C51_c(), RSH_C51_z2(), RSH_C51_z4())                     ; }
__device__ double sh5p2(const double x, const double y, const double z) { return f_phi_p2(x, y) * p_z2z(z, RSH_C52_c(), RSH_C52_z2())                                   ; }
__device__ double sh5p3(const double x, const double y, const double z) { return f_phi_p3(x, y) * p_z2 (z, RSH_C53_c(), RSH_C53_z2())                                   ; }
__device__ double sh5p4(const double x, const double y, const double z) { return f_phi_p4(x, y) * p_z  (z, RSH_C54())                                                   ; }
__device__ double sh5p5(const double x, const double y, const double z) { return f_phi_p5(x, y) * p_c  (z, RSH_C55())                                                   ; }

__device__ double sh6n6(const double x, const double y, const double z) { return f_phi_n6(x, y) * p_c  (z, RSH_C66())                                                   ; }
__device__ double sh6n5(const double x, const double y, const double z) { return f_phi_n5(x, y) * p_z  (z, RSH_C65())                                                   ; }
__device__ double sh6n4(const double x, const double y, const double z) { return f_phi_n4(x, y) * p_z2 (z, RSH_C64_c(), RSH_C64_z2())                                   ; }
__device__ double sh6n3(const double x, const double y, const double z) { return f_phi_n3(x, y) * p_z2z(z, RSH_C63_c(), RSH_C63_z2())                                   ; }
__device__ double sh6n2(const double x, const double y, const double z) { return f_phi_n2(x, y) * p_z4 (z, RSH_C62_c(), RSH_C62_z2(), RSH_C62_z4())                     ; }
__device__ double sh6n1(const double x, const double y, const double z) { return f_phi_n1(x, y) * p_z4z(z, RSH_C61_c(), RSH_C61_z2(), RSH_C61_z4())                     ; }
__device__ double sh60 (const double x, const double y, const double z) { return special(x,y,z) * p_z6 (z, RSH_C60_c(), RSH_C60_z2(), RSH_C60_z4(), RSH_C60_z6())       ; }
__device__ double sh6p1(const double x, const double y, const double z) { return f_phi_p1(x, y) * p_z4z(z, RSH_C61_c(), RSH_C61_z2(), RSH_C61_z4())                     ; }
__device__ double sh6p2(const double x, const double y, const double z) { return f_phi_p2(x, y) * p_z4 (z, RSH_C62_c(), RSH_C62_z2(), RSH_C62_z4())                     ; }
__device__ double sh6p3(const double x, const double y, const double z) { return f_phi_p3(x, y) * p_z2z(z, RSH_C63_c(), RSH_C63_z2())                                   ; }
__device__ double sh6p4(const double x, const double y, const double z) { return f_phi_p4(x, y) * p_z2 (z, RSH_C64_c(), RSH_C64_z2())                                   ; }
__device__ double sh6p5(const double x, const double y, const double z) { return f_phi_p5(x, y) * p_z  (z, RSH_C65())                                                   ; }
__device__ double sh6p6(const double x, const double y, const double z) { return f_phi_p6(x, y) * p_c  (z, RSH_C66())                                                   ; }


// array of pointers to functions stored to "constant memory" (__constant__) in GPU - common for all blocks.
__constant__ double (*const fptr[]) (const double, const double, const double) = {
		                                          sh00, 											//                         0
		                                   sh1n1, sh10,  sh1p1, 									//                     1,  2,  3
		                            sh2n2, sh2n1, sh20,  sh2p1, sh2p2,								//                  4, 5,  6,  7,  8
		                     sh3n3, sh3n2, sh3n1, sh30,  sh3p1, sh3p2, sh3p3,					    //              9, 10, 11, 12, 13, 14, 15
		              sh4n4, sh4n3, sh4n2, sh4n1, sh40,  sh4p1, sh4p2, sh4p3, sh4p4,				//         16, 17, 18, 19, 20, 21, 22, 23, 24
		       sh5n5, sh5n4, sh5n3, sh5n2, sh5n1, sh50,  sh5p1, sh5p2, sh5p3, sh5p4, sh5p5,		    //     25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35
		sh6n6, sh6n5, sh6n4, sh6n3, sh6n2, sh6n1, sh60,  sh6p1, sh6p2, sh6p3, sh6p4, sh6p5, sh6p6   // 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48
	};


/*
    Preceding const means underlying data stays constant.
    Trailing const means that pointer to the data remains constant.
    __restrict__ makes a promise that underlying data can be accessed only with this pointer.
*/
__global__ void rsh_cuda_kernel(const double* const __restrict__ radii, double* const __restrict__ Ys, const unsigned int batch_size) {
	const unsigned int entry_pos = blockDim.x*blockIdx.x + threadIdx.x;     // position of entry in batch
	if (entry_pos >= batch_size) return;                                    // early terminate if outside the batch - last warp (of threads) can be only partially filled

	const double x = radii[3*entry_pos];                                    // "strided memory access" is generally not nice and severely drops throughput
	const double y = radii[3*entry_pos+1];                                  // padding to 4 and packing in double4 (single read transaction) showed no noticeable improvement (is scale to0 small measure?)
	const double z = radii[3*entry_pos+2];                                  // 100+ GB/s of throughput would be great, but even 3 GB/s does not make a bottleneck

    Ys[blockIdx.y*batch_size + entry_pos] = fptr[blockIdx.y](x, y, z);      // select and apply function, store result to the "global memory"
}


void real_spherical_harmonics_cuda(
        torch::Tensor output_placeholder,
        torch::Tensor radii) {
    const unsigned int filters = output_placeholder.size(0);
    const unsigned int batch_size = radii.size(0);

    double* const Ys_ptr = (double*) output_placeholder.data_ptr();
    const double* const radii_ptr = (const double*) radii.data_ptr();

    const unsigned int threads_per_block = 32;                                              // warp size in contemporary GPUs is 32 threads, this variable should be a multiple of warp size
    dim3 numBlocks((batch_size + threads_per_block - 1)/threads_per_block, filters, 1);     // batch_size/threads_per_block is fractional in general case - round it up

    rsh_cuda_kernel<<<numBlocks, threads_per_block>>>(radii_ptr, Ys_ptr, batch_size);
}
