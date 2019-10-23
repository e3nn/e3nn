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

// L = 0
__device__ constexpr double RSH_C00() 		{ return 		RSQRT_PI()/2.			  		    ;}

// L = 1
__device__ constexpr double RSH_C10() 		{ return 		RSQRT_PI()*sqrt(3.)/2.    		    ;}
__device__ constexpr double RSH_C11() 		{ return 		RSQRT_PI()*sqrt(3.)/2.    		    ;}

// L = 2
__device__ constexpr double RSH_C20_c() 	{ return 	   -RSQRT_PI()*sqrt(5.)/4.    		    ;}
__device__ constexpr double RSH_C20_z2() 	{ return 		RSQRT_PI()*sqrt(5.)*3./4.           ;}
__device__ constexpr double RSH_C21() 		{ return 		RSQRT_PI()*sqrt(15.)/2.   		    ;}
__device__ constexpr double RSH_C22() 		{ return 		RSQRT_PI()*sqrt(15.)/4.   		    ;}

// L = 3
__device__ constexpr double RSH_C30_c() 	{ return 	   -RSQRT_PI()*sqrt(7.)*3./4.  		    ;}
__device__ constexpr double RSH_C30_z2() 	{ return 		RSQRT_PI()*sqrt(7.)*5./4.  		    ;}
__device__ constexpr double RSH_C31_c() 	{ return 	   -RSQRT_PI()*sqrt(42.)/8.   		    ;}
__device__ constexpr double RSH_C31_z2() 	{ return 		RSQRT_PI()*sqrt(42.)*5./8. 		    ;}
__device__ constexpr double RSH_C32() 		{ return 		RSQRT_PI()*sqrt(105.)/4.  		    ;}
__device__ constexpr double RSH_C33() 		{ return 		RSQRT_PI()*sqrt(70.)/8.   		    ;}

// L = 4
__device__ constexpr double RSH_C40_c()		{ return 		RSQRT_PI()*9./16.          		    ;}
__device__ constexpr double RSH_C40_z2() 	{ return 	   -RSQRT_PI()*90./16.         		    ;}
__device__ constexpr double RSH_C40_z4() 	{ return 		RSQRT_PI()*105./16.        		    ;}
__device__ constexpr double RSH_C41_c() 	{ return 	   -RSQRT_PI()*sqrt(10.)*9./8.		    ;}
__device__ constexpr double RSH_C41_z2() 	{ return 		RSQRT_PI()*sqrt(10.)*21./8.		    ;}
__device__ constexpr double RSH_C42_c() 	{ return 	   -RSQRT_PI()*sqrt(5.)*3./8.  		    ;}
__device__ constexpr double RSH_C42_z2() 	{ return 		RSQRT_PI()*sqrt(5.)*21./8. 		    ;}
__device__ constexpr double RSH_C43() 		{ return 		RSQRT_PI()*sqrt(70.)*3./8. 		    ;}
__device__ constexpr double RSH_C44() 		{ return 		RSQRT_PI()*sqrt(35.)*3./16.		    ;}

// L = 5
__device__ constexpr double RSH_C50_c() 	{ return 		RSQRT_PI()*sqrt(11.)*15./16. 	    ;}
__device__ constexpr double RSH_C50_z2() 	{ return 	   -RSQRT_PI()*sqrt(11.)*70./16. 	    ;}
__device__ constexpr double RSH_C50_z4() 	{ return 		RSQRT_PI()*sqrt(11.)*63./16. 	    ;}
__device__ constexpr double RSH_C51_c() 	{ return 		RSQRT_PI()*sqrt(165.)/16.   	    ;}
__device__ constexpr double RSH_C51_z2() 	{ return 	   -RSQRT_PI()*sqrt(165.)*14./16.	    ;}
__device__ constexpr double RSH_C51_z4() 	{ return 		RSQRT_PI()*sqrt(165.)*21./16.	    ;}
__device__ constexpr double RSH_C52_c() 	{ return 	   -RSQRT_PI()*sqrt(1155.)/8.   	    ;}
__device__ constexpr double RSH_C52_z2() 	{ return 		RSQRT_PI()*sqrt(1155.)*3./8. 	    ;}
__device__ constexpr double RSH_C53_c() 	{ return 	   -RSQRT_PI()*sqrt(770.)/32.   	    ;}
__device__ constexpr double RSH_C53_z2() 	{ return 		RSQRT_PI()*sqrt(770.)*9./32. 	    ;}
__device__ constexpr double RSH_C54() 		{ return 		RSQRT_PI()*sqrt(385.)*3./16.  	    ;}
__device__ constexpr double RSH_C55() 		{ return 		RSQRT_PI()*sqrt(154.)*3./32. 	    ;}

// L = 6
__device__ constexpr double RSH_C60_c() 	{ return 	   -RSQRT_PI()*sqrt(13.)*5./32. 	    ;}
__device__ constexpr double RSH_C60_z2() 	{ return 		RSQRT_PI()*sqrt(13.)*105./32. 	    ;}
__device__ constexpr double RSH_C60_z4() 	{ return 	   -RSQRT_PI()*sqrt(13.)*315./32. 	    ;}
__device__ constexpr double RSH_C60_z6() 	{ return 		RSQRT_PI()*sqrt(13.)*231./32. 	    ;}
__device__ constexpr double RSH_C61_c() 	{ return 		RSQRT_PI()*sqrt(273.)*5./16.  	    ;}
__device__ constexpr double RSH_C61_z2() 	{ return 	   -RSQRT_PI()*sqrt(273.)*30./16.	    ;}
__device__ constexpr double RSH_C61_z4() 	{ return 		RSQRT_PI()*sqrt(273.)*33./16.	    ;}
__device__ constexpr double RSH_C62_c() 	{ return 		RSQRT_PI()*sqrt(2730.)/64.   	    ;}
__device__ constexpr double RSH_C62_z2() 	{ return 	   -RSQRT_PI()*sqrt(2730.)*18./64. 	    ;}
__device__ constexpr double RSH_C62_z4() 	{ return 		RSQRT_PI()*sqrt(2730.)*33./64. 	    ;}
__device__ constexpr double RSH_C63_c() 	{ return 	   -RSQRT_PI()*sqrt(2730.)*3./32.       ;}
__device__ constexpr double RSH_C63_z2() 	{ return 		RSQRT_PI()*sqrt(2730.)*11./32. 	    ;}
__device__ constexpr double RSH_C64_c() 	{ return 	   -RSQRT_PI()*sqrt(91.)*3./32.  	    ;}
__device__ constexpr double RSH_C64_z2() 	{ return 		RSQRT_PI()*sqrt(91.)*33./32.  	    ;}
__device__ constexpr double RSH_C65() 		{ return 		RSQRT_PI()*sqrt(2002.)*3./32. 	    ;}
__device__ constexpr double RSH_C66() 		{ return 		RSQRT_PI()*sqrt(6006.)/64.  	    ;}

// L = 7
__device__ constexpr double RSH_C70_c() 	{ return 	   -RSQRT_PI()*sqrt(15.)*35./32. 	    ;}
__device__ constexpr double RSH_C70_z2() 	{ return 	    RSQRT_PI()*sqrt(15.)*315./32. 	    ;}
__device__ constexpr double RSH_C70_z4() 	{ return 	   -RSQRT_PI()*sqrt(15.)*693./32. 	    ;}
__device__ constexpr double RSH_C70_z6() 	{ return 	    RSQRT_PI()*sqrt(15.)*429./32. 	    ;}
__device__ constexpr double RSH_C71_c() 	{ return 	   -RSQRT_PI()*sqrt(105.)*5./64. 	    ;}
__device__ constexpr double RSH_C71_z2() 	{ return 	    RSQRT_PI()*sqrt(105.)*135./64. 	    ;}
__device__ constexpr double RSH_C71_z4() 	{ return 	   -RSQRT_PI()*sqrt(105.)*495./64. 	    ;}
__device__ constexpr double RSH_C71_z6() 	{ return 	    RSQRT_PI()*sqrt(105.)*429./64. 	    ;}
__device__ constexpr double RSH_C72_c() 	{ return 	    RSQRT_PI()*sqrt(70.)*45./64. 	    ;}
__device__ constexpr double RSH_C72_z2() 	{ return 	   -RSQRT_PI()*sqrt(70.)*330./64. 	    ;}
__device__ constexpr double RSH_C72_z4() 	{ return 	    RSQRT_PI()*sqrt(70.)*429./64. 	    ;}
__device__ constexpr double RSH_C73_c() 	{ return 	    RSQRT_PI()*sqrt(35.)*9./64. 	    ;}
__device__ constexpr double RSH_C73_z2() 	{ return 	   -RSQRT_PI()*sqrt(35.)*198./64. 	    ;}
__device__ constexpr double RSH_C73_z4() 	{ return 	    RSQRT_PI()*sqrt(35.)*429./64. 	    ;}
__device__ constexpr double RSH_C74_c() 	{ return 	   -RSQRT_PI()*sqrt(385.)*9./32. 	    ;}
__device__ constexpr double RSH_C74_z2() 	{ return 	    RSQRT_PI()*sqrt(385.)*39./32. 	    ;}
__device__ constexpr double RSH_C75_c() 	{ return 	   -RSQRT_PI()*sqrt(385.)*3./64. 	    ;}
__device__ constexpr double RSH_C75_z2() 	{ return 	    RSQRT_PI()*sqrt(385.)*39./64. 	    ;}
__device__ constexpr double RSH_C76() 		{ return 		RSQRT_PI()*sqrt(10010.)*3./64. 	    ;}
__device__ constexpr double RSH_C77() 		{ return 		RSQRT_PI()*sqrt(715.)*3./64.  	    ;}

// L = 8
__device__ constexpr double RSH_C80_c() 	{ return 	    RSQRT_PI()*sqrt(17.)*35./256. 	    ;}
__device__ constexpr double RSH_C80_z2() 	{ return 	   -RSQRT_PI()*sqrt(17.)*1260./256.     ;}
__device__ constexpr double RSH_C80_z4() 	{ return 	    RSQRT_PI()*sqrt(17.)*6930./256.     ;}
__device__ constexpr double RSH_C80_z6() 	{ return 	   -RSQRT_PI()*sqrt(17.)*12012./256.    ;}
__device__ constexpr double RSH_C80_z8() 	{ return 	    RSQRT_PI()*sqrt(17.)*6435./256.     ;}
__device__ constexpr double RSH_C81_c() 	{ return 	   -RSQRT_PI()*sqrt(17.)*105./64. 	    ;}
__device__ constexpr double RSH_C81_z2() 	{ return 	    RSQRT_PI()*sqrt(17.)*1155./64. 	    ;}
__device__ constexpr double RSH_C81_z4() 	{ return 	   -RSQRT_PI()*sqrt(17.)*3003./64. 	    ;}
__device__ constexpr double RSH_C81_z6() 	{ return 	    RSQRT_PI()*sqrt(17.)*2145./64. 	    ;}
__device__ constexpr double RSH_C82_c() 	{ return 	   -RSQRT_PI()*sqrt(1190.)*3./128. 	    ;}
__device__ constexpr double RSH_C82_z2() 	{ return 	    RSQRT_PI()*sqrt(1190.)*99./128. 	;}
__device__ constexpr double RSH_C82_z4() 	{ return 	   -RSQRT_PI()*sqrt(1190.)*429./128. 	;}
__device__ constexpr double RSH_C82_z6() 	{ return 	    RSQRT_PI()*sqrt(1190.)*429./128. 	;}
__device__ constexpr double RSH_C83_c() 	{ return 	    RSQRT_PI()*sqrt(19635.)*3./64. 	    ;}
__device__ constexpr double RSH_C83_z2() 	{ return 	   -RSQRT_PI()*sqrt(19635.)*26./64. 	;}
__device__ constexpr double RSH_C83_z4() 	{ return 	    RSQRT_PI()*sqrt(19635.)*39./64. 	;}
__device__ constexpr double RSH_C84_c() 	{ return 	    RSQRT_PI()*sqrt(1309.)*3./128. 	    ;}
__device__ constexpr double RSH_C84_z2() 	{ return 	   -RSQRT_PI()*sqrt(1309.)*78./128. 	;}
__device__ constexpr double RSH_C84_z4() 	{ return 	    RSQRT_PI()*sqrt(1309.)*195./128. 	;}
__device__ constexpr double RSH_C85_c() 	{ return 	   -RSQRT_PI()*sqrt(17017.)*3./64. 	    ;}
__device__ constexpr double RSH_C85_z2() 	{ return 	    RSQRT_PI()*sqrt(17017.)*15./64. 	;}
__device__ constexpr double RSH_C86_c() 	{ return 	   -RSQRT_PI()*sqrt(14586.)/128. 	    ;}
__device__ constexpr double RSH_C86_z2() 	{ return 	    RSQRT_PI()*sqrt(14586.)*15./128. 	;}
__device__ constexpr double RSH_C87() 		{ return 		RSQRT_PI()*sqrt(12155.)*3./64. 	    ;}
__device__ constexpr double RSH_C88() 		{ return 		RSQRT_PI()*sqrt(12155.)*3./256.  	;}

// L = 9
__device__ constexpr double RSH_C90_c() 	{ return 	    RSQRT_PI()*sqrt(19.)*315./256. 	    ;}
__device__ constexpr double RSH_C90_z2() 	{ return 	   -RSQRT_PI()*sqrt(19.)*4620./256.     ;}
__device__ constexpr double RSH_C90_z4() 	{ return 	    RSQRT_PI()*sqrt(19.)*18018./256.    ;}
__device__ constexpr double RSH_C90_z6() 	{ return 	   -RSQRT_PI()*sqrt(19.)*25740./256.    ;}
__device__ constexpr double RSH_C90_z8() 	{ return 	    RSQRT_PI()*sqrt(19.)*12155./256.    ;}
__device__ constexpr double RSH_C91_c() 	{ return 	    RSQRT_PI()*sqrt(95.)*21./256. 	    ;}
__device__ constexpr double RSH_C91_z2() 	{ return 	   -RSQRT_PI()*sqrt(95.)*924./256. 	    ;}
__device__ constexpr double RSH_C91_z4() 	{ return 	    RSQRT_PI()*sqrt(95.)*6006./256. 	;}
__device__ constexpr double RSH_C91_z6() 	{ return 	   -RSQRT_PI()*sqrt(95.)*12012./256. 	;}
__device__ constexpr double RSH_C91_z8() 	{ return 	    RSQRT_PI()*sqrt(95.)*7293./256. 	;}
__device__ constexpr double RSH_C92_c() 	{ return 	   -RSQRT_PI()*sqrt(2090.)*21./128. 	;}
__device__ constexpr double RSH_C92_z2() 	{ return 	    RSQRT_PI()*sqrt(2090.)*273./128. 	;}
__device__ constexpr double RSH_C92_z4() 	{ return 	   -RSQRT_PI()*sqrt(2090.)*819./128. 	;}
__device__ constexpr double RSH_C92_z6() 	{ return 	    RSQRT_PI()*sqrt(2090.)*663./128. 	;}
__device__ constexpr double RSH_C93_c() 	{ return 	   -RSQRT_PI()*sqrt(43890.)/256. 	    ;}
__device__ constexpr double RSH_C93_z2() 	{ return 	    RSQRT_PI()*sqrt(43890.)*39./256. 	;}
__device__ constexpr double RSH_C93_z4() 	{ return 	   -RSQRT_PI()*sqrt(43890.)*195./256. 	;}
__device__ constexpr double RSH_C93_z6() 	{ return 	    RSQRT_PI()*sqrt(43890.)*221./256. 	;}
__device__ constexpr double RSH_C94_c() 	{ return 	    RSQRT_PI()*sqrt(95095.)*3./128. 	;}
__device__ constexpr double RSH_C94_z2() 	{ return 	   -RSQRT_PI()*sqrt(95095.)*30./128. 	;}
__device__ constexpr double RSH_C94_z4() 	{ return 	    RSQRT_PI()*sqrt(95095.)*51./128. 	;}
__device__ constexpr double RSH_C95_c() 	{ return 	    RSQRT_PI()*sqrt(5434.)*3./256. 	    ;}
__device__ constexpr double RSH_C95_z2() 	{ return 	   -RSQRT_PI()*sqrt(5434.)*90./256. 	;}
__device__ constexpr double RSH_C95_z4() 	{ return 	    RSQRT_PI()*sqrt(5434.)*255./256. 	;}
__device__ constexpr double RSH_C96_c() 	{ return 	   -RSQRT_PI()*sqrt(81510.)*3./128. 	;}
__device__ constexpr double RSH_C96_z2() 	{ return 	    RSQRT_PI()*sqrt(81510.)*17./128. 	;}
__device__ constexpr double RSH_C97_c() 	{ return 	   -RSQRT_PI()*sqrt(27170.)*3./512. 	;}
__device__ constexpr double RSH_C97_z2() 	{ return 		RSQRT_PI()*sqrt(27170.)*51./512. 	;}
__device__ constexpr double RSH_C98() 		{ return 		RSQRT_PI()*sqrt(230945.)*3./256.  	;}
__device__ constexpr double RSH_C99() 		{ return 		RSQRT_PI()*sqrt(461890.)/512.  	    ;}

// L = 10
__device__ constexpr double RSH_CA0_c() 	{ return 	   -RSQRT_PI()*sqrt(21.)*63./512. 	    ;}
__device__ constexpr double RSH_CA0_z2() 	{ return 	    RSQRT_PI()*sqrt(21.)*3465./512.     ;}
__device__ constexpr double RSH_CA0_z4() 	{ return 	   -RSQRT_PI()*sqrt(21.)*30030./512.    ;}
__device__ constexpr double RSH_CA0_z6() 	{ return 	    RSQRT_PI()*sqrt(21.)*90090./512.    ;}
__device__ constexpr double RSH_CA0_z8() 	{ return 	   -RSQRT_PI()*sqrt(21.)*109395./512.   ;}
__device__ constexpr double RSH_CA0_zA() 	{ return 	    RSQRT_PI()*sqrt(21.)*46189./512.    ;}
__device__ constexpr double RSH_CA1_c() 	{ return 	    RSQRT_PI()*sqrt(1155.)*63./256. 	;}
__device__ constexpr double RSH_CA1_z2() 	{ return 	   -RSQRT_PI()*sqrt(1155.)*1092./256. 	;}
__device__ constexpr double RSH_CA1_z4() 	{ return 	    RSQRT_PI()*sqrt(1155.)*4914./256. 	;}
__device__ constexpr double RSH_CA1_z6() 	{ return 	   -RSQRT_PI()*sqrt(1155.)*7956./256. 	;}
__device__ constexpr double RSH_CA1_z8() 	{ return 	    RSQRT_PI()*sqrt(1155.)*4199./256. 	;}
__device__ constexpr double RSH_CA2_c() 	{ return 	    RSQRT_PI()*sqrt(385.)*21./512. 	    ;}
__device__ constexpr double RSH_CA2_z2() 	{ return 	   -RSQRT_PI()*sqrt(385.)*1092./512. 	;}
__device__ constexpr double RSH_CA2_z4() 	{ return 	    RSQRT_PI()*sqrt(385.)*8190./512. 	;}
__device__ constexpr double RSH_CA2_z6() 	{ return 	   -RSQRT_PI()*sqrt(385.)*18564./512. 	;}
__device__ constexpr double RSH_CA2_z8() 	{ return 	    RSQRT_PI()*sqrt(385.)*12597./512. 	;}
__device__ constexpr double RSH_CA3_c() 	{ return 	   -RSQRT_PI()*sqrt(10010.)*21./256. 	;}
__device__ constexpr double RSH_CA3_z2() 	{ return 	    RSQRT_PI()*sqrt(10010.)*315./256. 	;}
__device__ constexpr double RSH_CA3_z4() 	{ return 	   -RSQRT_PI()*sqrt(10010.)*1071./256. 	;}
__device__ constexpr double RSH_CA3_z6() 	{ return 	    RSQRT_PI()*sqrt(10010.)*969./256. 	;}
__device__ constexpr double RSH_CA4_c() 	{ return 	   -RSQRT_PI()*sqrt(5005.)*3./256. 	    ;}
__device__ constexpr double RSH_CA4_z2() 	{ return 	    RSQRT_PI()*sqrt(5005.)*135./256. 	;}
__device__ constexpr double RSH_CA4_z4() 	{ return 	   -RSQRT_PI()*sqrt(5005.)*765./256. 	;}
__device__ constexpr double RSH_CA4_z6() 	{ return 	    RSQRT_PI()*sqrt(5005.)*969./256. 	;}
__device__ constexpr double RSH_CA5_c() 	{ return 	    RSQRT_PI()*sqrt(2002.)*45./256. 	;}
__device__ constexpr double RSH_CA5_z2() 	{ return 	   -RSQRT_PI()*sqrt(2002.)*510./256. 	;}
__device__ constexpr double RSH_CA5_z4() 	{ return 	    RSQRT_PI()*sqrt(2002.)*969./256. 	;}
__device__ constexpr double RSH_CA6_c() 	{ return 	    RSQRT_PI()*sqrt(10010.)*9./1024. 	;}
__device__ constexpr double RSH_CA6_z2() 	{ return 	   -RSQRT_PI()*sqrt(10010.)*306./1024. 	;}
__device__ constexpr double RSH_CA6_z4() 	{ return 	    RSQRT_PI()*sqrt(10010.)*969./1024. 	;}
__device__ constexpr double RSH_CA7_c() 	{ return 	   -RSQRT_PI()*sqrt(170170.)*9./512. 	;}
__device__ constexpr double RSH_CA7_z2() 	{ return 		RSQRT_PI()*sqrt(170170.)*57./512. 	;}
__device__ constexpr double RSH_CA8_c() 	{ return 	   -RSQRT_PI()*sqrt(255255.)/512.  	    ;}
__device__ constexpr double RSH_CA8_z2() 	{ return 		RSQRT_PI()*sqrt(255255.)*19./512.  	;}
__device__ constexpr double RSH_CA9() 		{ return 		RSQRT_PI()*sqrt(9699690.)/512.  	;}
__device__ constexpr double RSH_CAA() 		{ return 		RSQRT_PI()*sqrt(1939938.)/1024.  	;}


/*
    Compressed sin^m(theta)*[exp^(-i*m*phi) - exp^(i*m*phi)] and sin^m(theta)*[exp^(-i*m*phi) + exp^(i*m*phi)].
    These are shared multipliers for multiple L.

    __forceinline__ forces body of the function to be substituted in the place of the call.
    It proportionally enlarges executable size, but on the other hand saves time otherwise required to resolve function call.
*/
__device__ __forceinline__ double f_phi_nA(const double x, const double y) { const double x2 = x*x, y2 = y*y, dx2y2 = x2 - y2;
                                                                             const double dx2y2_sq = dx2y2*dx2y2;               return x * y * (10.*dx2y2_sq * (dx2y2_sq - 8.*x2*y2) + 32.*x2*x2*y2*y2) ; }
__device__ __forceinline__ double f_phi_n9(const double x, const double y) { const double x2 = x*x, y2 = y*y;
                                                                             const double x4 = x2*x2, y4 = y2*y2;               return y * (y4*y4 + 126.*x4*y4 + 9.*x4*x4 - x2*y2 * (36.*y4 + 84*x4))   ; }
__device__ __forceinline__ double f_phi_n8(const double x, const double y) { const double x2 = x*x, y2 = y*y, dx2y2 = x2 - y2;  return x * y * dx2y2 * (dx2y2*dx2y2 - 4.*x2*y2) * 8.                    ; }
__device__ __forceinline__ double f_phi_n7(const double x, const double y) { const double x2 = x*x, y2 = y*y; 			        return -y * (y2*y2 * (y2 - 21.*x2) + 7.*x2*x2 * (5.*y2 - x2))           ; }
__device__ __forceinline__ double f_phi_n6(const double x, const double y) { const double x2 = x*x, y2 = y*y; 				    return x * y * (3.*x2 - y2) * (x2 - 3.*y2) * 2.                         ; }
__device__ __forceinline__ double f_phi_n5(const double x, const double y) { const double x2 = x*x, y2 = y*y; 				    return y * (y2*y2 + 5.*x2 * (x2 - 2.*y2))	                            ; }
__device__ __forceinline__ double f_phi_n4(const double x, const double y) { 												    return x * y * (x + y) * (x - y) * 4.		                            ; }
__device__ __forceinline__ double f_phi_n3(const double x, const double y) { 												    return y * (3.*x*x - y*y)				                                ; }
__device__ __forceinline__ double f_phi_n2(const double x, const double y) { 												    return x * y * 2.							                            ; }
__device__ __forceinline__ double f_phi_n1(const double x, const double y) { 												    return y								                                ; }

__device__ __forceinline__ double f_phi_p1(const double x, const double y) { 													return x								                                ; }
__device__ __forceinline__ double f_phi_p2(const double x, const double y) { 													return (x + y) * (x - y)				                                ; }
__device__ __forceinline__ double f_phi_p3(const double x, const double y) { 													return x * (x*x - 3.*y*y)				                                ; }
__device__ __forceinline__ double f_phi_p4(const double x, const double y) { const double x2 = x*x, y2 = y*y; 					return x2 * (x2 - 6.*y2) + y2*y2			                            ; }
__device__ __forceinline__ double f_phi_p5(const double x, const double y) { const double x2 = x*x, y2 = y*y; 					return x * (x2*x2 + 5.*y2 * (y2 - 2.*x2))	                            ; }
__device__ __forceinline__ double f_phi_p6(const double x, const double y) { const double x2 = x*x, y2 = y*y, dx2y2 = x2 - y2;  return dx2y2 * (dx2y2*dx2y2 - 12.*x2*y2)                                ; }
__device__ __forceinline__ double f_phi_p7(const double x, const double y) { const double x2 = x*x, y2 = y*y; 					return x * (x2*x2 * (x2 - 21.*y2) + 7.*y2*y2 * (5.*x2 - y2))            ; }
__device__ __forceinline__ double f_phi_p8(const double x, const double y) { const double x2 = x*x, y2 = y*y, dx2y2 = x2 - y2;
                                                                             const double dx2y2_sq = dx2y2*dx2y2;               return dx2y2_sq*dx2y2_sq + x2*y2 * (16.*x2*y2 - 24.*dx2y2_sq)           ; }
__device__ __forceinline__ double f_phi_p9(const double x, const double y) { const double x2 = x*x, y2 = y*y;
                                                                             const double x4 = x2*x2, y4 = y2*y2;               return x * (x4*x4 + 126.*x4*y4 + 9.*y4*y4 - x2*y2 * (36.*x4 + 84*y4))   ; }
__device__ __forceinline__ double f_phi_pA(const double x, const double y) { const double x2 = x*x, y2 = y*y, dx2y2 = x2 - y2;
                                                                             const double dx2y2_sq = dx2y2*dx2y2;               return dx2y2 * (dx2y2_sq*dx2y2_sq + 40.*x2*y2 * (2.*x2*y2 - dx2y2_sq))  ; }

/*
    Polynoms in z.
*/
__device__ __forceinline__ double p_c  (const double z, const double c0)                                    {                           return c0                                                               ; }
__device__ __forceinline__ double p_z  (const double z, const double c1)                                    {                           return c1 * z                                                           ; }
__device__ __forceinline__ double p_z2 (const double z, const double c0, const double c2)                   {                           return c0 + c2 * z * z                                                  ; }
__device__ __forceinline__ double p_z2z(const double z, const double c0, const double c2)                   {                           return (c0 + c2 * z * z) * z                                            ; }
__device__ __forceinline__ double p_z4 (const double z, const double c0, const double c2, const double c4)  { const double z2 = z*z;    return c0 + (c2 + c4 * z2) * z2                                         ; }
__device__ __forceinline__ double p_z4z(const double z, const double c0, const double c2, const double c4)  { const double z2 = z*z;    return (c0 + (c2 + c4 * z2) * z2) * z                                   ; }
__device__ __forceinline__ double p_z6 (const double z, const double c0, const double c2, const double c4,
                                                        const double c6)                                    { const double z2 = z*z;    return c0 + (c2 + (c4 + c6 * z2) * z2) * z2                             ; }
__device__ __forceinline__ double p_z6z(const double z, const double c0, const double c2, const double c4,
                                                        const double c6)                                    { const double z2 = z*z;    return (c0 + (c2 + (c4 + c6 * z2) * z2) * z2) * z                       ; }
__device__ __forceinline__ double p_z8 (const double z, const double c0, const double c2, const double c4,
                                                        const double c6, const double c8)                   { const double z2 = z*z;    return c0 + (c2 + (c4 + (c6 + c8 * z2) * z2) * z2) * z2                 ; }
__device__ __forceinline__ double p_z8z(const double z, const double c0, const double c2, const double c4,
                                                        const double c6, const double c8)                   { const double z2 = z*z;    return (c0 + (c2 + (c4 + (c6 + c8 * z2) * z2) * z2) * z2) * z           ; }
__device__ __forceinline__ double p_zA (const double z, const double c0, const double c2, const double c4,
                                                        const double c6, const double c8, const double c10) { const double z2 = z*z;    return c0 + (c2 + (c4 + (c6 + (c8 + c10 * z2) * z2) * z2) * z2) * z2    ; }

/*
    Handle special case of xyz = (0., 0., 0.) with additional multiplier, either 0. or 1.
*/
__device__ __forceinline__ double special(const double x, const double y, const double z) { return (double) (x != 0. || y != 0. || z != 0.); }


/*
    Functions for specific (L, m). Product of common multiplier from m and polynomial in z.
    Cases (L, 0) with L even and > 0, have additional multiplier to treat special case of input (0, 0, 0) - return 0.
    Multiplier is either 1. if any of x, y, z is different from 0, or 0. otherwise.
    It is constructed as a multiplier in opposite to if-else statement in order to avoid branch divergence.
*/

__device__ double sh00 (const double x, const double y, const double z) { return RSH_C00()                                                                                                      ; }

__device__ double sh1n1(const double x, const double y, const double z) { return f_phi_n1(x, y) * p_c  (z, RSH_C11())                                                                           ; }
__device__ double sh10 (const double x, const double y, const double z) { return 				  p_z  (z, RSH_C10())                                                                           ; }
__device__ double sh1p1(const double x, const double y, const double z) { return f_phi_p1(x, y) * p_c  (z, RSH_C11())                                                                           ; }

__device__ double sh2n2(const double x, const double y, const double z) { return f_phi_n2(x, y) * p_c  (z, RSH_C22())                                                                           ; }
__device__ double sh2n1(const double x, const double y, const double z) { return f_phi_n1(x, y) * p_z  (z, RSH_C21())                                                                           ; }
__device__ double sh20 (const double x, const double y, const double z) { return special(x,y,z) * p_z2 (z, RSH_C20_c(), RSH_C20_z2())                                                           ; }
__device__ double sh2p1(const double x, const double y, const double z) { return f_phi_p1(x, y) * p_z  (z, RSH_C21())                                                                           ; }
__device__ double sh2p2(const double x, const double y, const double z) { return f_phi_p2(x, y) * p_c  (z, RSH_C22())                                                                           ; }

__device__ double sh3n3(const double x, const double y, const double z) { return f_phi_n3(x, y) * p_c  (z, RSH_C33())                                                                           ; }
__device__ double sh3n2(const double x, const double y, const double z) { return f_phi_n2(x, y) * p_z  (z, RSH_C32())                                                                           ; }
__device__ double sh3n1(const double x, const double y, const double z) { return f_phi_n1(x, y) * p_z2 (z, RSH_C31_c(), RSH_C31_z2())                                                           ; }
__device__ double sh30 (const double x, const double y, const double z) { return 				  p_z2z(z, RSH_C30_c(), RSH_C30_z2())                                                           ; }
__device__ double sh3p1(const double x, const double y, const double z) { return f_phi_p1(x, y) * p_z2 (z, RSH_C31_c(), RSH_C31_z2())                                                           ; }
__device__ double sh3p2(const double x, const double y, const double z) { return f_phi_p2(x, y) * p_z  (z, RSH_C32())                                                                           ; }
__device__ double sh3p3(const double x, const double y, const double z) { return f_phi_p3(x, y) * p_c  (z, RSH_C33())                                                                           ; }

__device__ double sh4n4(const double x, const double y, const double z) { return f_phi_n4(x, y) * p_c  (z, RSH_C44())                                                                           ; }
__device__ double sh4n3(const double x, const double y, const double z) { return f_phi_n3(x, y) * p_z  (z, RSH_C43())                                                                           ; }
__device__ double sh4n2(const double x, const double y, const double z) { return f_phi_n2(x, y) * p_z2 (z, RSH_C42_c(), RSH_C42_z2())                                                           ; }
__device__ double sh4n1(const double x, const double y, const double z) { return f_phi_n1(x, y) * p_z2z(z, RSH_C41_c(), RSH_C41_z2())                                                           ; }
__device__ double sh40 (const double x, const double y, const double z) { return special(x,y,z) * p_z4 (z, RSH_C40_c(), RSH_C40_z2(), RSH_C40_z4())                                             ; }
__device__ double sh4p1(const double x, const double y, const double z) { return f_phi_p1(x, y) * p_z2z(z, RSH_C41_c(), RSH_C41_z2())                                                           ; }
__device__ double sh4p2(const double x, const double y, const double z) { return f_phi_p2(x, y) * p_z2 (z, RSH_C42_c(), RSH_C42_z2())                                                           ; }
__device__ double sh4p3(const double x, const double y, const double z) { return f_phi_p3(x, y) * p_z  (z, RSH_C43())                                                                           ; }
__device__ double sh4p4(const double x, const double y, const double z) { return f_phi_p4(x, y) * p_c  (z, RSH_C44())                                                                           ; }

__device__ double sh5n5(const double x, const double y, const double z) { return f_phi_n5(x, y) * p_c  (z, RSH_C55())                                                                           ; }
__device__ double sh5n4(const double x, const double y, const double z) { return f_phi_n4(x, y) * p_z  (z, RSH_C54())                                                                           ; }
__device__ double sh5n3(const double x, const double y, const double z) { return f_phi_n3(x, y) * p_z2 (z, RSH_C53_c(), RSH_C53_z2())                                                           ; }
__device__ double sh5n2(const double x, const double y, const double z) { return f_phi_n2(x, y) * p_z2z(z, RSH_C52_c(), RSH_C52_z2())                                                           ; }
__device__ double sh5n1(const double x, const double y, const double z) { return f_phi_n1(x, y) * p_z4 (z, RSH_C51_c(), RSH_C51_z2(), RSH_C51_z4())                                             ; }
__device__ double sh50 (const double x, const double y, const double z) { return 				  p_z4z(z, RSH_C50_c(), RSH_C50_z2(), RSH_C50_z4())                                             ; }
__device__ double sh5p1(const double x, const double y, const double z) { return f_phi_p1(x, y) * p_z4 (z, RSH_C51_c(), RSH_C51_z2(), RSH_C51_z4())                                             ; }
__device__ double sh5p2(const double x, const double y, const double z) { return f_phi_p2(x, y) * p_z2z(z, RSH_C52_c(), RSH_C52_z2())                                                           ; }
__device__ double sh5p3(const double x, const double y, const double z) { return f_phi_p3(x, y) * p_z2 (z, RSH_C53_c(), RSH_C53_z2())                                                           ; }
__device__ double sh5p4(const double x, const double y, const double z) { return f_phi_p4(x, y) * p_z  (z, RSH_C54())                                                                           ; }
__device__ double sh5p5(const double x, const double y, const double z) { return f_phi_p5(x, y) * p_c  (z, RSH_C55())                                                                           ; }

__device__ double sh6n6(const double x, const double y, const double z) { return f_phi_n6(x, y) * p_c  (z, RSH_C66())                                                                           ; }
__device__ double sh6n5(const double x, const double y, const double z) { return f_phi_n5(x, y) * p_z  (z, RSH_C65())                                                                           ; }
__device__ double sh6n4(const double x, const double y, const double z) { return f_phi_n4(x, y) * p_z2 (z, RSH_C64_c(), RSH_C64_z2())                                                           ; }
__device__ double sh6n3(const double x, const double y, const double z) { return f_phi_n3(x, y) * p_z2z(z, RSH_C63_c(), RSH_C63_z2())                                                           ; }
__device__ double sh6n2(const double x, const double y, const double z) { return f_phi_n2(x, y) * p_z4 (z, RSH_C62_c(), RSH_C62_z2(), RSH_C62_z4())                                             ; }
__device__ double sh6n1(const double x, const double y, const double z) { return f_phi_n1(x, y) * p_z4z(z, RSH_C61_c(), RSH_C61_z2(), RSH_C61_z4())                                             ; }
__device__ double sh60 (const double x, const double y, const double z) { return special(x,y,z) * p_z6 (z, RSH_C60_c(), RSH_C60_z2(), RSH_C60_z4(), RSH_C60_z6())                               ; }
__device__ double sh6p1(const double x, const double y, const double z) { return f_phi_p1(x, y) * p_z4z(z, RSH_C61_c(), RSH_C61_z2(), RSH_C61_z4())                                             ; }
__device__ double sh6p2(const double x, const double y, const double z) { return f_phi_p2(x, y) * p_z4 (z, RSH_C62_c(), RSH_C62_z2(), RSH_C62_z4())                                             ; }
__device__ double sh6p3(const double x, const double y, const double z) { return f_phi_p3(x, y) * p_z2z(z, RSH_C63_c(), RSH_C63_z2())                                                           ; }
__device__ double sh6p4(const double x, const double y, const double z) { return f_phi_p4(x, y) * p_z2 (z, RSH_C64_c(), RSH_C64_z2())                                                           ; }
__device__ double sh6p5(const double x, const double y, const double z) { return f_phi_p5(x, y) * p_z  (z, RSH_C65())                                                                           ; }
__device__ double sh6p6(const double x, const double y, const double z) { return f_phi_p6(x, y) * p_c  (z, RSH_C66())                                                                           ; }

__device__ double sh7n7(const double x, const double y, const double z) { return f_phi_n7(x, y) * p_c  (z, RSH_C77())                                                                           ; }
__device__ double sh7n6(const double x, const double y, const double z) { return f_phi_n6(x, y) * p_z  (z, RSH_C76())                                                                           ; }
__device__ double sh7n5(const double x, const double y, const double z) { return f_phi_n5(x, y) * p_z2 (z, RSH_C75_c(), RSH_C75_z2())                                                           ; }
__device__ double sh7n4(const double x, const double y, const double z) { return f_phi_n4(x, y) * p_z2z(z, RSH_C74_c(), RSH_C74_z2())                                                           ; }
__device__ double sh7n3(const double x, const double y, const double z) { return f_phi_n3(x, y) * p_z4 (z, RSH_C73_c(), RSH_C73_z2(), RSH_C73_z4())                                             ; }
__device__ double sh7n2(const double x, const double y, const double z) { return f_phi_n2(x, y) * p_z4z(z, RSH_C72_c(), RSH_C72_z2(), RSH_C72_z4())                                             ; }
__device__ double sh7n1(const double x, const double y, const double z) { return f_phi_n1(x, y) * p_z6 (z, RSH_C71_c(), RSH_C71_z2(), RSH_C71_z4(), RSH_C71_z6())                               ; }
__device__ double sh70 (const double x, const double y, const double z) { return                  p_z6z(z, RSH_C70_c(), RSH_C70_z2(), RSH_C70_z4(), RSH_C70_z6())                               ; }
__device__ double sh7p1(const double x, const double y, const double z) { return f_phi_p1(x, y) * p_z6 (z, RSH_C71_c(), RSH_C71_z2(), RSH_C71_z4(), RSH_C71_z6())                               ; }
__device__ double sh7p2(const double x, const double y, const double z) { return f_phi_p2(x, y) * p_z4z(z, RSH_C72_c(), RSH_C72_z2(), RSH_C72_z4())                                             ; }
__device__ double sh7p3(const double x, const double y, const double z) { return f_phi_p3(x, y) * p_z4 (z, RSH_C73_c(), RSH_C73_z2(), RSH_C73_z4())                                             ; }
__device__ double sh7p4(const double x, const double y, const double z) { return f_phi_p4(x, y) * p_z2z(z, RSH_C74_c(), RSH_C74_z2())                                                           ; }
__device__ double sh7p5(const double x, const double y, const double z) { return f_phi_p5(x, y) * p_z2 (z, RSH_C75_c(), RSH_C75_z2())                                                           ; }
__device__ double sh7p6(const double x, const double y, const double z) { return f_phi_p6(x, y) * p_z  (z, RSH_C76())                                                                           ; }
__device__ double sh7p7(const double x, const double y, const double z) { return f_phi_p7(x, y) * p_c  (z, RSH_C77())                                                                           ; }

__device__ double sh8n8(const double x, const double y, const double z) { return f_phi_n8(x, y) * p_c  (z, RSH_C88())                                                                           ; }
__device__ double sh8n7(const double x, const double y, const double z) { return f_phi_n7(x, y) * p_z  (z, RSH_C87())                                                                           ; }
__device__ double sh8n6(const double x, const double y, const double z) { return f_phi_n6(x, y) * p_z2 (z, RSH_C86_c(), RSH_C86_z2())                                                           ; }
__device__ double sh8n5(const double x, const double y, const double z) { return f_phi_n5(x, y) * p_z2z(z, RSH_C85_c(), RSH_C85_z2())                                                           ; }
__device__ double sh8n4(const double x, const double y, const double z) { return f_phi_n4(x, y) * p_z4 (z, RSH_C84_c(), RSH_C84_z2(), RSH_C84_z4())                                             ; }
__device__ double sh8n3(const double x, const double y, const double z) { return f_phi_n3(x, y) * p_z4z(z, RSH_C83_c(), RSH_C83_z2(), RSH_C83_z4())                                             ; }
__device__ double sh8n2(const double x, const double y, const double z) { return f_phi_n2(x, y) * p_z6 (z, RSH_C82_c(), RSH_C82_z2(), RSH_C82_z4(), RSH_C82_z6())                               ; }
__device__ double sh8n1(const double x, const double y, const double z) { return f_phi_n1(x, y) * p_z6z(z, RSH_C81_c(), RSH_C81_z2(), RSH_C81_z4(), RSH_C81_z6())                               ; }
__device__ double sh80 (const double x, const double y, const double z) { return special(x,y,z) * p_z8 (z, RSH_C80_c(), RSH_C80_z2(), RSH_C80_z4(), RSH_C80_z6(), RSH_C80_z8())                 ; }
__device__ double sh8p1(const double x, const double y, const double z) { return f_phi_p1(x, y) * p_z6z(z, RSH_C81_c(), RSH_C81_z2(), RSH_C81_z4(), RSH_C81_z6())                               ; }
__device__ double sh8p2(const double x, const double y, const double z) { return f_phi_p2(x, y) * p_z6 (z, RSH_C82_c(), RSH_C82_z2(), RSH_C82_z4(), RSH_C82_z6())                               ; }
__device__ double sh8p3(const double x, const double y, const double z) { return f_phi_p3(x, y) * p_z4z(z, RSH_C83_c(), RSH_C83_z2(), RSH_C83_z4())                                             ; }
__device__ double sh8p4(const double x, const double y, const double z) { return f_phi_p4(x, y) * p_z4 (z, RSH_C84_c(), RSH_C84_z2(), RSH_C84_z4())                                             ; }
__device__ double sh8p5(const double x, const double y, const double z) { return f_phi_p5(x, y) * p_z2z(z, RSH_C85_c(), RSH_C85_z2())                                                           ; }
__device__ double sh8p6(const double x, const double y, const double z) { return f_phi_p6(x, y) * p_z2 (z, RSH_C86_c(), RSH_C86_z2())                                                           ; }
__device__ double sh8p7(const double x, const double y, const double z) { return f_phi_p7(x, y) * p_z  (z, RSH_C87())                                                                           ; }
__device__ double sh8p8(const double x, const double y, const double z) { return f_phi_p8(x, y) * p_c  (z, RSH_C88())                                                                           ; }

__device__ double sh9n9(const double x, const double y, const double z) { return f_phi_n9(x, y) * p_c  (z, RSH_C99())                                                                           ; }
__device__ double sh9n8(const double x, const double y, const double z) { return f_phi_n8(x, y) * p_z  (z, RSH_C98())                                                                           ; }
__device__ double sh9n7(const double x, const double y, const double z) { return f_phi_n7(x, y) * p_z2 (z, RSH_C97_c(), RSH_C97_z2())                                                           ; }
__device__ double sh9n6(const double x, const double y, const double z) { return f_phi_n6(x, y) * p_z2z(z, RSH_C96_c(), RSH_C96_z2())                                                           ; }
__device__ double sh9n5(const double x, const double y, const double z) { return f_phi_n5(x, y) * p_z4 (z, RSH_C95_c(), RSH_C95_z2(), RSH_C95_z4())                                             ; }
__device__ double sh9n4(const double x, const double y, const double z) { return f_phi_n4(x, y) * p_z4z(z, RSH_C94_c(), RSH_C94_z2(), RSH_C94_z4())                                             ; }
__device__ double sh9n3(const double x, const double y, const double z) { return f_phi_n3(x, y) * p_z6 (z, RSH_C93_c(), RSH_C93_z2(), RSH_C93_z4(), RSH_C93_z6())                               ; }
__device__ double sh9n2(const double x, const double y, const double z) { return f_phi_n2(x, y) * p_z6z(z, RSH_C92_c(), RSH_C92_z2(), RSH_C92_z4(), RSH_C92_z6())                               ; }
__device__ double sh9n1(const double x, const double y, const double z) { return f_phi_n1(x, y) * p_z8 (z, RSH_C91_c(), RSH_C91_z2(), RSH_C91_z4(), RSH_C91_z6(), RSH_C91_z8())                 ; }
__device__ double sh90 (const double x, const double y, const double z) { return                  p_z8z(z, RSH_C90_c(), RSH_C90_z2(), RSH_C90_z4(), RSH_C90_z6(), RSH_C90_z8())                 ; }
__device__ double sh9p1(const double x, const double y, const double z) { return f_phi_p1(x, y) * p_z8 (z, RSH_C91_c(), RSH_C91_z2(), RSH_C91_z4(), RSH_C91_z6(), RSH_C91_z8())                 ; }
__device__ double sh9p2(const double x, const double y, const double z) { return f_phi_p2(x, y) * p_z6z(z, RSH_C92_c(), RSH_C92_z2(), RSH_C92_z4(), RSH_C92_z6())                               ; }
__device__ double sh9p3(const double x, const double y, const double z) { return f_phi_p3(x, y) * p_z6 (z, RSH_C93_c(), RSH_C93_z2(), RSH_C93_z4(), RSH_C93_z6())                               ; }
__device__ double sh9p4(const double x, const double y, const double z) { return f_phi_p4(x, y) * p_z4z(z, RSH_C94_c(), RSH_C94_z2(), RSH_C94_z4())                                             ; }
__device__ double sh9p5(const double x, const double y, const double z) { return f_phi_p5(x, y) * p_z4 (z, RSH_C95_c(), RSH_C95_z2(), RSH_C95_z4())                                             ; }
__device__ double sh9p6(const double x, const double y, const double z) { return f_phi_p6(x, y) * p_z2z(z, RSH_C96_c(), RSH_C96_z2())                                                           ; }
__device__ double sh9p7(const double x, const double y, const double z) { return f_phi_p7(x, y) * p_z2 (z, RSH_C97_c(), RSH_C97_z2())                                                           ; }
__device__ double sh9p8(const double x, const double y, const double z) { return f_phi_p8(x, y) * p_z  (z, RSH_C98())                                                                           ; }
__device__ double sh9p9(const double x, const double y, const double z) { return f_phi_p9(x, y) * p_c  (z, RSH_C99())                                                                           ; }

__device__ double shAnA(const double x, const double y, const double z) { return f_phi_nA(x, y) * p_c  (z, RSH_CAA())                                                                           ; }
__device__ double shAn9(const double x, const double y, const double z) { return f_phi_n9(x, y) * p_z  (z, RSH_CA9())                                                                           ; }
__device__ double shAn8(const double x, const double y, const double z) { return f_phi_n8(x, y) * p_z2 (z, RSH_CA8_c(), RSH_CA8_z2())                                                           ; }
__device__ double shAn7(const double x, const double y, const double z) { return f_phi_n7(x, y) * p_z2z(z, RSH_CA7_c(), RSH_CA7_z2())                                                           ; }
__device__ double shAn6(const double x, const double y, const double z) { return f_phi_n6(x, y) * p_z4 (z, RSH_CA6_c(), RSH_CA6_z2(), RSH_CA6_z4())                                             ; }
__device__ double shAn5(const double x, const double y, const double z) { return f_phi_n5(x, y) * p_z4z(z, RSH_CA5_c(), RSH_CA5_z2(), RSH_CA5_z4())                                             ; }
__device__ double shAn4(const double x, const double y, const double z) { return f_phi_n4(x, y) * p_z6 (z, RSH_CA4_c(), RSH_CA4_z2(), RSH_CA4_z4(), RSH_CA4_z6())                               ; }
__device__ double shAn3(const double x, const double y, const double z) { return f_phi_n3(x, y) * p_z6z(z, RSH_CA3_c(), RSH_CA3_z2(), RSH_CA3_z4(), RSH_CA3_z6())                               ; }
__device__ double shAn2(const double x, const double y, const double z) { return f_phi_n2(x, y) * p_z8 (z, RSH_CA2_c(), RSH_CA2_z2(), RSH_CA2_z4(), RSH_CA2_z6(), RSH_CA2_z8())                 ; }
__device__ double shAn1(const double x, const double y, const double z) { return f_phi_n1(x, y) * p_z8z(z, RSH_CA1_c(), RSH_CA1_z2(), RSH_CA1_z4(), RSH_CA1_z6(), RSH_CA1_z8())                 ; }
__device__ double shA0 (const double x, const double y, const double z) { return special(x,y,z) * p_zA (z, RSH_CA0_c(), RSH_CA0_z2(), RSH_CA0_z4(), RSH_CA0_z6(), RSH_CA0_z8(), RSH_CA0_zA())   ; }
__device__ double shAp1(const double x, const double y, const double z) { return f_phi_p1(x, y) * p_z8z(z, RSH_CA1_c(), RSH_CA1_z2(), RSH_CA1_z4(), RSH_CA1_z6(), RSH_CA1_z8())                 ; }
__device__ double shAp2(const double x, const double y, const double z) { return f_phi_p2(x, y) * p_z8 (z, RSH_CA2_c(), RSH_CA2_z2(), RSH_CA2_z4(), RSH_CA2_z6(), RSH_CA2_z8())                 ; }
__device__ double shAp3(const double x, const double y, const double z) { return f_phi_p3(x, y) * p_z6z(z, RSH_CA3_c(), RSH_CA3_z2(), RSH_CA3_z4(), RSH_CA3_z6())                               ; }
__device__ double shAp4(const double x, const double y, const double z) { return f_phi_p4(x, y) * p_z6 (z, RSH_CA4_c(), RSH_CA4_z2(), RSH_CA4_z4(), RSH_CA4_z6())                               ; }
__device__ double shAp5(const double x, const double y, const double z) { return f_phi_p5(x, y) * p_z4z(z, RSH_CA5_c(), RSH_CA5_z2(), RSH_CA5_z4())                                             ; }
__device__ double shAp6(const double x, const double y, const double z) { return f_phi_p6(x, y) * p_z4 (z, RSH_CA6_c(), RSH_CA6_z2(), RSH_CA6_z4())                                             ; }
__device__ double shAp7(const double x, const double y, const double z) { return f_phi_p7(x, y) * p_z2z(z, RSH_CA7_c(), RSH_CA7_z2())                                                           ; }
__device__ double shAp8(const double x, const double y, const double z) { return f_phi_p8(x, y) * p_z2 (z, RSH_CA8_c(), RSH_CA8_z2())                                                           ; }
__device__ double shAp9(const double x, const double y, const double z) { return f_phi_p9(x, y) * p_z  (z, RSH_CA9())                                                                           ; }
__device__ double shApA(const double x, const double y, const double z) { return f_phi_pA(x, y) * p_c  (z, RSH_CAA())                                                                           ; }

// array of pointers to functions stored to "constant memory" (__constant__) in GPU - common for all blocks.
__constant__ double (*const fptr[]) (const double, const double, const double) = {
                    		                                          sh00, 											                                    //                                                     0
                    		                                   sh1n1, sh10,  sh1p1, 									                                    //                                                1,   2,   3
                     		                            sh2n2, sh2n1, sh20,  sh2p1, sh2p2,								                                    //                                           4,   5,   6,   7,   8
                    		                     sh3n3, sh3n2, sh3n1, sh30,  sh3p1, sh3p2, sh3p3,					                                        //                                      9,  10,  11,  12,  13,  14,  15
                    		              sh4n4, sh4n3, sh4n2, sh4n1, sh40,  sh4p1, sh4p2, sh4p3, sh4p4,				                                    //                                16,  17,  18,  19,  20,  21,  22,  23,  24
                    		       sh5n5, sh5n4, sh5n3, sh5n2, sh5n1, sh50,  sh5p1, sh5p2, sh5p3, sh5p4, sh5p5,		                                        //                           25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35
                    		sh6n6, sh6n5, sh6n4, sh6n3, sh6n2, sh6n1, sh60,  sh6p1, sh6p2, sh6p3, sh6p4, sh6p5, sh6p6,                                      //                      36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48
                     sh7n7, sh7n6, sh7n5, sh7n4, sh7n3, sh7n2, sh7n1, sh70,  sh7p1, sh7p2, sh7p3, sh7p4, sh7p5, sh7p6, sh7p7, 	                            //                 49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63
              sh8n8, sh8n7, sh8n6, sh8n5, sh8n4, sh8n3, sh8n2, sh8n1, sh80,  sh8p1, sh8p2, sh8p3, sh8p4, sh8p5, sh8p6, sh8p7, sh8p8,                        //            64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80
       sh9n9, sh9n8, sh9n7, sh9n6, sh9n5, sh9n4, sh9n3, sh9n2, sh9n1, sh90,  sh9p1, sh9p2, sh9p3, sh9p4, sh9p5, sh9p6, sh9p7, sh9p8, sh9p9,                 //       81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95.  96,  97,  98,  99
shAnA, shAn9, shAn8, shAn7, shAn6, shAn5, shAn4, shAn3, shAn2, shAn1, shA0,  shAp1, shAp2, shAp3, shAp4, shAp5, shAp6, shAp7, shAp8, shAp9, shApA           // 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120
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
