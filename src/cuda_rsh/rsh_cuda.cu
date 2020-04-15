/*
    Calculates real spherical harmonics up to L=6 (inclusive) from Cartesian coordinates.
    Coordinates x, y, z are expected to form unit length vector.
*/

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

// 1./sqrt(pi) - on higher precision than actual execution of operation over double can provide
template<typename T> __device__ constexpr T RSQRT_PI() 		{ return 		0.564189583547756286948079451560772585844050629328998856844;}

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
template<typename T> __device__ constexpr T RSH_C00() 		{ return 		RSQRT_PI<T>()/2.			  		    ;}

// L = 1
template<typename T> __device__ constexpr T RSH_C10() 		{ return 		RSQRT_PI<T>()*sqrt(3.)/2.    		    ;}
template<typename T> __device__ constexpr T RSH_C11() 		{ return 		RSQRT_PI<T>()*sqrt(3.)/2.    		    ;}

// L = 2
template<typename T> __device__ constexpr T RSH_C20_c() 	{ return 	   -RSQRT_PI<T>()*sqrt(5.)/4.    		    ;}
template<typename T> __device__ constexpr T RSH_C20_z2() 	{ return 		RSQRT_PI<T>()*sqrt(5.)*3./4.            ;}
template<typename T> __device__ constexpr T RSH_C21() 		{ return 		RSQRT_PI<T>()*sqrt(15.)/2.   		    ;}
template<typename T> __device__ constexpr T RSH_C22() 		{ return 		RSQRT_PI<T>()*sqrt(15.)/4.   		    ;}

// L = 3
template<typename T> __device__ constexpr T RSH_C30_c() 	{ return 	   -RSQRT_PI<T>()*sqrt(7.)*3./4.  		    ;}
template<typename T> __device__ constexpr T RSH_C30_z2() 	{ return 		RSQRT_PI<T>()*sqrt(7.)*5./4.  		    ;}
template<typename T> __device__ constexpr T RSH_C31_c() 	{ return 	   -RSQRT_PI<T>()*sqrt(42.)/8.   		    ;}
template<typename T> __device__ constexpr T RSH_C31_z2() 	{ return 		RSQRT_PI<T>()*sqrt(42.)*5./8. 		    ;}
template<typename T> __device__ constexpr T RSH_C32() 		{ return 		RSQRT_PI<T>()*sqrt(105.)/4.  		    ;}
template<typename T> __device__ constexpr T RSH_C33() 		{ return 		RSQRT_PI<T>()*sqrt(70.)/8.   		    ;}

// L = 4
template<typename T> __device__ constexpr T RSH_C40_c()		{ return 		RSQRT_PI<T>()*9./16.          		    ;}
template<typename T> __device__ constexpr T RSH_C40_z2() 	{ return 	   -RSQRT_PI<T>()*90./16.         		    ;}
template<typename T> __device__ constexpr T RSH_C40_z4() 	{ return 		RSQRT_PI<T>()*105./16.        		    ;}
template<typename T> __device__ constexpr T RSH_C41_c() 	{ return 	   -RSQRT_PI<T>()*sqrt(10.)*9./8.		    ;}
template<typename T> __device__ constexpr T RSH_C41_z2() 	{ return 		RSQRT_PI<T>()*sqrt(10.)*21./8.		    ;}
template<typename T> __device__ constexpr T RSH_C42_c() 	{ return 	   -RSQRT_PI<T>()*sqrt(5.)*3./8.  		    ;}
template<typename T> __device__ constexpr T RSH_C42_z2() 	{ return 		RSQRT_PI<T>()*sqrt(5.)*21./8. 		    ;}
template<typename T> __device__ constexpr T RSH_C43() 		{ return 		RSQRT_PI<T>()*sqrt(70.)*3./8. 		    ;}
template<typename T> __device__ constexpr T RSH_C44() 		{ return 		RSQRT_PI<T>()*sqrt(35.)*3./16.		    ;}

// L = 5
template<typename T> __device__ constexpr T RSH_C50_c() 	{ return 		RSQRT_PI<T>()*sqrt(11.)*15./16. 	    ;}
template<typename T> __device__ constexpr T RSH_C50_z2() 	{ return 	   -RSQRT_PI<T>()*sqrt(11.)*70./16. 	    ;}
template<typename T> __device__ constexpr T RSH_C50_z4() 	{ return 		RSQRT_PI<T>()*sqrt(11.)*63./16. 	    ;}
template<typename T> __device__ constexpr T RSH_C51_c() 	{ return 		RSQRT_PI<T>()*sqrt(165.)/16.   	        ;}
template<typename T> __device__ constexpr T RSH_C51_z2() 	{ return 	   -RSQRT_PI<T>()*sqrt(165.)*14./16.	    ;}
template<typename T> __device__ constexpr T RSH_C51_z4() 	{ return 		RSQRT_PI<T>()*sqrt(165.)*21./16.	    ;}
template<typename T> __device__ constexpr T RSH_C52_c() 	{ return 	   -RSQRT_PI<T>()*sqrt(1155.)/8.   	        ;}
template<typename T> __device__ constexpr T RSH_C52_z2() 	{ return 		RSQRT_PI<T>()*sqrt(1155.)*3./8. 	    ;}
template<typename T> __device__ constexpr T RSH_C53_c() 	{ return 	   -RSQRT_PI<T>()*sqrt(770.)/32.   	        ;}
template<typename T> __device__ constexpr T RSH_C53_z2() 	{ return 		RSQRT_PI<T>()*sqrt(770.)*9./32. 	    ;}
template<typename T> __device__ constexpr T RSH_C54() 		{ return 		RSQRT_PI<T>()*sqrt(385.)*3./16.  	    ;}
template<typename T> __device__ constexpr T RSH_C55() 		{ return 		RSQRT_PI<T>()*sqrt(154.)*3./32. 	    ;}

// L = 6
template<typename T> __device__ constexpr T RSH_C60_c() 	{ return 	   -RSQRT_PI<T>()*sqrt(13.)*5./32. 	        ;}
template<typename T> __device__ constexpr T RSH_C60_z2() 	{ return 		RSQRT_PI<T>()*sqrt(13.)*105./32. 	    ;}
template<typename T> __device__ constexpr T RSH_C60_z4() 	{ return 	   -RSQRT_PI<T>()*sqrt(13.)*315./32. 	    ;}
template<typename T> __device__ constexpr T RSH_C60_z6() 	{ return 		RSQRT_PI<T>()*sqrt(13.)*231./32. 	    ;}
template<typename T> __device__ constexpr T RSH_C61_c() 	{ return 		RSQRT_PI<T>()*sqrt(273.)*5./16.  	    ;}
template<typename T> __device__ constexpr T RSH_C61_z2() 	{ return 	   -RSQRT_PI<T>()*sqrt(273.)*30./16.	    ;}
template<typename T> __device__ constexpr T RSH_C61_z4() 	{ return 		RSQRT_PI<T>()*sqrt(273.)*33./16.	    ;}
template<typename T> __device__ constexpr T RSH_C62_c() 	{ return 		RSQRT_PI<T>()*sqrt(2730.)/64.   	    ;}
template<typename T> __device__ constexpr T RSH_C62_z2() 	{ return 	   -RSQRT_PI<T>()*sqrt(2730.)*18./64. 	    ;}
template<typename T> __device__ constexpr T RSH_C62_z4() 	{ return 		RSQRT_PI<T>()*sqrt(2730.)*33./64. 	    ;}
template<typename T> __device__ constexpr T RSH_C63_c() 	{ return 	   -RSQRT_PI<T>()*sqrt(2730.)*3./32.        ;}
template<typename T> __device__ constexpr T RSH_C63_z2() 	{ return 		RSQRT_PI<T>()*sqrt(2730.)*11./32. 	    ;}
template<typename T> __device__ constexpr T RSH_C64_c() 	{ return 	   -RSQRT_PI<T>()*sqrt(91.)*3./32.  	    ;}
template<typename T> __device__ constexpr T RSH_C64_z2() 	{ return 		RSQRT_PI<T>()*sqrt(91.)*33./32.  	    ;}
template<typename T> __device__ constexpr T RSH_C65() 		{ return 		RSQRT_PI<T>()*sqrt(2002.)*3./32. 	    ;}
template<typename T> __device__ constexpr T RSH_C66() 		{ return 		RSQRT_PI<T>()*sqrt(6006.)/64.  	        ;}

// L = 7
template<typename T> __device__ constexpr T RSH_C70_c() 	{ return 	   -RSQRT_PI<T>()*sqrt(15.)*35./32. 	    ;}
template<typename T> __device__ constexpr T RSH_C70_z2() 	{ return 	    RSQRT_PI<T>()*sqrt(15.)*315./32. 	    ;}
template<typename T> __device__ constexpr T RSH_C70_z4() 	{ return 	   -RSQRT_PI<T>()*sqrt(15.)*693./32. 	    ;}
template<typename T> __device__ constexpr T RSH_C70_z6() 	{ return 	    RSQRT_PI<T>()*sqrt(15.)*429./32. 	    ;}
template<typename T> __device__ constexpr T RSH_C71_c() 	{ return 	   -RSQRT_PI<T>()*sqrt(105.)*5./64. 	    ;}
template<typename T> __device__ constexpr T RSH_C71_z2() 	{ return 	    RSQRT_PI<T>()*sqrt(105.)*135./64. 	    ;}
template<typename T> __device__ constexpr T RSH_C71_z4() 	{ return 	   -RSQRT_PI<T>()*sqrt(105.)*495./64. 	    ;}
template<typename T> __device__ constexpr T RSH_C71_z6() 	{ return 	    RSQRT_PI<T>()*sqrt(105.)*429./64. 	    ;}
template<typename T> __device__ constexpr T RSH_C72_c() 	{ return 	    RSQRT_PI<T>()*sqrt(70.)*45./64. 	    ;}
template<typename T> __device__ constexpr T RSH_C72_z2() 	{ return 	   -RSQRT_PI<T>()*sqrt(70.)*330./64. 	    ;}
template<typename T> __device__ constexpr T RSH_C72_z4() 	{ return 	    RSQRT_PI<T>()*sqrt(70.)*429./64. 	    ;}
template<typename T> __device__ constexpr T RSH_C73_c() 	{ return 	    RSQRT_PI<T>()*sqrt(35.)*9./64. 	        ;}
template<typename T> __device__ constexpr T RSH_C73_z2() 	{ return 	   -RSQRT_PI<T>()*sqrt(35.)*198./64. 	    ;}
template<typename T> __device__ constexpr T RSH_C73_z4() 	{ return 	    RSQRT_PI<T>()*sqrt(35.)*429./64. 	    ;}
template<typename T> __device__ constexpr T RSH_C74_c() 	{ return 	   -RSQRT_PI<T>()*sqrt(385.)*9./32. 	    ;}
template<typename T> __device__ constexpr T RSH_C74_z2() 	{ return 	    RSQRT_PI<T>()*sqrt(385.)*39./32. 	    ;}
template<typename T> __device__ constexpr T RSH_C75_c() 	{ return 	   -RSQRT_PI<T>()*sqrt(385.)*3./64. 	    ;}
template<typename T> __device__ constexpr T RSH_C75_z2() 	{ return 	    RSQRT_PI<T>()*sqrt(385.)*39./64. 	    ;}
template<typename T> __device__ constexpr T RSH_C76() 		{ return 		RSQRT_PI<T>()*sqrt(10010.)*3./64. 	    ;}
template<typename T> __device__ constexpr T RSH_C77() 		{ return 		RSQRT_PI<T>()*sqrt(715.)*3./64.  	    ;}

// L = 8
template<typename T> __device__ constexpr T RSH_C80_c() 	{ return 	    RSQRT_PI<T>()*sqrt(17.)*35./256. 	    ;}
template<typename T> __device__ constexpr T RSH_C80_z2() 	{ return 	   -RSQRT_PI<T>()*sqrt(17.)*1260./256.      ;}
template<typename T> __device__ constexpr T RSH_C80_z4() 	{ return 	    RSQRT_PI<T>()*sqrt(17.)*6930./256.      ;}
template<typename T> __device__ constexpr T RSH_C80_z6() 	{ return 	   -RSQRT_PI<T>()*sqrt(17.)*12012./256.     ;}
template<typename T> __device__ constexpr T RSH_C80_z8() 	{ return 	    RSQRT_PI<T>()*sqrt(17.)*6435./256.      ;}
template<typename T> __device__ constexpr T RSH_C81_c() 	{ return 	   -RSQRT_PI<T>()*sqrt(17.)*105./64. 	    ;}
template<typename T> __device__ constexpr T RSH_C81_z2() 	{ return 	    RSQRT_PI<T>()*sqrt(17.)*1155./64. 	    ;}
template<typename T> __device__ constexpr T RSH_C81_z4() 	{ return 	   -RSQRT_PI<T>()*sqrt(17.)*3003./64. 	    ;}
template<typename T> __device__ constexpr T RSH_C81_z6() 	{ return 	    RSQRT_PI<T>()*sqrt(17.)*2145./64. 	    ;}
template<typename T> __device__ constexpr T RSH_C82_c() 	{ return 	   -RSQRT_PI<T>()*sqrt(1190.)*3./128. 	    ;}
template<typename T> __device__ constexpr T RSH_C82_z2() 	{ return 	    RSQRT_PI<T>()*sqrt(1190.)*99./128. 	    ;}
template<typename T> __device__ constexpr T RSH_C82_z4() 	{ return 	   -RSQRT_PI<T>()*sqrt(1190.)*429./128. 	;}
template<typename T> __device__ constexpr T RSH_C82_z6() 	{ return 	    RSQRT_PI<T>()*sqrt(1190.)*429./128. 	;}
template<typename T> __device__ constexpr T RSH_C83_c() 	{ return 	    RSQRT_PI<T>()*sqrt(19635.)*3./64. 	    ;}
template<typename T> __device__ constexpr T RSH_C83_z2() 	{ return 	   -RSQRT_PI<T>()*sqrt(19635.)*26./64. 	    ;}
template<typename T> __device__ constexpr T RSH_C83_z4() 	{ return 	    RSQRT_PI<T>()*sqrt(19635.)*39./64. 	    ;}
template<typename T> __device__ constexpr T RSH_C84_c() 	{ return 	    RSQRT_PI<T>()*sqrt(1309.)*3./128. 	    ;}
template<typename T> __device__ constexpr T RSH_C84_z2() 	{ return 	   -RSQRT_PI<T>()*sqrt(1309.)*78./128. 	    ;}
template<typename T> __device__ constexpr T RSH_C84_z4() 	{ return 	    RSQRT_PI<T>()*sqrt(1309.)*195./128. 	;}
template<typename T> __device__ constexpr T RSH_C85_c() 	{ return 	   -RSQRT_PI<T>()*sqrt(17017.)*3./64. 	    ;}
template<typename T> __device__ constexpr T RSH_C85_z2() 	{ return 	    RSQRT_PI<T>()*sqrt(17017.)*15./64. 	    ;}
template<typename T> __device__ constexpr T RSH_C86_c() 	{ return 	   -RSQRT_PI<T>()*sqrt(14586.)/128. 	    ;}
template<typename T> __device__ constexpr T RSH_C86_z2() 	{ return 	    RSQRT_PI<T>()*sqrt(14586.)*15./128. 	;}
template<typename T> __device__ constexpr T RSH_C87() 		{ return 		RSQRT_PI<T>()*sqrt(12155.)*3./64. 	    ;}
template<typename T> __device__ constexpr T RSH_C88() 		{ return 		RSQRT_PI<T>()*sqrt(12155.)*3./256.  	;}

// L = 9
template<typename T> __device__ constexpr T RSH_C90_c() 	{ return 	    RSQRT_PI<T>()*sqrt(19.)*315./256. 	    ;}
template<typename T> __device__ constexpr T RSH_C90_z2() 	{ return 	   -RSQRT_PI<T>()*sqrt(19.)*4620./256.      ;}
template<typename T> __device__ constexpr T RSH_C90_z4() 	{ return 	    RSQRT_PI<T>()*sqrt(19.)*18018./256.     ;}
template<typename T> __device__ constexpr T RSH_C90_z6() 	{ return 	   -RSQRT_PI<T>()*sqrt(19.)*25740./256.     ;}
template<typename T> __device__ constexpr T RSH_C90_z8() 	{ return 	    RSQRT_PI<T>()*sqrt(19.)*12155./256.     ;}
template<typename T> __device__ constexpr T RSH_C91_c() 	{ return 	    RSQRT_PI<T>()*sqrt(95.)*21./256. 	    ;}
template<typename T> __device__ constexpr T RSH_C91_z2() 	{ return 	   -RSQRT_PI<T>()*sqrt(95.)*924./256. 	    ;}
template<typename T> __device__ constexpr T RSH_C91_z4() 	{ return 	    RSQRT_PI<T>()*sqrt(95.)*6006./256. 	    ;}
template<typename T> __device__ constexpr T RSH_C91_z6() 	{ return 	   -RSQRT_PI<T>()*sqrt(95.)*12012./256. 	;}
template<typename T> __device__ constexpr T RSH_C91_z8() 	{ return 	    RSQRT_PI<T>()*sqrt(95.)*7293./256. 	    ;}
template<typename T> __device__ constexpr T RSH_C92_c() 	{ return 	   -RSQRT_PI<T>()*sqrt(2090.)*21./128. 	    ;}
template<typename T> __device__ constexpr T RSH_C92_z2() 	{ return 	    RSQRT_PI<T>()*sqrt(2090.)*273./128. 	;}
template<typename T> __device__ constexpr T RSH_C92_z4() 	{ return 	   -RSQRT_PI<T>()*sqrt(2090.)*819./128. 	;}
template<typename T> __device__ constexpr T RSH_C92_z6() 	{ return 	    RSQRT_PI<T>()*sqrt(2090.)*663./128. 	;}
template<typename T> __device__ constexpr T RSH_C93_c() 	{ return 	   -RSQRT_PI<T>()*sqrt(43890.)/256. 	    ;}
template<typename T> __device__ constexpr T RSH_C93_z2() 	{ return 	    RSQRT_PI<T>()*sqrt(43890.)*39./256. 	;}
template<typename T> __device__ constexpr T RSH_C93_z4() 	{ return 	   -RSQRT_PI<T>()*sqrt(43890.)*195./256. 	;}
template<typename T> __device__ constexpr T RSH_C93_z6() 	{ return 	    RSQRT_PI<T>()*sqrt(43890.)*221./256. 	;}
template<typename T> __device__ constexpr T RSH_C94_c() 	{ return 	    RSQRT_PI<T>()*sqrt(95095.)*3./128. 	    ;}
template<typename T> __device__ constexpr T RSH_C94_z2() 	{ return 	   -RSQRT_PI<T>()*sqrt(95095.)*30./128. 	;}
template<typename T> __device__ constexpr T RSH_C94_z4() 	{ return 	    RSQRT_PI<T>()*sqrt(95095.)*51./128. 	;}
template<typename T> __device__ constexpr T RSH_C95_c() 	{ return 	    RSQRT_PI<T>()*sqrt(5434.)*3./256. 	    ;}
template<typename T> __device__ constexpr T RSH_C95_z2() 	{ return 	   -RSQRT_PI<T>()*sqrt(5434.)*90./256. 	    ;}
template<typename T> __device__ constexpr T RSH_C95_z4() 	{ return 	    RSQRT_PI<T>()*sqrt(5434.)*255./256. 	;}
template<typename T> __device__ constexpr T RSH_C96_c() 	{ return 	   -RSQRT_PI<T>()*sqrt(81510.)*3./128. 	    ;}
template<typename T> __device__ constexpr T RSH_C96_z2() 	{ return 	    RSQRT_PI<T>()*sqrt(81510.)*17./128. 	;}
template<typename T> __device__ constexpr T RSH_C97_c() 	{ return 	   -RSQRT_PI<T>()*sqrt(27170.)*3./512. 	    ;}
template<typename T> __device__ constexpr T RSH_C97_z2() 	{ return 		RSQRT_PI<T>()*sqrt(27170.)*51./512. 	;}
template<typename T> __device__ constexpr T RSH_C98() 		{ return 		RSQRT_PI<T>()*sqrt(230945.)*3./256.  	;}
template<typename T> __device__ constexpr T RSH_C99() 		{ return 		RSQRT_PI<T>()*sqrt(461890.)/512.  	    ;}

// L = 10
template<typename T> __device__ constexpr T RSH_CA0_c() 	{ return 	   -RSQRT_PI<T>()*sqrt(21.)*63./512. 	    ;}
template<typename T> __device__ constexpr T RSH_CA0_z2() 	{ return 	    RSQRT_PI<T>()*sqrt(21.)*3465./512.      ;}
template<typename T> __device__ constexpr T RSH_CA0_z4() 	{ return 	   -RSQRT_PI<T>()*sqrt(21.)*30030./512.     ;}
template<typename T> __device__ constexpr T RSH_CA0_z6() 	{ return 	    RSQRT_PI<T>()*sqrt(21.)*90090./512.     ;}
template<typename T> __device__ constexpr T RSH_CA0_z8() 	{ return 	   -RSQRT_PI<T>()*sqrt(21.)*109395./512.    ;}
template<typename T> __device__ constexpr T RSH_CA0_zA() 	{ return 	    RSQRT_PI<T>()*sqrt(21.)*46189./512.     ;}
template<typename T> __device__ constexpr T RSH_CA1_c() 	{ return 	    RSQRT_PI<T>()*sqrt(1155.)*63./256. 	    ;}
template<typename T> __device__ constexpr T RSH_CA1_z2() 	{ return 	   -RSQRT_PI<T>()*sqrt(1155.)*1092./256. 	;}
template<typename T> __device__ constexpr T RSH_CA1_z4() 	{ return 	    RSQRT_PI<T>()*sqrt(1155.)*4914./256. 	;}
template<typename T> __device__ constexpr T RSH_CA1_z6() 	{ return 	   -RSQRT_PI<T>()*sqrt(1155.)*7956./256. 	;}
template<typename T> __device__ constexpr T RSH_CA1_z8() 	{ return 	    RSQRT_PI<T>()*sqrt(1155.)*4199./256. 	;}
template<typename T> __device__ constexpr T RSH_CA2_c() 	{ return 	    RSQRT_PI<T>()*sqrt(385.)*21./512. 	    ;}
template<typename T> __device__ constexpr T RSH_CA2_z2() 	{ return 	   -RSQRT_PI<T>()*sqrt(385.)*1092./512. 	;}
template<typename T> __device__ constexpr T RSH_CA2_z4() 	{ return 	    RSQRT_PI<T>()*sqrt(385.)*8190./512. 	;}
template<typename T> __device__ constexpr T RSH_CA2_z6() 	{ return 	   -RSQRT_PI<T>()*sqrt(385.)*18564./512. 	;}
template<typename T> __device__ constexpr T RSH_CA2_z8() 	{ return 	    RSQRT_PI<T>()*sqrt(385.)*12597./512. 	;}
template<typename T> __device__ constexpr T RSH_CA3_c() 	{ return 	   -RSQRT_PI<T>()*sqrt(10010.)*21./256. 	;}
template<typename T> __device__ constexpr T RSH_CA3_z2() 	{ return 	    RSQRT_PI<T>()*sqrt(10010.)*315./256. 	;}
template<typename T> __device__ constexpr T RSH_CA3_z4() 	{ return 	   -RSQRT_PI<T>()*sqrt(10010.)*1071./256. 	;}
template<typename T> __device__ constexpr T RSH_CA3_z6() 	{ return 	    RSQRT_PI<T>()*sqrt(10010.)*969./256. 	;}
template<typename T> __device__ constexpr T RSH_CA4_c() 	{ return 	   -RSQRT_PI<T>()*sqrt(5005.)*3./256. 	    ;}
template<typename T> __device__ constexpr T RSH_CA4_z2() 	{ return 	    RSQRT_PI<T>()*sqrt(5005.)*135./256. 	;}
template<typename T> __device__ constexpr T RSH_CA4_z4() 	{ return 	   -RSQRT_PI<T>()*sqrt(5005.)*765./256. 	;}
template<typename T> __device__ constexpr T RSH_CA4_z6() 	{ return 	    RSQRT_PI<T>()*sqrt(5005.)*969./256. 	;}
template<typename T> __device__ constexpr T RSH_CA5_c() 	{ return 	    RSQRT_PI<T>()*sqrt(2002.)*45./256. 	    ;}
template<typename T> __device__ constexpr T RSH_CA5_z2() 	{ return 	   -RSQRT_PI<T>()*sqrt(2002.)*510./256. 	;}
template<typename T> __device__ constexpr T RSH_CA5_z4() 	{ return 	    RSQRT_PI<T>()*sqrt(2002.)*969./256. 	;}
template<typename T> __device__ constexpr T RSH_CA6_c() 	{ return 	    RSQRT_PI<T>()*sqrt(10010.)*9./1024. 	;}
template<typename T> __device__ constexpr T RSH_CA6_z2() 	{ return 	   -RSQRT_PI<T>()*sqrt(10010.)*306./1024. 	;}
template<typename T> __device__ constexpr T RSH_CA6_z4() 	{ return 	    RSQRT_PI<T>()*sqrt(10010.)*969./1024. 	;}
template<typename T> __device__ constexpr T RSH_CA7_c() 	{ return 	   -RSQRT_PI<T>()*sqrt(170170.)*9./512. 	;}
template<typename T> __device__ constexpr T RSH_CA7_z2() 	{ return 		RSQRT_PI<T>()*sqrt(170170.)*57./512. 	;}
template<typename T> __device__ constexpr T RSH_CA8_c() 	{ return 	   -RSQRT_PI<T>()*sqrt(255255.)/512.  	    ;}
template<typename T> __device__ constexpr T RSH_CA8_z2() 	{ return 		RSQRT_PI<T>()*sqrt(255255.)*19./512.  	;}
template<typename T> __device__ constexpr T RSH_CA9() 		{ return 		RSQRT_PI<T>()*sqrt(9699690.)/512.  	    ;}
template<typename T> __device__ constexpr T RSH_CAA() 		{ return 		RSQRT_PI<T>()*sqrt(1939938.)/1024.  	;}


/*
    Compressed sin^m(theta)*[exp^(-i*m*phi) - exp^(i*m*phi)] and sin^m(theta)*[exp^(-i*m*phi) + exp^(i*m*phi)].
    These are shared multipliers for multiple L.

    __forceinline__ forces body of the function to be substituted in the place of the call.
    It proportionally enlarges executable size, but on the other hand saves time otherwise required to resolve function call.
*/
template<typename T> __device__ __forceinline__ T f_phi_nA(const T x, const T y) {  const T x2 = x*x, y2 = y*y, dx2y2 = x2 - y2;
                                                                                    const T dx2y2_sq = dx2y2*dx2y2;                 return x * y * (10.*dx2y2_sq * (dx2y2_sq - 8.*x2*y2) + 32.*x2*x2*y2*y2) ; }
template<typename T> __device__ __forceinline__ T f_phi_n9(const T x, const T y) {  const T x2 = x*x, y2 = y*y;
                                                                                    const T x4 = x2*x2, y4 = y2*y2;                 return y * (y4*y4 + 126.*x4*y4 + 9.*x4*x4 - x2*y2 * (36.*y4 + 84*x4))   ; }
template<typename T> __device__ __forceinline__ T f_phi_n8(const T x, const T y) {  const T x2 = x*x, y2 = y*y, dx2y2 = x2 - y2;    return x * y * dx2y2 * (dx2y2*dx2y2 - 4.*x2*y2) * 8.                    ; }
template<typename T> __device__ __forceinline__ T f_phi_n7(const T x, const T y) {  const T x2 = x*x, y2 = y*y; 			        return -y * (y2*y2 * (y2 - 21.*x2) + 7.*x2*x2 * (5.*y2 - x2))           ; }
template<typename T> __device__ __forceinline__ T f_phi_n6(const T x, const T y) {  const T x2 = x*x, y2 = y*y; 				    return x * y * (3.*x2 - y2) * (x2 - 3.*y2) * 2.                         ; }
template<typename T> __device__ __forceinline__ T f_phi_n5(const T x, const T y) {  const T x2 = x*x, y2 = y*y; 				    return y * (y2*y2 + 5.*x2 * (x2 - 2.*y2))	                            ; }
template<typename T> __device__ __forceinline__ T f_phi_n4(const T x, const T y) { 												    return x * y * (x + y) * (x - y) * 4.		                            ; }
template<typename T> __device__ __forceinline__ T f_phi_n3(const T x, const T y) { 												    return y * (3.*x*x - y*y)				                                ; }
template<typename T> __device__ __forceinline__ T f_phi_n2(const T x, const T y) { 												    return x * y * 2.							                            ; }
template<typename T> __device__ __forceinline__ T f_phi_n1(const T x, const T y) { 												    return y								                                ; }

template<typename T> __device__ __forceinline__ T f_phi_p1(const T x, const T y) { 													return x								                                ; }
template<typename T> __device__ __forceinline__ T f_phi_p2(const T x, const T y) { 													return (x + y) * (x - y)				                                ; }
template<typename T> __device__ __forceinline__ T f_phi_p3(const T x, const T y) { 													return x * (x*x - 3.*y*y)				                                ; }
template<typename T> __device__ __forceinline__ T f_phi_p4(const T x, const T y) {  const T x2 = x*x, y2 = y*y; 					return x2 * (x2 - 6.*y2) + y2*y2			                            ; }
template<typename T> __device__ __forceinline__ T f_phi_p5(const T x, const T y) {  const T x2 = x*x, y2 = y*y; 					return x * (x2*x2 + 5.*y2 * (y2 - 2.*x2))	                            ; }
template<typename T> __device__ __forceinline__ T f_phi_p6(const T x, const T y) {  const T x2 = x*x, y2 = y*y, dx2y2 = x2 - y2;    return dx2y2 * (dx2y2*dx2y2 - 12.*x2*y2)                                ; }
template<typename T> __device__ __forceinline__ T f_phi_p7(const T x, const T y) {  const T x2 = x*x, y2 = y*y; 					return x * (x2*x2 * (x2 - 21.*y2) + 7.*y2*y2 * (5.*x2 - y2))            ; }
template<typename T> __device__ __forceinline__ T f_phi_p8(const T x, const T y) {  const T x2 = x*x, y2 = y*y, dx2y2 = x2 - y2;
                                                                                    const T dx2y2_sq = dx2y2*dx2y2;                 return dx2y2_sq*dx2y2_sq + x2*y2 * (16.*x2*y2 - 24.*dx2y2_sq)           ; }
template<typename T> __device__ __forceinline__ T f_phi_p9(const T x, const T y) {  const T x2 = x*x, y2 = y*y;
                                                                                    const T x4 = x2*x2, y4 = y2*y2;                 return x * (x4*x4 + 126.*x4*y4 + 9.*y4*y4 - x2*y2 * (36.*x4 + 84*y4))   ; }
template<typename T> __device__ __forceinline__ T f_phi_pA(const T x, const T y) {  const T x2 = x*x, y2 = y*y, dx2y2 = x2 - y2;
                                                                                    const T dx2y2_sq = dx2y2*dx2y2;                 return dx2y2 * (dx2y2_sq*dx2y2_sq + 40.*x2*y2 * (2.*x2*y2 - dx2y2_sq))  ; }

/*
    Polynoms in z.
*/
template<typename T> __device__ __forceinline__ T p_c  (const T z,  const T c0)                         {                       return c0                                                               ; }
template<typename T> __device__ __forceinline__ T p_z  (const T z,  const T c1)                         {                       return c1 * z                                                           ; }
template<typename T> __device__ __forceinline__ T p_z2 (const T z,  const T c0, const T c2)             {                       return c0 + c2 * z * z                                                  ; }
template<typename T> __device__ __forceinline__ T p_z2z(const T z,  const T c0, const T c2)             {                       return (c0 + c2 * z * z) * z                                            ; }
template<typename T> __device__ __forceinline__ T p_z4 (const T z,  const T c0, const T c2, const T c4) { const T z2 = z*z;     return c0 + (c2 + c4 * z2) * z2                                         ; }
template<typename T> __device__ __forceinline__ T p_z4z(const T z,  const T c0, const T c2, const T c4) { const T z2 = z*z;     return (c0 + (c2 + c4 * z2) * z2) * z                                   ; }
template<typename T> __device__ __forceinline__ T p_z6 (const T z,  const T c0, const T c2, const T c4,
                                                        const T c6)                                     { const T z2 = z*z;     return c0 + (c2 + (c4 + c6 * z2) * z2) * z2                             ; }
template<typename T> __device__ __forceinline__ T p_z6z(const T z,  const T c0, const T c2, const T c4,
                                                        const T c6)                                     { const T z2 = z*z;     return (c0 + (c2 + (c4 + c6 * z2) * z2) * z2) * z                       ; }
template<typename T> __device__ __forceinline__ T p_z8 (const T z,  const T c0, const T c2, const T c4,
                                                        const T c6, const T c8)                         { const T z2 = z*z;     return c0 + (c2 + (c4 + (c6 + c8 * z2) * z2) * z2) * z2                 ; }
template<typename T> __device__ __forceinline__ T p_z8z(const T z,  const T c0, const T c2, const T c4,
                                                        const T c6, const T c8)                         { const T z2 = z*z;     return (c0 + (c2 + (c4 + (c6 + c8 * z2) * z2) * z2) * z2) * z           ; }
template<typename T> __device__ __forceinline__ T p_zA (const T z,  const T c0, const T c2, const T c4,
                                                        const T c6, const T c8, const T c10)            { const T z2 = z*z;     return c0 + (c2 + (c4 + (c6 + (c8 + c10 * z2) * z2) * z2) * z2) * z2    ; }

/*
    Handle special case of xyz = (0., 0., 0.) with additional multiplier, either 0. or 1.
*/
template<typename T> __device__ __forceinline__ T special(const T x, const T y, const T z) { return (T) (x != 0. || y != 0. || z != 0.); }


/*
    Functions for specific (L, m). Product of common multiplier from m and polynomial in z.
    Cases (L, 0) with L even and > 0, have additional multiplier to treat special case of input (0, 0, 0) - return 0.
    Multiplier is either 1. if any of x, y, z is different from 0, or 0. otherwise.
    It is constructed as a multiplier in opposite to if-else statement in order to avoid branch divergence.
*/

template<typename T> __device__ T sh00 (const T x, const T y, const T z) { return RSH_C00<T>()                                                                                                                          ; }

template<typename T> __device__ T sh1n1(const T x, const T y, const T z) { return f_phi_n1<T>(x, y) * p_c  <T>(z, RSH_C11<T>())                                                                                         ; }
template<typename T> __device__ T sh10 (const T x, const T y, const T z) { return 				      p_z  <T>(z, RSH_C10<T>())                                                                                         ; }
template<typename T> __device__ T sh1p1(const T x, const T y, const T z) { return f_phi_p1<T>(x, y) * p_c  <T>(z, RSH_C11<T>())                                                                                         ; }

template<typename T> __device__ T sh2n2(const T x, const T y, const T z) { return f_phi_n2<T>(x, y) * p_c  <T>(z, RSH_C22<T>())                                                                                         ; }
template<typename T> __device__ T sh2n1(const T x, const T y, const T z) { return f_phi_n1<T>(x, y) * p_z  <T>(z, RSH_C21<T>())                                                                                         ; }
template<typename T> __device__ T sh20 (const T x, const T y, const T z) { return special<T>(x,y,z) * p_z2 <T>(z, RSH_C20_c<T>(), RSH_C20_z2<T>())                                                                      ; }
template<typename T> __device__ T sh2p1(const T x, const T y, const T z) { return f_phi_p1<T>(x, y) * p_z  <T>(z, RSH_C21<T>())                                                                                         ; }
template<typename T> __device__ T sh2p2(const T x, const T y, const T z) { return f_phi_p2<T>(x, y) * p_c  <T>(z, RSH_C22<T>())                                                                                         ; }

template<typename T> __device__ T sh3n3(const T x, const T y, const T z) { return f_phi_n3<T>(x, y) * p_c  <T>(z, RSH_C33<T>())                                                                                         ; }
template<typename T> __device__ T sh3n2(const T x, const T y, const T z) { return f_phi_n2<T>(x, y) * p_z  <T>(z, RSH_C32<T>())                                                                                         ; }
template<typename T> __device__ T sh3n1(const T x, const T y, const T z) { return f_phi_n1<T>(x, y) * p_z2 <T>(z, RSH_C31_c<T>(), RSH_C31_z2<T>())                                                                      ; }
template<typename T> __device__ T sh30 (const T x, const T y, const T z) { return 				      p_z2z<T>(z, RSH_C30_c<T>(), RSH_C30_z2<T>())                                                                      ; }
template<typename T> __device__ T sh3p1(const T x, const T y, const T z) { return f_phi_p1<T>(x, y) * p_z2 <T>(z, RSH_C31_c<T>(), RSH_C31_z2<T>())                                                                      ; }
template<typename T> __device__ T sh3p2(const T x, const T y, const T z) { return f_phi_p2<T>(x, y) * p_z  <T>(z, RSH_C32<T>())                                                                                         ; }
template<typename T> __device__ T sh3p3(const T x, const T y, const T z) { return f_phi_p3<T>(x, y) * p_c  <T>(z, RSH_C33<T>())                                                                                         ; }

template<typename T> __device__ T sh4n4(const T x, const T y, const T z) { return f_phi_n4<T>(x, y) * p_c  <T>(z, RSH_C44<T>())                                                                                         ; }
template<typename T> __device__ T sh4n3(const T x, const T y, const T z) { return f_phi_n3<T>(x, y) * p_z  <T>(z, RSH_C43<T>())                                                                                         ; }
template<typename T> __device__ T sh4n2(const T x, const T y, const T z) { return f_phi_n2<T>(x, y) * p_z2 <T>(z, RSH_C42_c<T>(), RSH_C42_z2<T>())                                                                      ; }
template<typename T> __device__ T sh4n1(const T x, const T y, const T z) { return f_phi_n1<T>(x, y) * p_z2z<T>(z, RSH_C41_c<T>(), RSH_C41_z2<T>())                                                                      ; }
template<typename T> __device__ T sh40 (const T x, const T y, const T z) { return special<T>(x,y,z) * p_z4 <T>(z, RSH_C40_c<T>(), RSH_C40_z2<T>(), RSH_C40_z4<T>())                                                     ; }
template<typename T> __device__ T sh4p1(const T x, const T y, const T z) { return f_phi_p1<T>(x, y) * p_z2z<T>(z, RSH_C41_c<T>(), RSH_C41_z2<T>())                                                                      ; }
template<typename T> __device__ T sh4p2(const T x, const T y, const T z) { return f_phi_p2<T>(x, y) * p_z2 <T>(z, RSH_C42_c<T>(), RSH_C42_z2<T>())                                                                      ; }
template<typename T> __device__ T sh4p3(const T x, const T y, const T z) { return f_phi_p3<T>(x, y) * p_z  <T>(z, RSH_C43<T>())                                                                                         ; }
template<typename T> __device__ T sh4p4(const T x, const T y, const T z) { return f_phi_p4<T>(x, y) * p_c  <T>(z, RSH_C44<T>())                                                                                         ; }

template<typename T> __device__ T sh5n5(const T x, const T y, const T z) { return f_phi_n5<T>(x, y) * p_c  <T>(z, RSH_C55<T>())                                                                                         ; }
template<typename T> __device__ T sh5n4(const T x, const T y, const T z) { return f_phi_n4<T>(x, y) * p_z  <T>(z, RSH_C54<T>())                                                                                         ; }
template<typename T> __device__ T sh5n3(const T x, const T y, const T z) { return f_phi_n3<T>(x, y) * p_z2 <T>(z, RSH_C53_c<T>(), RSH_C53_z2<T>())                                                                      ; }
template<typename T> __device__ T sh5n2(const T x, const T y, const T z) { return f_phi_n2<T>(x, y) * p_z2z<T>(z, RSH_C52_c<T>(), RSH_C52_z2<T>())                                                                      ; }
template<typename T> __device__ T sh5n1(const T x, const T y, const T z) { return f_phi_n1<T>(x, y) * p_z4 <T>(z, RSH_C51_c<T>(), RSH_C51_z2<T>(), RSH_C51_z4<T>())                                                     ; }
template<typename T> __device__ T sh50 (const T x, const T y, const T z) { return 				      p_z4z<T>(z, RSH_C50_c<T>(), RSH_C50_z2<T>(), RSH_C50_z4<T>())                                                     ; }
template<typename T> __device__ T sh5p1(const T x, const T y, const T z) { return f_phi_p1<T>(x, y) * p_z4 <T>(z, RSH_C51_c<T>(), RSH_C51_z2<T>(), RSH_C51_z4<T>())                                                     ; }
template<typename T> __device__ T sh5p2(const T x, const T y, const T z) { return f_phi_p2<T>(x, y) * p_z2z<T>(z, RSH_C52_c<T>(), RSH_C52_z2<T>())                                                                      ; }
template<typename T> __device__ T sh5p3(const T x, const T y, const T z) { return f_phi_p3<T>(x, y) * p_z2 <T>(z, RSH_C53_c<T>(), RSH_C53_z2<T>())                                                                      ; }
template<typename T> __device__ T sh5p4(const T x, const T y, const T z) { return f_phi_p4<T>(x, y) * p_z  <T>(z, RSH_C54<T>())                                                                                         ; }
template<typename T> __device__ T sh5p5(const T x, const T y, const T z) { return f_phi_p5<T>(x, y) * p_c  <T>(z, RSH_C55<T>())                                                                                         ; }

template<typename T> __device__ T sh6n6(const T x, const T y, const T z) { return f_phi_n6<T>(x, y) * p_c  <T>(z, RSH_C66<T>())                                                                                         ; }
template<typename T> __device__ T sh6n5(const T x, const T y, const T z) { return f_phi_n5<T>(x, y) * p_z  <T>(z, RSH_C65<T>())                                                                                         ; }
template<typename T> __device__ T sh6n4(const T x, const T y, const T z) { return f_phi_n4<T>(x, y) * p_z2 <T>(z, RSH_C64_c<T>(), RSH_C64_z2<T>())                                                                      ; }
template<typename T> __device__ T sh6n3(const T x, const T y, const T z) { return f_phi_n3<T>(x, y) * p_z2z<T>(z, RSH_C63_c<T>(), RSH_C63_z2<T>())                                                                      ; }
template<typename T> __device__ T sh6n2(const T x, const T y, const T z) { return f_phi_n2<T>(x, y) * p_z4 <T>(z, RSH_C62_c<T>(), RSH_C62_z2<T>(), RSH_C62_z4<T>())                                                     ; }
template<typename T> __device__ T sh6n1(const T x, const T y, const T z) { return f_phi_n1<T>(x, y) * p_z4z<T>(z, RSH_C61_c<T>(), RSH_C61_z2<T>(), RSH_C61_z4<T>())                                                     ; }
template<typename T> __device__ T sh60 (const T x, const T y, const T z) { return special<T>(x,y,z) * p_z6 <T>(z, RSH_C60_c<T>(), RSH_C60_z2<T>(), RSH_C60_z4<T>(), RSH_C60_z6<T>())                                    ; }
template<typename T> __device__ T sh6p1(const T x, const T y, const T z) { return f_phi_p1<T>(x, y) * p_z4z<T>(z, RSH_C61_c<T>(), RSH_C61_z2<T>(), RSH_C61_z4<T>())                                                     ; }
template<typename T> __device__ T sh6p2(const T x, const T y, const T z) { return f_phi_p2<T>(x, y) * p_z4 <T>(z, RSH_C62_c<T>(), RSH_C62_z2<T>(), RSH_C62_z4<T>())                                                     ; }
template<typename T> __device__ T sh6p3(const T x, const T y, const T z) { return f_phi_p3<T>(x, y) * p_z2z<T>(z, RSH_C63_c<T>(), RSH_C63_z2<T>())                                                                      ; }
template<typename T> __device__ T sh6p4(const T x, const T y, const T z) { return f_phi_p4<T>(x, y) * p_z2 <T>(z, RSH_C64_c<T>(), RSH_C64_z2<T>())                                                                      ; }
template<typename T> __device__ T sh6p5(const T x, const T y, const T z) { return f_phi_p5<T>(x, y) * p_z  <T>(z, RSH_C65<T>())                                                                                         ; }
template<typename T> __device__ T sh6p6(const T x, const T y, const T z) { return f_phi_p6<T>(x, y) * p_c  <T>(z, RSH_C66<T>())                                                                                         ; }

template<typename T> __device__ T sh7n7(const T x, const T y, const T z) { return f_phi_n7<T>(x, y) * p_c  <T>(z, RSH_C77<T>())                                                                                         ; }
template<typename T> __device__ T sh7n6(const T x, const T y, const T z) { return f_phi_n6<T>(x, y) * p_z  <T>(z, RSH_C76<T>())                                                                                         ; }
template<typename T> __device__ T sh7n5(const T x, const T y, const T z) { return f_phi_n5<T>(x, y) * p_z2 <T>(z, RSH_C75_c<T>(), RSH_C75_z2<T>())                                                                      ; }
template<typename T> __device__ T sh7n4(const T x, const T y, const T z) { return f_phi_n4<T>(x, y) * p_z2z<T>(z, RSH_C74_c<T>(), RSH_C74_z2<T>())                                                                      ; }
template<typename T> __device__ T sh7n3(const T x, const T y, const T z) { return f_phi_n3<T>(x, y) * p_z4 <T>(z, RSH_C73_c<T>(), RSH_C73_z2<T>(), RSH_C73_z4<T>())                                                     ; }
template<typename T> __device__ T sh7n2(const T x, const T y, const T z) { return f_phi_n2<T>(x, y) * p_z4z<T>(z, RSH_C72_c<T>(), RSH_C72_z2<T>(), RSH_C72_z4<T>())                                                     ; }
template<typename T> __device__ T sh7n1(const T x, const T y, const T z) { return f_phi_n1<T>(x, y) * p_z6 <T>(z, RSH_C71_c<T>(), RSH_C71_z2<T>(), RSH_C71_z4<T>(), RSH_C71_z6<T>())                                    ; }
template<typename T> __device__ T sh70 (const T x, const T y, const T z) { return                     p_z6z<T>(z, RSH_C70_c<T>(), RSH_C70_z2<T>(), RSH_C70_z4<T>(), RSH_C70_z6<T>())                                    ; }
template<typename T> __device__ T sh7p1(const T x, const T y, const T z) { return f_phi_p1<T>(x, y) * p_z6 <T>(z, RSH_C71_c<T>(), RSH_C71_z2<T>(), RSH_C71_z4<T>(), RSH_C71_z6<T>())                                    ; }
template<typename T> __device__ T sh7p2(const T x, const T y, const T z) { return f_phi_p2<T>(x, y) * p_z4z<T>(z, RSH_C72_c<T>(), RSH_C72_z2<T>(), RSH_C72_z4<T>())                                                     ; }
template<typename T> __device__ T sh7p3(const T x, const T y, const T z) { return f_phi_p3<T>(x, y) * p_z4 <T>(z, RSH_C73_c<T>(), RSH_C73_z2<T>(), RSH_C73_z4<T>())                                                     ; }
template<typename T> __device__ T sh7p4(const T x, const T y, const T z) { return f_phi_p4<T>(x, y) * p_z2z<T>(z, RSH_C74_c<T>(), RSH_C74_z2<T>())                                                                      ; }
template<typename T> __device__ T sh7p5(const T x, const T y, const T z) { return f_phi_p5<T>(x, y) * p_z2 <T>(z, RSH_C75_c<T>(), RSH_C75_z2<T>())                                                                      ; }
template<typename T> __device__ T sh7p6(const T x, const T y, const T z) { return f_phi_p6<T>(x, y) * p_z  <T>(z, RSH_C76<T>())                                                                                         ; }
template<typename T> __device__ T sh7p7(const T x, const T y, const T z) { return f_phi_p7<T>(x, y) * p_c  <T>(z, RSH_C77<T>())                                                                                         ; }

template<typename T> __device__ T sh8n8(const T x, const T y, const T z) { return f_phi_n8<T>(x, y) * p_c  <T>(z, RSH_C88<T>())                                                                                         ; }
template<typename T> __device__ T sh8n7(const T x, const T y, const T z) { return f_phi_n7<T>(x, y) * p_z  <T>(z, RSH_C87<T>())                                                                                         ; }
template<typename T> __device__ T sh8n6(const T x, const T y, const T z) { return f_phi_n6<T>(x, y) * p_z2 <T>(z, RSH_C86_c<T>(), RSH_C86_z2<T>())                                                                      ; }
template<typename T> __device__ T sh8n5(const T x, const T y, const T z) { return f_phi_n5<T>(x, y) * p_z2z<T>(z, RSH_C85_c<T>(), RSH_C85_z2<T>())                                                                      ; }
template<typename T> __device__ T sh8n4(const T x, const T y, const T z) { return f_phi_n4<T>(x, y) * p_z4 <T>(z, RSH_C84_c<T>(), RSH_C84_z2<T>(), RSH_C84_z4<T>())                                                     ; }
template<typename T> __device__ T sh8n3(const T x, const T y, const T z) { return f_phi_n3<T>(x, y) * p_z4z<T>(z, RSH_C83_c<T>(), RSH_C83_z2<T>(), RSH_C83_z4<T>())                                                     ; }
template<typename T> __device__ T sh8n2(const T x, const T y, const T z) { return f_phi_n2<T>(x, y) * p_z6 <T>(z, RSH_C82_c<T>(), RSH_C82_z2<T>(), RSH_C82_z4<T>(), RSH_C82_z6<T>())                                    ; }
template<typename T> __device__ T sh8n1(const T x, const T y, const T z) { return f_phi_n1<T>(x, y) * p_z6z<T>(z, RSH_C81_c<T>(), RSH_C81_z2<T>(), RSH_C81_z4<T>(), RSH_C81_z6<T>())                                    ; }
template<typename T> __device__ T sh80 (const T x, const T y, const T z) { return special<T>(x,y,z) * p_z8 <T>(z, RSH_C80_c<T>(), RSH_C80_z2<T>(), RSH_C80_z4<T>(), RSH_C80_z6<T>(), RSH_C80_z8<T>())                   ; }
template<typename T> __device__ T sh8p1(const T x, const T y, const T z) { return f_phi_p1<T>(x, y) * p_z6z<T>(z, RSH_C81_c<T>(), RSH_C81_z2<T>(), RSH_C81_z4<T>(), RSH_C81_z6<T>())                                    ; }
template<typename T> __device__ T sh8p2(const T x, const T y, const T z) { return f_phi_p2<T>(x, y) * p_z6 <T>(z, RSH_C82_c<T>(), RSH_C82_z2<T>(), RSH_C82_z4<T>(), RSH_C82_z6<T>())                                    ; }
template<typename T> __device__ T sh8p3(const T x, const T y, const T z) { return f_phi_p3<T>(x, y) * p_z4z<T>(z, RSH_C83_c<T>(), RSH_C83_z2<T>(), RSH_C83_z4<T>())                                                     ; }
template<typename T> __device__ T sh8p4(const T x, const T y, const T z) { return f_phi_p4<T>(x, y) * p_z4 <T>(z, RSH_C84_c<T>(), RSH_C84_z2<T>(), RSH_C84_z4<T>())                                                     ; }
template<typename T> __device__ T sh8p5(const T x, const T y, const T z) { return f_phi_p5<T>(x, y) * p_z2z<T>(z, RSH_C85_c<T>(), RSH_C85_z2<T>())                                                                      ; }
template<typename T> __device__ T sh8p6(const T x, const T y, const T z) { return f_phi_p6<T>(x, y) * p_z2 <T>(z, RSH_C86_c<T>(), RSH_C86_z2<T>())                                                                      ; }
template<typename T> __device__ T sh8p7(const T x, const T y, const T z) { return f_phi_p7<T>(x, y) * p_z  <T>(z, RSH_C87<T>())                                                                                         ; }
template<typename T> __device__ T sh8p8(const T x, const T y, const T z) { return f_phi_p8<T>(x, y) * p_c  <T>(z, RSH_C88<T>())                                                                                         ; }

template<typename T> __device__ T sh9n9(const T x, const T y, const T z) { return f_phi_n9<T>(x, y) * p_c  <T>(z, RSH_C99<T>())                                                                                         ; }
template<typename T> __device__ T sh9n8(const T x, const T y, const T z) { return f_phi_n8<T>(x, y) * p_z  <T>(z, RSH_C98<T>())                                                                                         ; }
template<typename T> __device__ T sh9n7(const T x, const T y, const T z) { return f_phi_n7<T>(x, y) * p_z2 <T>(z, RSH_C97_c<T>(), RSH_C97_z2<T>())                                                                      ; }
template<typename T> __device__ T sh9n6(const T x, const T y, const T z) { return f_phi_n6<T>(x, y) * p_z2z<T>(z, RSH_C96_c<T>(), RSH_C96_z2<T>())                                                                      ; }
template<typename T> __device__ T sh9n5(const T x, const T y, const T z) { return f_phi_n5<T>(x, y) * p_z4 <T>(z, RSH_C95_c<T>(), RSH_C95_z2<T>(), RSH_C95_z4<T>())                                                     ; }
template<typename T> __device__ T sh9n4(const T x, const T y, const T z) { return f_phi_n4<T>(x, y) * p_z4z<T>(z, RSH_C94_c<T>(), RSH_C94_z2<T>(), RSH_C94_z4<T>())                                                     ; }
template<typename T> __device__ T sh9n3(const T x, const T y, const T z) { return f_phi_n3<T>(x, y) * p_z6 <T>(z, RSH_C93_c<T>(), RSH_C93_z2<T>(), RSH_C93_z4<T>(), RSH_C93_z6<T>())                                    ; }
template<typename T> __device__ T sh9n2(const T x, const T y, const T z) { return f_phi_n2<T>(x, y) * p_z6z<T>(z, RSH_C92_c<T>(), RSH_C92_z2<T>(), RSH_C92_z4<T>(), RSH_C92_z6<T>())                                    ; }
template<typename T> __device__ T sh9n1(const T x, const T y, const T z) { return f_phi_n1<T>(x, y) * p_z8 <T>(z, RSH_C91_c<T>(), RSH_C91_z2<T>(), RSH_C91_z4<T>(), RSH_C91_z6<T>(), RSH_C91_z8<T>())                   ; }
template<typename T> __device__ T sh90 (const T x, const T y, const T z) { return                     p_z8z<T>(z, RSH_C90_c<T>(), RSH_C90_z2<T>(), RSH_C90_z4<T>(), RSH_C90_z6<T>(), RSH_C90_z8<T>())                   ; }
template<typename T> __device__ T sh9p1(const T x, const T y, const T z) { return f_phi_p1<T>(x, y) * p_z8 <T>(z, RSH_C91_c<T>(), RSH_C91_z2<T>(), RSH_C91_z4<T>(), RSH_C91_z6<T>(), RSH_C91_z8<T>())                   ; }
template<typename T> __device__ T sh9p2(const T x, const T y, const T z) { return f_phi_p2<T>(x, y) * p_z6z<T>(z, RSH_C92_c<T>(), RSH_C92_z2<T>(), RSH_C92_z4<T>(), RSH_C92_z6<T>())                                    ; }
template<typename T> __device__ T sh9p3(const T x, const T y, const T z) { return f_phi_p3<T>(x, y) * p_z6 <T>(z, RSH_C93_c<T>(), RSH_C93_z2<T>(), RSH_C93_z4<T>(), RSH_C93_z6<T>())                                    ; }
template<typename T> __device__ T sh9p4(const T x, const T y, const T z) { return f_phi_p4<T>(x, y) * p_z4z<T>(z, RSH_C94_c<T>(), RSH_C94_z2<T>(), RSH_C94_z4<T>())                                                     ; }
template<typename T> __device__ T sh9p5(const T x, const T y, const T z) { return f_phi_p5<T>(x, y) * p_z4 <T>(z, RSH_C95_c<T>(), RSH_C95_z2<T>(), RSH_C95_z4<T>())                                                     ; }
template<typename T> __device__ T sh9p6(const T x, const T y, const T z) { return f_phi_p6<T>(x, y) * p_z2z<T>(z, RSH_C96_c<T>(), RSH_C96_z2<T>())                                                                      ; }
template<typename T> __device__ T sh9p7(const T x, const T y, const T z) { return f_phi_p7<T>(x, y) * p_z2 <T>(z, RSH_C97_c<T>(), RSH_C97_z2<T>())                                                                      ; }
template<typename T> __device__ T sh9p8(const T x, const T y, const T z) { return f_phi_p8<T>(x, y) * p_z  <T>(z, RSH_C98<T>())                                                                                         ; }
template<typename T> __device__ T sh9p9(const T x, const T y, const T z) { return f_phi_p9<T>(x, y) * p_c  <T>(z, RSH_C99<T>())                                                                                         ; }

template<typename T> __device__ T shAnA(const T x, const T y, const T z) { return f_phi_nA<T>(x, y) * p_c  <T>(z, RSH_CAA<T>())                                                                                         ; }
template<typename T> __device__ T shAn9(const T x, const T y, const T z) { return f_phi_n9<T>(x, y) * p_z  <T>(z, RSH_CA9<T>())                                                                                         ; }
template<typename T> __device__ T shAn8(const T x, const T y, const T z) { return f_phi_n8<T>(x, y) * p_z2 <T>(z, RSH_CA8_c<T>(), RSH_CA8_z2<T>())                                                                      ; }
template<typename T> __device__ T shAn7(const T x, const T y, const T z) { return f_phi_n7<T>(x, y) * p_z2z<T>(z, RSH_CA7_c<T>(), RSH_CA7_z2<T>())                                                                      ; }
template<typename T> __device__ T shAn6(const T x, const T y, const T z) { return f_phi_n6<T>(x, y) * p_z4 <T>(z, RSH_CA6_c<T>(), RSH_CA6_z2<T>(), RSH_CA6_z4<T>())                                                     ; }
template<typename T> __device__ T shAn5(const T x, const T y, const T z) { return f_phi_n5<T>(x, y) * p_z4z<T>(z, RSH_CA5_c<T>(), RSH_CA5_z2<T>(), RSH_CA5_z4<T>())                                                     ; }
template<typename T> __device__ T shAn4(const T x, const T y, const T z) { return f_phi_n4<T>(x, y) * p_z6 <T>(z, RSH_CA4_c<T>(), RSH_CA4_z2<T>(), RSH_CA4_z4<T>(), RSH_CA4_z6<T>())                                    ; }
template<typename T> __device__ T shAn3(const T x, const T y, const T z) { return f_phi_n3<T>(x, y) * p_z6z<T>(z, RSH_CA3_c<T>(), RSH_CA3_z2<T>(), RSH_CA3_z4<T>(), RSH_CA3_z6<T>())                                    ; }
template<typename T> __device__ T shAn2(const T x, const T y, const T z) { return f_phi_n2<T>(x, y) * p_z8 <T>(z, RSH_CA2_c<T>(), RSH_CA2_z2<T>(), RSH_CA2_z4<T>(), RSH_CA2_z6<T>(), RSH_CA2_z8<T>())                   ; }
template<typename T> __device__ T shAn1(const T x, const T y, const T z) { return f_phi_n1<T>(x, y) * p_z8z<T>(z, RSH_CA1_c<T>(), RSH_CA1_z2<T>(), RSH_CA1_z4<T>(), RSH_CA1_z6<T>(), RSH_CA1_z8<T>())                   ; }
template<typename T> __device__ T shA0 (const T x, const T y, const T z) { return special<T>(x,y,z) * p_zA <T>(z, RSH_CA0_c<T>(), RSH_CA0_z2<T>(), RSH_CA0_z4<T>(), RSH_CA0_z6<T>(), RSH_CA0_z8<T>(), RSH_CA0_zA<T>())  ; }
template<typename T> __device__ T shAp1(const T x, const T y, const T z) { return f_phi_p1<T>(x, y) * p_z8z<T>(z, RSH_CA1_c<T>(), RSH_CA1_z2<T>(), RSH_CA1_z4<T>(), RSH_CA1_z6<T>(), RSH_CA1_z8<T>())                   ; }
template<typename T> __device__ T shAp2(const T x, const T y, const T z) { return f_phi_p2<T>(x, y) * p_z8 <T>(z, RSH_CA2_c<T>(), RSH_CA2_z2<T>(), RSH_CA2_z4<T>(), RSH_CA2_z6<T>(), RSH_CA2_z8<T>())                   ; }
template<typename T> __device__ T shAp3(const T x, const T y, const T z) { return f_phi_p3<T>(x, y) * p_z6z<T>(z, RSH_CA3_c<T>(), RSH_CA3_z2<T>(), RSH_CA3_z4<T>(), RSH_CA3_z6<T>())                                    ; }
template<typename T> __device__ T shAp4(const T x, const T y, const T z) { return f_phi_p4<T>(x, y) * p_z6 <T>(z, RSH_CA4_c<T>(), RSH_CA4_z2<T>(), RSH_CA4_z4<T>(), RSH_CA4_z6<T>())                                    ; }
template<typename T> __device__ T shAp5(const T x, const T y, const T z) { return f_phi_p5<T>(x, y) * p_z4z<T>(z, RSH_CA5_c<T>(), RSH_CA5_z2<T>(), RSH_CA5_z4<T>())                                                     ; }
template<typename T> __device__ T shAp6(const T x, const T y, const T z) { return f_phi_p6<T>(x, y) * p_z4 <T>(z, RSH_CA6_c<T>(), RSH_CA6_z2<T>(), RSH_CA6_z4<T>())                                                     ; }
template<typename T> __device__ T shAp7(const T x, const T y, const T z) { return f_phi_p7<T>(x, y) * p_z2z<T>(z, RSH_CA7_c<T>(), RSH_CA7_z2<T>())                                                                      ; }
template<typename T> __device__ T shAp8(const T x, const T y, const T z) { return f_phi_p8<T>(x, y) * p_z2 <T>(z, RSH_CA8_c<T>(), RSH_CA8_z2<T>())                                                                      ; }
template<typename T> __device__ T shAp9(const T x, const T y, const T z) { return f_phi_p9<T>(x, y) * p_z  <T>(z, RSH_CA9<T>())                                                                                         ; }
template<typename T> __device__ T shApA(const T x, const T y, const T z) { return f_phi_pA<T>(x, y) * p_c  <T>(z, RSH_CAA<T>())                                                                                         ; }

// array of pointers to functions stored to "constant memory" (__constant__) in GPU - common for all blocks.
template<typename T> __constant__ T (*const fptr[]) (const T, const T, const T) = {
                    		                                                                        sh00<T>, 											                                                                    //                                                     0
                    		                                                              sh1n1<T>, sh10<T>, sh1p1<T>, 									                                                                    //                                                1,   2,   3
                     		                                                    sh2n2<T>, sh2n1<T>, sh20<T>, sh2p1<T>, sh2p2<T>,								                                                            //                                           4,   5,   6,   7,   8
                    		                                          sh3n3<T>, sh3n2<T>, sh3n1<T>, sh30<T>, sh3p1<T>, sh3p2<T>, sh3p3<T>,					                                                                //                                      9,  10,  11,  12,  13,  14,  15
                    		                                sh4n4<T>, sh4n3<T>, sh4n2<T>, sh4n1<T>, sh40<T>, sh4p1<T>, sh4p2<T>, sh4p3<T>, sh4p4<T>,				                                                        //                                16,  17,  18,  19,  20,  21,  22,  23,  24
                    		                      sh5n5<T>, sh5n4<T>, sh5n3<T>, sh5n2<T>, sh5n1<T>, sh50<T>, sh5p1<T>, sh5p2<T>, sh5p3<T>, sh5p4<T>, sh5p5<T>,		                                                        //                           25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35
                    	            	sh6n6<T>, sh6n5<T>, sh6n4<T>, sh6n3<T>, sh6n2<T>, sh6n1<T>, sh60<T>, sh6p1<T>, sh6p2<T>, sh6p3<T>, sh6p4<T>, sh6p5<T>, sh6p6<T>,                                                    //                      36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48
                              sh7n7<T>, sh7n6<T>, sh7n5<T>, sh7n4<T>, sh7n3<T>, sh7n2<T>, sh7n1<T>, sh70<T>, sh7p1<T>, sh7p2<T>, sh7p3<T>, sh7p4<T>, sh7p5<T>, sh7p6<T>, sh7p7<T>, 	                                        //                 49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63
                    sh8n8<T>, sh8n7<T>, sh8n6<T>, sh8n5<T>, sh8n4<T>, sh8n3<T>, sh8n2<T>, sh8n1<T>, sh80<T>, sh8p1<T>, sh8p2<T>, sh8p3<T>, sh8p4<T>, sh8p5<T>, sh8p6<T>, sh8p7<T>, sh8p8<T>,                                //            64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80
          sh9n9<T>, sh9n8<T>, sh9n7<T>, sh9n6<T>, sh9n5<T>, sh9n4<T>, sh9n3<T>, sh9n2<T>, sh9n1<T>, sh90<T>, sh9p1<T>, sh9p2<T>, sh9p3<T>, sh9p4<T>, sh9p5<T>, sh9p6<T>, sh9p7<T>, sh9p8<T>, sh9p9<T>,                      //       81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95.  96,  97,  98,  99
shAnA<T>, shAn9<T>, shAn8<T>, shAn7<T>, shAn6<T>, shAn5<T>, shAn4<T>, shAn3<T>, shAn2<T>, shAn1<T>, shA0<T>, shAp1<T>, shAp2<T>, shAp3<T>, shAp4<T>, shAp5<T>, shAp6<T>, shAp7<T>, shAp8<T>, shAp9<T>, shApA<T>             // 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120
};


/*
    Preceding const means underlying data stays constant.
    Trailing const means that pointer to the data remains constant.
    __restrict__ makes a promise that underlying data can be accessed only with this pointer.
*/
template<typename T>
__global__ void rsh_cuda_kernel(const T* const __restrict__ radii, T* const __restrict__ Ys, const size_t batch_size) {
	const size_t entry_pos = blockDim.x*blockIdx.x + threadIdx.x;                           // position of entry in batch
	if (entry_pos >= batch_size) return;                                                    // early terminate if outside the batch - last warp (of threads) can be only partially filled

	const T x = radii[3*entry_pos];                                                         // "strided memory access" is generally not nice and severely drops throughput
	const T y = radii[3*entry_pos+1];                                                       // padding to 4 and packing in double4 (single read transaction) showed no noticeable improvement (is scale to0 small measure?)
	const T z = radii[3*entry_pos+2];                                                       // 100+ GB/s of throughput would be great, but even 3 GB/s does not make a bottleneck

    Ys[blockIdx.y*batch_size + entry_pos] = fptr<T>[blockIdx.y](x, y, z);                   // select and apply function, store result to the "global memory"
}


void real_spherical_harmonics_cuda(
        torch::Tensor Ys,
        torch::Tensor radii) {
    const size_t filters    = Ys.size(0);
    const size_t batch_size = radii.size(0);

    const size_t threads_per_block = 32;                                                    // warp size in contemporary GPUs is 32 threads, this variable should be a multiple of warp size
    dim3 numBlocks((batch_size + threads_per_block - 1)/threads_per_block, filters, 1);     // batch_size/threads_per_block is fractional in general case - round it up

    if (radii.dtype() == torch::kFloat64) {
        rsh_cuda_kernel<double><<<numBlocks, threads_per_block>>>(
            (const double*) radii.data_ptr(), (double*) Ys.data_ptr(), batch_size
        );
    }
    else {                                                                                  // check in C++ binding guarantee that data type is either double (float64) or float (float32)
        rsh_cuda_kernel<float><<<numBlocks, threads_per_block>>>(
            (const float*) radii.data_ptr(), (float*) Ys.data_ptr(), batch_size
        );
    }
}
