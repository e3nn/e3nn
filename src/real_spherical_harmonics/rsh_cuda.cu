/*
    Calculates real spherical harmonics up to L=10 (inclusive) from Cartesian coordinates.
    Coordinates x, y, z are expected to be normalized (form unit length vector).
*/

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

// 1./sqrt(pi) - at higher precision than actual execution using double type can provide
template<typename T> __device__ constexpr T RSQRT_PI() { return 0.564189583547756286948079451560772585844050629328998856844; }

/*
   Idea of the following list of constexpr functions is to compute coefficients once at compile time,
   and embed them as a direct access constants, same as if they were explicitly written (e.g. 5.5).
   Strictly speaking constexpr expected, but does not guarantee calculations at compile time.
   Here function accepts no parameters, so probability of fail scenario is low.
   When in doubt, run:
        nvcc -ptx rsh_cuda.cu
   and check if constexpr computed in rsh_cuda.ptx file.
*/
template<typename T, int L, int abs_M, int power> __device__ constexpr T RSH_C() {
    T output;
    switch (L*10000 + abs_M*100 + power) {
        // L = 0
        case      0: output =  RSQRT_PI<T>() / 2.;                        break; // (0, 0, 0)
        // L = 1
        case  10001: output =  RSQRT_PI<T>() * sqrt(3.)/2.;               break; // (1, 0, 1)
        case  10100: output =  RSQRT_PI<T>() * sqrt(3.)/2.;               break; // (1, 1, 0)
        // L = 2
        case  20000: output = -RSQRT_PI<T>() * sqrt(5.)/4.;               break; // (2, 0, 0)
        case  20002: output =  RSQRT_PI<T>() * sqrt(5.)*3./4.;            break; // (2, 0, 2)
        case  20101: output =  RSQRT_PI<T>() * sqrt(15.)/2.;              break; // (2, 1, 1)
        case  20200: output =  RSQRT_PI<T>() * sqrt(15.)/4.;              break; // (2, 2, 0)
        // L = 3
        case  30001: output = -RSQRT_PI<T>() * sqrt(7.)*3./4.;            break; // (3, 0, 1)
        case  30003: output =  RSQRT_PI<T>() * sqrt(7.)*5./4.;            break; // (3, 0, 3)
        case  30100: output = -RSQRT_PI<T>() * sqrt(42.)/8.;              break; // (3, 1, 0)
        case  30102: output =  RSQRT_PI<T>() * sqrt(42.)*5./8.;           break; // (3, 1, 2)
        case  30201: output =  RSQRT_PI<T>() * sqrt(105.)/4.;             break; // (3, 2, 1)
        case  30300: output =  RSQRT_PI<T>() * sqrt(70.)/8.;              break; // (3, 3, 0)
        // L = 4
        case  40000: output =  RSQRT_PI<T>() * 9./16.;                    break; // (4, 0, 0)
        case  40002: output = -RSQRT_PI<T>() * 90./16.;                   break; // (4, 0, 2)
        case  40004: output =  RSQRT_PI<T>() * 105./16.;                  break; // (4, 0, 4)
        case  40101: output = -RSQRT_PI<T>() * sqrt(10.)*9./8.;           break; // (4, 1, 1)
        case  40103: output =  RSQRT_PI<T>() * sqrt(10.)*21./8.;          break; // (4, 1, 3)
        case  40200: output = -RSQRT_PI<T>() * sqrt(5.)*3./8.;            break; // (4, 2, 0)
        case  40202: output =  RSQRT_PI<T>() * sqrt(5.)*21./8.;           break; // (4, 2, 2)
        case  40301: output =  RSQRT_PI<T>() * sqrt(70.)*3./8.;           break; // (4, 3, 1)
        case  40400: output =  RSQRT_PI<T>() * sqrt(35.)*3./16.;          break; // (4, 4, 0)
        // L = 5
        case  50001: output =  RSQRT_PI<T>() * sqrt(11.)*15./16.;         break; // (5, 0, 1)
        case  50003: output = -RSQRT_PI<T>() * sqrt(11.)*70./16.;         break; // (5, 0, 3)
        case  50005: output =  RSQRT_PI<T>() * sqrt(11.)*63./16.;         break; // (5, 0, 5)
        case  50100: output =  RSQRT_PI<T>() * sqrt(165.)/16.;            break; // (5, 1, 0)
        case  50102: output = -RSQRT_PI<T>() * sqrt(165.)*14./16.;        break; // (5, 1, 2)
        case  50104: output =  RSQRT_PI<T>() * sqrt(165.)*21./16.;        break; // (5, 1, 4)
        case  50201: output = -RSQRT_PI<T>() * sqrt(1155.)/8.;            break; // (5, 2, 1)
        case  50203: output =  RSQRT_PI<T>() * sqrt(1155.)*3./8.;         break; // (5, 2, 3)
        case  50300: output = -RSQRT_PI<T>() * sqrt(770.)/32.;            break; // (5, 3, 0)
        case  50302: output =  RSQRT_PI<T>() * sqrt(770.)*9./32.;         break; // (5, 3, 2)
        case  50401: output =  RSQRT_PI<T>() * sqrt(385.)*3./16.;         break; // (5, 4, 1)
        case  50500: output =  RSQRT_PI<T>() * sqrt(154.)*3./32.;         break; // (5, 5, 0)
        // L = 6
        case  60000: output = -RSQRT_PI<T>() * sqrt(13.)*5./32.;          break; // (6, 0, 0)
        case  60002: output =  RSQRT_PI<T>() * sqrt(13.)*105./32.;        break; // (6, 0, 2)
        case  60004: output = -RSQRT_PI<T>() * sqrt(13.)*315./32.;        break; // (6, 0, 4)
        case  60006: output =  RSQRT_PI<T>() * sqrt(13.)*231./32.;        break; // (6, 0, 6)
        case  60101: output =  RSQRT_PI<T>() * sqrt(273.)*5./16.;         break; // (6, 1, 1)
        case  60103: output = -RSQRT_PI<T>() * sqrt(273.)*30./16.;        break; // (6, 1, 3)
        case  60105: output =  RSQRT_PI<T>() * sqrt(273.)*33./16.;        break; // (6, 1, 5)
        case  60200: output =  RSQRT_PI<T>() * sqrt(2730.)/64.;           break; // (6, 2, 0)
        case  60202: output = -RSQRT_PI<T>() * sqrt(2730.)*18./64.;       break; // (6, 2, 2)
        case  60204: output =  RSQRT_PI<T>() * sqrt(2730.)*33./64.;       break; // (6, 2, 4)
        case  60301: output = -RSQRT_PI<T>() * sqrt(2730.)*3./32.;        break; // (6, 3, 1)
        case  60303: output =  RSQRT_PI<T>() * sqrt(2730.)*11./32.;       break; // (6, 3, 3)
        case  60400: output = -RSQRT_PI<T>() * sqrt(91.)*3./32.;          break; // (6, 4, 0)
        case  60402: output =  RSQRT_PI<T>() * sqrt(91.)*33./32.;         break; // (6, 4, 2)
        case  60501: output =  RSQRT_PI<T>() * sqrt(2002.)*3./32.;        break; // (6, 5, 1)
        case  60600: output =  RSQRT_PI<T>() * sqrt(6006.)/64.;           break; // (6, 6, 0)
        // L = 7
        case  70001: output = -RSQRT_PI<T>() * sqrt(15.)*35./32.;         break; // (7, 0, 1)
        case  70003: output =  RSQRT_PI<T>() * sqrt(15.)*315./32.;        break; // (7, 0, 3)
        case  70005: output = -RSQRT_PI<T>() * sqrt(15.)*693./32.;        break; // (7, 0, 5)
        case  70007: output =  RSQRT_PI<T>() * sqrt(15.)*429./32.;        break; // (7, 0, 7)
        case  70100: output = -RSQRT_PI<T>() * sqrt(105.)*5./64.;         break; // (7, 1, 0)
        case  70102: output =  RSQRT_PI<T>() * sqrt(105.)*135./64.;       break; // (7, 1, 2)
        case  70104: output = -RSQRT_PI<T>() * sqrt(105.)*495./64.;       break; // (7, 1, 4)
        case  70106: output =  RSQRT_PI<T>() * sqrt(105.)*429./64.;       break; // (7, 1, 6)
        case  70201: output =  RSQRT_PI<T>() * sqrt(70.)*45./64.;         break; // (7, 2, 1)
        case  70203: output = -RSQRT_PI<T>() * sqrt(70.)*330./64.;        break; // (7, 2, 3)
        case  70205: output =  RSQRT_PI<T>() * sqrt(70.)*429./64.;        break; // (7, 2, 5)
        case  70300: output =  RSQRT_PI<T>() * sqrt(35.)*9./64.;          break; // (7, 3, 0)
        case  70302: output = -RSQRT_PI<T>() * sqrt(35.)*198./64.;        break; // (7, 3, 2)
        case  70304: output =  RSQRT_PI<T>() * sqrt(35.)*429./64.;        break; // (7, 3, 4)
        case  70401: output = -RSQRT_PI<T>() * sqrt(385.)*9./32.;         break; // (7, 4, 1)
        case  70403: output =  RSQRT_PI<T>() * sqrt(385.)*39./32.;        break; // (7, 4, 3)
        case  70500: output = -RSQRT_PI<T>() * sqrt(385.)*3./64.;         break; // (7, 5, 0)
        case  70502: output =  RSQRT_PI<T>() * sqrt(385.)*39./64.;        break; // (7, 5, 2)
        case  70601: output =  RSQRT_PI<T>() * sqrt(10010.)*3./64.;       break; // (7, 6, 1)
        case  70700: output =  RSQRT_PI<T>() * sqrt(715.)*3./64.;         break; // (7, 7, 0)
        // L = 8
        case  80000: output =  RSQRT_PI<T>() * sqrt(17.)*35./256.;        break; // (8, 0, 0)
        case  80002: output = -RSQRT_PI<T>() * sqrt(17.)*1260./256.;      break; // (8, 0, 2)
        case  80004: output =  RSQRT_PI<T>() * sqrt(17.)*6930./256.;      break; // (8, 0, 4)
        case  80006: output = -RSQRT_PI<T>() * sqrt(17.)*12012./256.;     break; // (8, 0, 6)
        case  80008: output =  RSQRT_PI<T>() * sqrt(17.)*6435./256.;      break; // (8, 0, 8)
        case  80101: output = -RSQRT_PI<T>() * sqrt(17.)*105./64.;        break; // (8, 1, 1)
        case  80103: output =  RSQRT_PI<T>() * sqrt(17.)*1155./64.;       break; // (8, 1, 3)
        case  80105: output = -RSQRT_PI<T>() * sqrt(17.)*3003./64.;       break; // (8, 1, 5)
        case  80107: output =  RSQRT_PI<T>() * sqrt(17.)*2145./64.;       break; // (8, 1, 7)
        case  80200: output = -RSQRT_PI<T>() * sqrt(1190.)*3./128.;       break; // (8, 2, 0)
        case  80202: output =  RSQRT_PI<T>() * sqrt(1190.)*99./128.;      break; // (8, 2, 2)
        case  80204: output = -RSQRT_PI<T>() * sqrt(1190.)*429./128.;     break; // (8, 2, 4)
        case  80206: output =  RSQRT_PI<T>() * sqrt(1190.)*429./128.;     break; // (8, 2, 6)
        case  80301: output =  RSQRT_PI<T>() * sqrt(19635.)*3./64.;       break; // (8, 3, 1)
        case  80303: output = -RSQRT_PI<T>() * sqrt(19635.)*26./64.;      break; // (8, 3, 3)
        case  80305: output =  RSQRT_PI<T>() * sqrt(19635.)*39./64.;      break; // (8, 3, 5)
        case  80400: output =  RSQRT_PI<T>() * sqrt(1309.)*3./128.;       break; // (8, 4, 0)
        case  80402: output = -RSQRT_PI<T>() * sqrt(1309.)*78./128.;      break; // (8, 4, 2)
        case  80404: output =  RSQRT_PI<T>() * sqrt(1309.)*195./128.;     break; // (8, 4, 4)
        case  80501: output = -RSQRT_PI<T>() * sqrt(17017.)*3./64.;       break; // (8, 5, 1)
        case  80503: output =  RSQRT_PI<T>() * sqrt(17017.)*15./64.;      break; // (8, 5, 3)
        case  80600: output = -RSQRT_PI<T>() * sqrt(14586.)/128.;         break; // (8, 6, 0)
        case  80602: output =  RSQRT_PI<T>() * sqrt(14586.)*15./128.;     break; // (8, 6, 2)
        case  80701: output =  RSQRT_PI<T>() * sqrt(12155.)*3./64.;       break; // (8, 7, 1)
        case  80800: output =  RSQRT_PI<T>() * sqrt(12155.)*3./256.;      break; // (8, 8, 0)
        // L = 9
        case  90001: output =  RSQRT_PI<T>() * sqrt(19.)*315./256.;       break; // (9, 0, 1)
        case  90003: output = -RSQRT_PI<T>() * sqrt(19.)*4620./256.;      break; // (9, 0, 3)
        case  90005: output =  RSQRT_PI<T>() * sqrt(19.)*18018./256.;     break; // (9, 0, 5)
        case  90007: output = -RSQRT_PI<T>() * sqrt(19.)*25740./256.;     break; // (9, 0, 7)
        case  90009: output =  RSQRT_PI<T>() * sqrt(19.)*12155./256.;     break; // (9, 0, 9)
        case  90100: output =  RSQRT_PI<T>() * sqrt(95.)*21./256.;        break; // (9, 1, 0)
        case  90102: output = -RSQRT_PI<T>() * sqrt(95.)*924./256.;       break; // (9, 1, 2)
        case  90104: output =  RSQRT_PI<T>() * sqrt(95.)*6006./256.;      break; // (9, 1, 4)
        case  90106: output = -RSQRT_PI<T>() * sqrt(95.)*12012./256.;     break; // (9, 1, 6)
        case  90108: output =  RSQRT_PI<T>() * sqrt(95.)*7293./256.;      break; // (9, 1, 8)
        case  90201: output = -RSQRT_PI<T>() * sqrt(2090.)*21./128.;      break; // (9, 2, 1)
        case  90203: output =  RSQRT_PI<T>() * sqrt(2090.)*273./128.;     break; // (9, 2, 3)
        case  90205: output = -RSQRT_PI<T>() * sqrt(2090.)*819./128.;     break; // (9, 2, 5)
        case  90207: output =  RSQRT_PI<T>() * sqrt(2090.)*663./128.;     break; // (9, 2, 7)
        case  90300: output = -RSQRT_PI<T>() * sqrt(43890.)/256.;         break; // (9, 3, 0)
        case  90302: output =  RSQRT_PI<T>() * sqrt(43890.)*39./256.;     break; // (9, 3, 2)
        case  90304: output = -RSQRT_PI<T>() * sqrt(43890.)*195./256.;    break; // (9, 3, 4)
        case  90306: output =  RSQRT_PI<T>() * sqrt(43890.)*221./256.;    break; // (9, 3, 6)
        case  90401: output =  RSQRT_PI<T>() * sqrt(95095.)*3./128.;      break; // (9, 4, 1)
        case  90403: output = -RSQRT_PI<T>() * sqrt(95095.)*30./128.;     break; // (9, 4, 3)
        case  90405: output =  RSQRT_PI<T>() * sqrt(95095.)*51./128.;     break; // (9, 4, 5)
        case  90500: output =  RSQRT_PI<T>() * sqrt(5434.)*3./256.;       break; // (9, 5, 0)
        case  90502: output = -RSQRT_PI<T>() * sqrt(5434.)*90./256.;      break; // (9, 5, 2)
        case  90504: output =  RSQRT_PI<T>() * sqrt(5434.)*255./256.;     break; // (9, 5, 4)
        case  90601: output = -RSQRT_PI<T>() * sqrt(81510.)*3./128.;      break; // (9, 6, 1)
        case  90603: output =  RSQRT_PI<T>() * sqrt(81510.)*17./128.;     break; // (9, 6, 3)
        case  90700: output = -RSQRT_PI<T>() * sqrt(27170.)*3./512.;      break; // (9, 7, 0)
        case  90702: output =  RSQRT_PI<T>() * sqrt(27170.)*51./512.;     break; // (9, 7, 2)
        case  90801: output =  RSQRT_PI<T>() * sqrt(230945.)*3./256.;     break; // (9, 8, 1)
        case  90900: output =  RSQRT_PI<T>() * sqrt(461890.)/512.;        break; // (9, 9, 0)
        // L = 10
        case 100000: output = -RSQRT_PI<T>() * sqrt(21.)*63./512.;        break; // (10, 0, 0)
        case 100002: output =  RSQRT_PI<T>() * sqrt(21.)*3465./512.;      break; // (10, 0, 2)
        case 100004: output = -RSQRT_PI<T>() * sqrt(21.)*30030./512.;     break; // (10, 0, 4)
        case 100006: output =  RSQRT_PI<T>() * sqrt(21.)*90090./512.;     break; // (10, 0, 6)
        case 100008: output = -RSQRT_PI<T>() * sqrt(21.)*109395./512.;    break; // (10, 0, 8)
        case 100010: output =  RSQRT_PI<T>() * sqrt(21.)*46189./512.;     break; // (10, 0, 10)
        case 100101: output =  RSQRT_PI<T>() * sqrt(1155.)*63./256.;      break; // (10, 1, 1)
        case 100103: output = -RSQRT_PI<T>() * sqrt(1155.)*1092./256.;    break; // (10, 1, 3)
        case 100105: output =  RSQRT_PI<T>() * sqrt(1155.)*4914./256.;    break; // (10, 1, 5)
        case 100107: output = -RSQRT_PI<T>() * sqrt(1155.)*7956./256.;    break; // (10, 1, 7)
        case 100109: output =  RSQRT_PI<T>() * sqrt(1155.)*4199./256.;    break; // (10, 1, 9)
        case 100200: output =  RSQRT_PI<T>() * sqrt(385.)*21./512.;       break; // (10, 2, 0)
        case 100202: output = -RSQRT_PI<T>() * sqrt(385.)*1092./512.;     break; // (10, 2, 2)
        case 100204: output =  RSQRT_PI<T>() * sqrt(385.)*8190./512.;     break; // (10, 2, 4)
        case 100206: output = -RSQRT_PI<T>() * sqrt(385.)*18564./512.;    break; // (10, 2, 6)
        case 100208: output =  RSQRT_PI<T>() * sqrt(385.)*12597./512.;    break; // (10, 2, 8)
        case 100301: output = -RSQRT_PI<T>() * sqrt(10010.)*21./256.;     break; // (10, 3, 1)
        case 100303: output =  RSQRT_PI<T>() * sqrt(10010.)*315./256.;    break; // (10, 3, 3)
        case 100305: output = -RSQRT_PI<T>() * sqrt(10010.)*1071./256.;   break; // (10, 3, 5)
        case 100307: output =  RSQRT_PI<T>() * sqrt(10010.)*969./256.;    break; // (10, 3, 7)
        case 100400: output = -RSQRT_PI<T>() * sqrt(5005.)*3./256.;       break; // (10, 4, 0)
        case 100402: output =  RSQRT_PI<T>() * sqrt(5005.)*135./256.;     break; // (10, 4, 2)
        case 100404: output = -RSQRT_PI<T>() * sqrt(5005.)*765./256.;     break; // (10, 4, 4)
        case 100406: output =  RSQRT_PI<T>() * sqrt(5005.)*969./256.;     break; // (10, 4, 6)
        case 100501: output =  RSQRT_PI<T>() * sqrt(2002.)*45./256.;      break; // (10, 5, 1)
        case 100503: output = -RSQRT_PI<T>() * sqrt(2002.)*510./256.;     break; // (10, 5, 3)
        case 100505: output =  RSQRT_PI<T>() * sqrt(2002.)*969./256.;     break; // (10, 5, 5)
        case 100600: output =  RSQRT_PI<T>() * sqrt(10010.)*9./1024.;     break; // (10, 6, 0)
        case 100602: output = -RSQRT_PI<T>() * sqrt(10010.)*306./1024.;   break; // (10, 6, 2)
        case 100604: output =  RSQRT_PI<T>() * sqrt(10010.)*969./1024.;   break; // (10, 6, 4)
        case 100701: output = -RSQRT_PI<T>() * sqrt(170170.)*9./512.;     break; // (10, 7, 1)
        case 100703: output =  RSQRT_PI<T>() * sqrt(170170.)*57./512.;    break; // (10, 7, 3)
        case 100800: output = -RSQRT_PI<T>() * sqrt(255255.)/512.;        break; // (10, 8, 0)
        case 100802: output =  RSQRT_PI<T>() * sqrt(255255.)*19./512.;    break; // (10, 8, 2)
        case 100901: output =  RSQRT_PI<T>() * sqrt(9699690.)/512.;       break; // (10, 9, 1)
        case 101000: output =  RSQRT_PI<T>() * sqrt(1939938.)/1024.;      break; // (10, 10, 0)
    }
    return output;
}


/*
    Handle special case of xyz = (0., 0., 0.) with additional multiplier, either 0. or 1.
*/
template<typename T> __device__ __forceinline__ T nonzero(const T x, const T y, const T z) { return (T) (x != 0. || y != 0. || z != 0.); }


/*
    Compressed sin^m(theta)*[exp^(-i*m*phi) - exp^(i*m*phi)] and sin^m(theta)*[exp^(-i*m*phi) + exp^(i*m*phi)].
    These are shared multipliers for multiple L.

    __forceinline__ forces body of the function to be substituted in the place of the call.
    It proportionally enlarges executable size, but on the other hand saves time otherwise required to resolve function call.
*/
template<typename T, int M, char part> __device__ __forceinline__ T f_phi(const T x, const T y) {
    T output;
    if (part == 'r') {
        switch (M) {
            case 0:  {                                                output = 1.;                                                               break; }
            case 1:  {                                                output = x;                                                                break; }
            case 2:  {                                                output = (x + y) * (x - y);                                                break; }
            case 3:  {                                                output = x * (x*x - 3.*y*y);                                               break; }
            case 4:  { const T x2 = x*x, y2 = y*y;                    output = x2 * (x2 - 6.*y2) + y2*y2;                                        break; }
            case 5:  { const T x2 = x*x, y2 = y*y;                    output = x * (x2*x2 + 5.*y2 * (y2 - 2.*x2));                               break; }
            case 6:  { const T x2 = x*x, y2 = y*y, dx2y2 = x2 - y2;   output = dx2y2 * (dx2y2*dx2y2 - 12.*x2*y2);                                break; }
            case 7:  { const T x2 = x*x, y2 = y*y;                    output = x * (x2*x2 * (x2 - 21.*y2) + 7.*y2*y2 * (5.*x2 - y2));            break; }
            case 8:  { const T x2 = x*x, y2 = y*y, dx2y2 = x2 - y2;
                       const T dx2y2_sq = dx2y2*dx2y2;                output = dx2y2_sq*dx2y2_sq + x2*y2 * (16.*x2*y2 - 24.*dx2y2_sq);           break; }
            case 9:  { const T x2 = x*x, y2 = y*y;
                       const T x4 = x2*x2, y4 = y2*y2;                output = x * (x4*x4 + 126.*x4*y4 + 9.*y4*y4 - x2*y2 * (36.*x4 + 84*y4));   break; }
            case 10: { const T x2 = x*x, y2 = y*y, dx2y2 = x2 - y2;
                       const T dx2y2_sq = dx2y2*dx2y2;                output = dx2y2 * (dx2y2_sq*dx2y2_sq + 40.*x2*y2 * (2.*x2*y2 - dx2y2_sq));  break; }
        }
    }
    else { /* part == 'i' */
        switch (M) {
            case 0:  {                                                output = 0.;                                                               break; }
            case 1:  {                                                output = y;                                                                break; }
            case 2:  {                                                output = x * y * 2.;                                                       break; }
            case 3:  {                                                output = y * (3.*x*x - y*y);                                               break; }
            case 4:  {                                                output = x * y * (x + y) * (x - y) * 4.;                                   break; }
            case 5:  { const T x2 = x*x, y2 = y*y;                    output = y * (y2*y2 + 5.*x2 * (x2 - 2.*y2));                               break; }
            case 6:  { const T x2 = x*x, y2 = y*y;                    output = x * y * (3.*x2 - y2) * (x2 - 3.*y2) * 2.;                         break; }
            case 7:  { const T x2 = x*x, y2 = y*y;                    output = -y * (y2*y2 * (y2 - 21.*x2) + 7.*x2*x2 * (5.*y2 - x2));           break; }
            case 8:  { const T x2 = x*x, y2 = y*y, dx2y2 = x2 - y2;   output = x * y * dx2y2 * (dx2y2*dx2y2 - 4.*x2*y2) * 8.;                    break; }
            case 9:  { const T x2 = x*x, y2 = y*y;
                       const T x4 = x2*x2, y4 = y2*y2;                output = y * (y4*y4 + 126.*x4*y4 + 9.*x4*x4 - x2*y2 * (36.*y4 + 84*x4));   break; }
            case 10: { const T x2 = x*x, y2 = y*y, dx2y2 = x2 - y2;
                       const T dx2y2_sq = dx2y2*dx2y2;                output = x * y * (10.*dx2y2_sq * (dx2y2_sq - 8.*x2*y2) + 32.*x2*x2*y2*y2); break; }
        }
    }
    return output;
}


/*
    Polynomials in z.
*/
template<typename T, int L, int abs_M> __device__ __forceinline__ T p_z(const T z){
    T output;
    switch (L - abs_M) {
        case -1: {                                            output = 0.;                                                                break; }                     
        case 0:  { const T c0 = RSH_C<T, L, abs_M, 0>();      output = c0;                                                                break; }
        case 1:  { const T c1 = RSH_C<T, L, abs_M, 1>();      output = c1 * z;                                                            break; }
        case 2:  { const T c0 = RSH_C<T, L, abs_M, 0>();
                   const T c2 = RSH_C<T, L, abs_M, 2>();      output = c0 + c2 * z * z;                                                   break; }
        case 3:  { const T c1 = RSH_C<T, L, abs_M, 1>();
                   const T c3 = RSH_C<T, L, abs_M, 3>();      output = (c1 + c3 * z * z) * z;                                             break; }
        case 4:  { const T z2 = z*z;
                   const T c0 = RSH_C<T, L, abs_M, 0>();
                   const T c2 = RSH_C<T, L, abs_M, 2>();
                   const T c4 = RSH_C<T, L, abs_M, 4>();      output = c0 + (c2 + c4 * z2) * z2;                                          break; }
        case 5:  { const T z2 = z*z;
                   const T c1 = RSH_C<T, L, abs_M, 1>();
                   const T c3 = RSH_C<T, L, abs_M, 3>();
                   const T c5 = RSH_C<T, L, abs_M, 5>();      output = (c1 + (c3 + c5 * z2) * z2) * z;                                    break; }
        case 6:  { const T z2 = z*z;
                   const T c0 = RSH_C<T, L, abs_M, 0>();
                   const T c2 = RSH_C<T, L, abs_M, 2>();
                   const T c4 = RSH_C<T, L, abs_M, 4>();
                   const T c6 = RSH_C<T, L, abs_M, 6>();      output = c0 + (c2 + (c4 + c6 * z2) * z2) * z2;                              break; }
        case 7:  { const T z2 = z*z;
                   const T c1 = RSH_C<T, L, abs_M, 1>();
                   const T c3 = RSH_C<T, L, abs_M, 3>();
                   const T c5 = RSH_C<T, L, abs_M, 5>();
                   const T c7 = RSH_C<T, L, abs_M, 7>();      output = (c1 + (c3 + (c5 + c7 * z2) * z2) * z2) * z;                        break; }
        case 8:  { const T z2 = z*z;
                   const T c0 = RSH_C<T, L, abs_M, 0>();
                   const T c2 = RSH_C<T, L, abs_M, 2>();
                   const T c4 = RSH_C<T, L, abs_M, 4>();
                   const T c6 = RSH_C<T, L, abs_M, 6>();
                   const T c8 = RSH_C<T, L, abs_M, 8>();      output = c0 + (c2 + (c4 + (c6 + c8 * z2) * z2) * z2) * z2;                  break; }
        case 9:  { const T z2 = z*z;
                   const T c1 = RSH_C<T, L, abs_M, 1>();
                   const T c3 = RSH_C<T, L, abs_M, 3>();
                   const T c5 = RSH_C<T, L, abs_M, 5>();
                   const T c7 = RSH_C<T, L, abs_M, 7>();
                   const T c9 = RSH_C<T, L, abs_M, 9>();      output = (c1 + (c3 + (c5 + (c7 + c9 * z2) * z2) * z2) * z2) * z;            break; }
        case 10: { const T z2 = z*z;
                   const T c0 = RSH_C<T, L, abs_M, 0>();
                   const T c2 = RSH_C<T, L, abs_M, 2>();
                   const T c4 = RSH_C<T, L, abs_M, 4>();
                   const T c6 = RSH_C<T, L, abs_M, 6>();
                   const T c8 = RSH_C<T, L, abs_M, 8>();
                   const T c10 = RSH_C<T, L, abs_M, 10>();    output = c0 + (c2 + (c4 + (c6 + (c8 + c10 * z2) * z2) * z2) * z2) * z2;     break; }
    }
    return output;
}


/*
    Partial derivatives of normalized coordinates over Cartesian basis, e.g.: xn = x/r
    In regards to input arguments, throughout this file {x, y, z} correspond to normalized version.
*/
template<typename T, char axis> __device__ __forceinline__ T dxn_d(const T x, const T y, const T z) {
    T output;
    if (axis == 'x')       output = 1 - x*x;
    else if (axis == 'y')  output = -x*y;
    else /* axis == 'z' */ output = -x*z;
    return output;
}

template<typename T, char axis> __device__ __forceinline__ T dyn_d(const T x, const T y, const T z) {
    T output;
    if (axis == 'x')       output = -y*x;
    else if (axis == 'y')  output = 1 - y*y;
    else /* axis == 'z' */ output = -y*z;
    return output;
}

template<typename T, char axis> __device__ __forceinline__ T dzn_d(const T x, const T y, const T z) {
    T output;
    if (axis == 'x')       output = -z*x;
    else if (axis == 'y')  output = -z*y;
    else /* axis == 'z' */ output = 1 - z*z;
    return output;
}


/*
    Extra coef TODO: doc
*/
template<typename T, int L, int M> __device__ constexpr T DRSH_C() {
    T output;
    if (M > 0)              output = sqrt(static_cast<T>((L-M)*(L+M+1)));
    else if (M < 0)         output = sqrt(static_cast<T>((L+M)*(L-M+1)));
    else { /* M == 0 */
        if (L%2 == 0)       output = sqrt(static_cast<T>((L+1)*(L/2)));
        else /* L%2 == 1 */ output = sqrt(static_cast<T>(L*((L+1)/2)));
    }
    return output;
}


/*
    Real spherical harmonics.
    Special case of (0., 0., 0.) handled via additional multiplier where necessary.
    Multiplication has been chosen over if-else to avoid branch divergence.
*/
template<typename T, int L, int M> __device__ T rsh(const T x, const T y, const T z) {
    T output;
    if (M > 0)      output = p_z<T, L,  M>(z) * f_phi<T,  M, 'r'>(x, y);
    else if (M < 0) output = p_z<T, L, -M>(z) * f_phi<T, -M, 'i'>(x, y);
    else /* M==0 */ {
                    output = p_z<T, L,  0>(z);
                    if (L > 0 && L%2 == 0) output *= nonzero<T>(x, y, z);
    }
    return output;
}


/*
    Partial derivatives of real spherical harmonics with respect to Cartesian coordinates. 
*/
template<typename T, int L, int M> __device__ void drsh(T* const __restrict__ output, const T x, const T y, const T z, const size_t entry_pos, const size_t n_entries) {
    if (M > 0) {
        const T mul_shared = M * p_z<T, L, M>(z);
        const T mul_xn =  f_phi<T, M - 1, 'r'>(x, y) * mul_shared;
        const T mul_yn = -f_phi<T, M - 1, 'i'>(x, y) * mul_shared;
        const T mul_zn = DRSH_C<T, L, M>() * p_z<T, L, M + 1>(z) * f_phi<T, M, 'r'>(x, y);
        output[entry_pos]               = mul_xn * dxn_d<T, 'x'>(x, y, z) + mul_yn * dyn_d<T, 'x'>(x, y, z) + mul_zn * dzn_d<T, 'x'>(x, y, z);
        output[n_entries + entry_pos]   = mul_xn * dxn_d<T, 'y'>(x, y, z) + mul_yn * dyn_d<T, 'y'>(x, y, z) + mul_zn * dzn_d<T, 'y'>(x, y, z);
        output[2*n_entries + entry_pos] = mul_xn * dxn_d<T, 'z'>(x, y, z) + mul_yn * dyn_d<T, 'z'>(x, y, z) + mul_zn * dzn_d<T, 'z'>(x, y, z);
    }
    else if (M < 0) {
        const T mul_shared = -M * p_z<T, L, -M>(z);
        const T mul_xn = f_phi<T, -M - 1, 'i'>(x, y) * mul_shared;
        const T mul_yn = f_phi<T, -M - 1, 'r'>(x, y) * mul_shared;
        const T mul_zn = DRSH_C<T, L, M>() * p_z<T, L, -M + 1>(z) * f_phi<T, -M, 'i'>(x, y);
        output[entry_pos]               = mul_xn * dxn_d<T, 'x'>(x, y, z) + mul_yn * dyn_d<T, 'x'>(x, y, z) + mul_zn * dzn_d<T, 'x'>(x, y, z);
        output[n_entries + entry_pos]   = mul_xn * dxn_d<T, 'y'>(x, y, z) + mul_yn * dyn_d<T, 'y'>(x, y, z) + mul_zn * dzn_d<T, 'y'>(x, y, z);
        output[2*n_entries + entry_pos] = mul_xn * dxn_d<T, 'z'>(x, y, z) + mul_yn * dyn_d<T, 'z'>(x, y, z) + mul_zn * dzn_d<T, 'z'>(x, y, z);
    }
    else { /* M == 0 */
        const T mul_zn = DRSH_C<T, L, 0>() * p_z<T, L, 1>(z);
        output[entry_pos]               = mul_zn * dzn_d<T, 'x'>(x, y, z);
        output[n_entries + entry_pos]   = mul_zn * dzn_d<T, 'y'>(x, y, z);
        output[2*n_entries + entry_pos] = mul_zn * dzn_d<T, 'z'>(x, y, z);
    }
}


/*
Array of pointers to functions stored to "constant memory" (__constant__) on GPU - common for all blocks.
                                                    0
                                               1,   2,   3
                                          4,   5,   6,   7,   8
                                     9,  10,  11,  12,  13,  14,  15
                               16,  17,  18,  19,  20,  21,  22,  23,  24
                          25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35
                     36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48
                49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63
           64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80
      81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95.  96,  97,  98,  99
100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120
*/
template<typename T> __constant__ T (*const rsh_fptr[]) (const T, const T, const T) = {
                                                                                                                                             rsh<T, 0,0>,
                                                                                                                               rsh<T, 1,-1>, rsh<T, 1,0>, rsh<T, 1,1>,
                                                                                                                 rsh<T, 2,-2>, rsh<T, 2,-1>, rsh<T, 2,0>, rsh<T, 2,1>, rsh<T, 2,2>,
                                                                                                   rsh<T, 3,-3>, rsh<T, 3,-2>, rsh<T, 3,-1>, rsh<T, 3,0>, rsh<T, 3,1>, rsh<T, 3,2>, rsh<T, 3,3>,
                                                                                     rsh<T, 4,-4>, rsh<T, 4,-3>, rsh<T, 4,-2>, rsh<T, 4,-1>, rsh<T, 4,0>, rsh<T, 4,1>, rsh<T, 4,2>, rsh<T, 4,3>, rsh<T, 4,4>,
                                                                       rsh<T, 5,-5>, rsh<T, 5,-4>, rsh<T, 5,-3>, rsh<T, 5,-2>, rsh<T, 5,-1>, rsh<T, 5,0>, rsh<T, 5,1>, rsh<T, 5,2>, rsh<T, 5,3>, rsh<T, 5,4>, rsh<T, 5,5>,
                                                         rsh<T, 6,-6>, rsh<T, 6,-5>, rsh<T, 6,-4>, rsh<T, 6,-3>, rsh<T, 6,-2>, rsh<T, 6,-1>, rsh<T, 6,0>, rsh<T, 6,1>, rsh<T, 6,2>, rsh<T, 6,3>, rsh<T, 6,4>, rsh<T, 6,5>, rsh<T, 6,6>,
                                           rsh<T, 7,-7>, rsh<T, 7,-6>, rsh<T, 7,-5>, rsh<T, 7,-4>, rsh<T, 7,-3>, rsh<T, 7,-2>, rsh<T, 7,-1>, rsh<T, 7,0>, rsh<T, 7,1>, rsh<T, 7,2>, rsh<T, 7,3>, rsh<T, 7,4>, rsh<T, 7,5>, rsh<T, 7,6>, rsh<T, 7,7>,
                             rsh<T, 8,-8>, rsh<T, 8,-7>, rsh<T, 8,-6>, rsh<T, 8,-5>, rsh<T, 8,-4>, rsh<T, 8,-3>, rsh<T, 8,-2>, rsh<T, 8,-1>, rsh<T, 8,0>, rsh<T, 8,1>, rsh<T, 8,2>, rsh<T, 8,3>, rsh<T, 8,4>, rsh<T, 8,5>, rsh<T, 8,6>, rsh<T, 8,7>, rsh<T, 8,8>,
               rsh<T, 9,-9>, rsh<T, 9,-8>, rsh<T, 9,-7>, rsh<T, 9,-6>, rsh<T, 9,-5>, rsh<T, 9,-4>, rsh<T, 9,-3>, rsh<T, 9,-2>, rsh<T, 9,-1>, rsh<T, 9,0>, rsh<T, 9,1>, rsh<T, 9,2>, rsh<T, 9,3>, rsh<T, 9,4>, rsh<T, 9,5>, rsh<T, 9,6>, rsh<T, 9,7>, rsh<T, 9,8>, rsh<T, 9,9>,
rsh<T,10,-10>, rsh<T,10,-9>, rsh<T,10,-8>, rsh<T,10,-7>, rsh<T,10,-6>, rsh<T,10,-5>, rsh<T,10,-4>, rsh<T,10,-3>, rsh<T,10,-2>, rsh<T,10,-1>, rsh<T,10,0>, rsh<T,10,1>, rsh<T,10,2>, rsh<T,10,3>, rsh<T,10,4>, rsh<T,10,5>, rsh<T,10,6>, rsh<T,10,7>, rsh<T,10,8>, rsh<T,10,9>, rsh<T,10,10>
};


template<typename T> __constant__ void (*const drsh_fptr[]) (T* const __restrict__, const T, const T, const T, const size_t, const size_t) = {
                                                                                                                                                       drsh<T, 0,0>,
                                                                                                                                        drsh<T, 1,-1>, drsh<T, 1,0>, drsh<T, 1,1>,
                                                                                                                         drsh<T, 2,-2>, drsh<T, 2,-1>, drsh<T, 2,0>, drsh<T, 2,1>, drsh<T, 2,2>,
                                                                                                          drsh<T, 3,-3>, drsh<T, 3,-2>, drsh<T, 3,-1>, drsh<T, 3,0>, drsh<T, 3,1>, drsh<T, 3,2>, drsh<T, 3,3>,
                                                                                           drsh<T, 4,-4>, drsh<T, 4,-3>, drsh<T, 4,-2>, drsh<T, 4,-1>, drsh<T, 4,0>, drsh<T, 4,1>, drsh<T, 4,2>, drsh<T, 4,3>, drsh<T, 4,4>,
                                                                            drsh<T, 5,-5>, drsh<T, 5,-4>, drsh<T, 5,-3>, drsh<T, 5,-2>, drsh<T, 5,-1>, drsh<T, 5,0>, drsh<T, 5,1>, drsh<T, 5,2>, drsh<T, 5,3>, drsh<T, 5,4>, drsh<T, 5,5>,
                                                             drsh<T, 6,-6>, drsh<T, 6,-5>, drsh<T, 6,-4>, drsh<T, 6,-3>, drsh<T, 6,-2>, drsh<T, 6,-1>, drsh<T, 6,0>, drsh<T, 6,1>, drsh<T, 6,2>, drsh<T, 6,3>, drsh<T, 6,4>, drsh<T, 6,5>, drsh<T, 6,6>,
                                              drsh<T, 7,-7>, drsh<T, 7,-6>, drsh<T, 7,-5>, drsh<T, 7,-4>, drsh<T, 7,-3>, drsh<T, 7,-2>, drsh<T, 7,-1>, drsh<T, 7,0>, drsh<T, 7,1>, drsh<T, 7,2>, drsh<T, 7,3>, drsh<T, 7,4>, drsh<T, 7,5>, drsh<T, 7,6>, drsh<T, 7,7>,
                               drsh<T, 8,-8>, drsh<T, 8,-7>, drsh<T, 8,-6>, drsh<T, 8,-5>, drsh<T, 8,-4>, drsh<T, 8,-3>, drsh<T, 8,-2>, drsh<T, 8,-1>, drsh<T, 8,0>, drsh<T, 8,1>, drsh<T, 8,2>, drsh<T, 8,3>, drsh<T, 8,4>, drsh<T, 8,5>, drsh<T, 8,6>, drsh<T, 8,7>, drsh<T, 8,8>,
                drsh<T, 9,-9>, drsh<T, 9,-8>, drsh<T, 9,-7>, drsh<T, 9,-6>, drsh<T, 9,-5>, drsh<T, 9,-4>, drsh<T, 9,-3>, drsh<T, 9,-2>, drsh<T, 9,-1>, drsh<T, 9,0>, drsh<T, 9,1>, drsh<T, 9,2>, drsh<T, 9,3>, drsh<T, 9,4>, drsh<T, 9,5>, drsh<T, 9,6>, drsh<T, 9,7>, drsh<T, 9,8>, drsh<T, 9,9>,
drsh<T,10,-10>, drsh<T,10,-9>, drsh<T,10,-8>, drsh<T,10,-7>, drsh<T,10,-6>, drsh<T,10,-5>, drsh<T,10,-4>, drsh<T,10,-3>, drsh<T,10,-2>, drsh<T,10,-1>, drsh<T,10,0>, drsh<T,10,1>, drsh<T,10,2>, drsh<T,10,3>, drsh<T,10,4>, drsh<T,10,5>, drsh<T,10,6>, drsh<T,10,7>, drsh<T,10,8>, drsh<T,10,9>, drsh<T,10,10>
};



/*
    Preceding const means underlying data stays constant.
    Trailing const means that pointer to the data remains constant.
    __restrict__ makes a promise that underlying data can be accessed only with this pointer.
*/
template<typename T>
__global__ void rsh_cuda_kernel(T* const __restrict__ output, const T* const __restrict__ xyz, const size_t n_entries) {
	const size_t entry_pos = blockDim.x*blockIdx.x + threadIdx.x;                   // position of entry
	if (entry_pos >= n_entries) return;                                             // terminate early if out-of-bound - last warp (of threads) can be partially filled

	const T x = xyz[3*entry_pos];                                                   // "strided memory access" is generally not nice and severely drops throughput
	const T y = xyz[3*entry_pos+1];                                                 // padding to 4 and packing in double4 (single read transaction) showed no noticeable improvement (is scale to0 small measure?)
	const T z = xyz[3*entry_pos+2];                                                 // 100+ GB/s of throughput would be great, but even 3 GB/s does not make a bottleneck

    output[blockIdx.y*n_entries + entry_pos] = rsh_fptr<T>[blockIdx.y](x, y, z);    // select and apply function, store result to the "global memory"
}


template<typename T>
__global__ void drsh_cuda_kernel(T* const __restrict__ output, const T* const __restrict__ xyz, const size_t n_entries) {
    const size_t entry_pos = blockDim.x*blockIdx.x + threadIdx.x;
    if (entry_pos >= n_entries) return;

    const T x = xyz[3*entry_pos];
    const T y = xyz[3*entry_pos+1];
    const T z = xyz[3*entry_pos+2];

    drsh_fptr<T>[blockIdx.y](&(output[3*blockIdx.y*n_entries]), x, y, z, entry_pos, n_entries);
}


void rsh_cuda(
        torch::Tensor output,
        torch::Tensor xyz) {
    const size_t lm_size    = output.size(0);
    const size_t n_entries  = xyz.size(0);

    const size_t threads_per_block = 32;                                                    // warp size in contemporary GPUs is 32 threads, this variable should be a multiple of warp size
    dim3 numBlocks((n_entries + threads_per_block - 1)/threads_per_block, lm_size, 1);      // n_entries/threads_per_block is fractional in general case - round it up

    if (xyz.dtype() == torch::kFloat64) {
        rsh_cuda_kernel<double><<<numBlocks, threads_per_block>>>(
            (double*) output.data_ptr(), (const double*) xyz.data_ptr(), n_entries
        );
    }
    else {                                                                                  // check in C++ binding guarantees that data type is either double (float64) or float (float32)
        rsh_cuda_kernel<float><<<numBlocks, threads_per_block>>>(
            (float*) output.data_ptr(), (const float*) xyz.data_ptr(), n_entries
        );
    }
}


void drsh_cuda(
        torch::Tensor output,
        torch::Tensor xyz) {
    const size_t lm_size    = output.size(0);
    const size_t n_entries  = xyz.size(0);

    const size_t threads_per_block = 32;
    dim3 numBlocks((n_entries + threads_per_block - 1)/threads_per_block, lm_size, 1);

    if (xyz.dtype() == torch::kFloat64) {
        drsh_cuda_kernel<double><<<numBlocks, threads_per_block>>>(
            (double*) output.data_ptr(), (const double*) xyz.data_ptr(), n_entries
        );
    }
    else {
        drsh_cuda_kernel<float><<<numBlocks, threads_per_block>>>(
            (float*) output.data_ptr(), (const float*) xyz.data_ptr(), n_entries
        );
    }
}