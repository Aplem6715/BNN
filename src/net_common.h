#ifndef NET_COMMON_H_INCLUDED_
#define NET_COMMON_H_INCLUDED_

#include <cmath>
#include <cstdint>

constexpr int BIT_WIDTH = 8;
constexpr int SIMD_BIT_WIDTH = 256; // AVX2
constexpr int POPCNT_BIT_WIDTH = 64;

constexpr int PaddingSize(int size)
{
	return std::ceil(size / (double)SIMD_BIT_WIDTH) * SIMD_BIT_WIDTH;
}

constexpr int AlignPopcntSize(int size){
	return std::ceil(size / (double)POPCNT_BIT_WIDTH) * POPCNT_BIT_WIDTH;
}

#endif