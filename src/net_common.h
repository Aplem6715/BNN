#ifndef NET_COMMON_H_INCLUDED_
#define NET_COMMON_H_INCLUDED_

#include <cmath>
#include <cstdint>

constexpr int BATCH_SIZE = 32;

constexpr int BIT_WIDTH = 8;
constexpr int SIMD_BIT_WIDTH = 256; // AVX2
constexpr int POPCNT_BIT_WIDTH = 64;

constexpr int AddPaddingBitSize(int bitSize)
{
	return std::ceil(bitSize / (double)SIMD_BIT_WIDTH) * SIMD_BIT_WIDTH;
}

constexpr int AlignPopcntBitSize(int bitSize){
	return std::ceil(bitSize / (double)POPCNT_BIT_WIDTH) * POPCNT_BIT_WIDTH;
}

#endif