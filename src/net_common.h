#ifndef NET_COMMON_H_INCLUDED_
#define NET_COMMON_H_INCLUDED_

#include <cmath>
#include <cstdint>

typedef uint8_t BitType;
typedef uint8_t ByteType;
typedef int IntType;
typedef double RealType;

constexpr int BATCH_SIZE = 1;

constexpr int BIT_WIDTH = 8;
constexpr int SIMD_BIT_WIDTH = 256; // AVX2
constexpr int POPCNT_BIT_WIDTH = 64;

enum DataType
{
	DataType_Real,
	DataType_Bit,
	DataType_Byte,
};

constexpr int AddPaddingBitSize(int bitSize)
{
	return std::ceil(bitSize / (double)SIMD_BIT_WIDTH) * SIMD_BIT_WIDTH;
}

constexpr int BitToByteSize(int bitSize)
{
	return std::ceil(bitSize / (float)BIT_WIDTH);
}

inline int GetBlockIndex(int bitIndex)
{
	return bitIndex / BIT_WIDTH;
}

inline int GetBitIndexInBlock(int bitIndex)
{
	return BIT_WIDTH - (bitIndex % BIT_WIDTH) - 1;
}
// constexpr int AlignPopcntBitSize(int bitSize){
// 	return std::ceil(bitSize / (double)POPCNT_BIT_WIDTH) * POPCNT_BIT_WIDTH;
// }

#endif