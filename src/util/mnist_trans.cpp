#include "mnist_trans.h"
#include <iostream>
#include <bitset>

constexpr int BLOCK_SIZE = 8;
constexpr int BIN_WIDTH = 8;

uint64_t pack8x8(uint8_t *raw)
{
	uint64_t result = 0;
	for (int b = 0; b < BLOCK_SIZE; b++)
	{
		result |= ((uint64_t)raw[b]) << ((BLOCK_SIZE - b - 1) * BIN_WIDTH);
	}
	return result;
}

uint64_t transpose8(uint64_t x)
{
	uint64_t t = (x ^ (x >> 7)) & 0x00aa00aa00aa00aa;
	x = x ^ t ^ (t << 7);
	t = (x ^ (x >> 14)) & 0x0000cccc0000cccc;
	x = x ^ t ^ (t << 14);
	t = (x ^ (x >> 28)) & 0x00000000f0f0f0f0;
	x = x ^ t ^ (t << 28);
	return x;
}

/**
 * @brief 8x8ブロックごとにビット列を転置してBNN入力用のビット列を作成
 * 
 * @param raw 入力生データ
 * @param out BNN用データ列（出力）
 * @return true 変換成功
 * @return false 変換失敗（データ数が8の倍数でない）
 */
bool TransformBinToBNNInput(std::vector<uint8_t> &raw, std::vector<uint8_t> *out)
{
	if (raw.size() % BLOCK_SIZE != 0)
	{
		return false;
	}

	auto loopLast = raw.size() / BLOCK_SIZE;
	for (int i = 0; i < loopLast; i++)
	{
		uint64_t serialized = pack8x8(&raw[i*BLOCK_SIZE]);
		uint64_t t = transpose8(serialized);

		// 8bitごとに取り出して出力列に登録
		for (int b = 0; b < BIN_WIDTH; b++)
		{
			auto shift = (BIN_WIDTH - b - 1) * BIN_WIDTH;
			out[b].push_back((t >> shift) & 0xff);
		}
	}

	return true;
}