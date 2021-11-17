#ifndef AFFINE_H_INCLUDED_
#define AFFINE_H_INCLUDED_

#include <intrin.h>
#include "../net_common.h"
#include "../util/random_real.h"

template <typename PrevLayer_t, int OutputSize>
class AffineLayer {
public:
#pragma region 出力サイズ関連
    static constexpr int kSettingOutDim = OutputSize;
	// // SIMDパディング付き出力ビット幅
	// static constexpr int kPaddedOutBitWidth = AddPaddingBitSize(kSettingOutDim);
	// // SIMDパディング付き出力バイト数
	// static constexpr int kPaddedOutBytes = BitToByteSize(kPaddedOutBitWidth);
#pragma endregion

#pragma region 入力サイズ関連(前の層の値を引き継ぐ)
	// 入力次元数
	static constexpr int kSettingInDim = PrevLayer_t::kSettingOutDim;
	// 入力ビット幅（前の層のニューロン数）
	static constexpr int kSettingInBits = PrevLayer_t::kSettingOutDim;
	// 入力バイト数
	static constexpr int kSettingInBlocks = BitToByteSize(kSettingInBits);
	// SIMDパディング付き入力ビット幅
	static constexpr int kPaddedInBits =
		AddPaddingBitSize(kSettingInBits);
	// SIMDパディング付き入力バイト数
	static constexpr int kPaddedInBlocks = BitToByteSize(kPaddedInBits);

	static constexpr int kPaddingInBits = kPaddedInBits - kSettingInBits;
#pragma endregion

	const int *Forward(const uint8_t *netInput)
	{
		const BitType *input = _prevLayer.Forward(netInput);
		// TODO
		return _outputBuffer;
	}

#pragma region Train
	double **BatchForward(const uint8_t **netInput)
	{
		_inputBufferPtr = _prevLayer.BatchForward(netInput);

		for (int b = 0; b < BATCH_SIZE; ++b)
		{
			const BitType *batchInput = _inputBufferPtr[b];
			double *batchOutput = _outputBatchBuffer[b];
			for (int i_out = 0; i_out < kSettingOutDim; ++i_out)
			{
				// パディング分も含めて±1積和演算
				int pop = 0;
				for (int block = 0; block < kPaddedInBlocks; ++block)
				{
					uint8_t xnor = ~(batchInput[block] ^ _weight[i_out][block]);
					pop += __popcnt64(xnor); // 8bitで十分だけど後々256bit演算になるので
				}

				// 学習時は確率論的Activationを行うので，立っているビットの割合を出力
				batchOutput[i_out] = (pop - kPaddingInBits) / (double)kSettingInDim;
			}
		}

		return _outputBatchBuffer;
	}

	void BatchBackward(const double **nextLayerGrads)
	{
		// 勾配更新
		for (int batch = 0; batch < BATCH_SIZE; ++batch)
		{
			double *nextBatchGrad = nextLayerGrads[batch];
			for (int in = 0; in < kSettingInDim; ++in)
			{
				int block = GetBlockIndex(in);
				int shift = GetBitIndexInBlock(in);
				double sum = 0;
				for (int out = 0; out < kSettingOutDim; ++out)
				{
					uint8_t w_bit = (_weight[out][block] >> shift) & 0x1;
					sum += nextBatchGrad[out] * w_bit;
				}
				_grads[batch][in] = sum;
			}
		}

		// 重み調整幅を計算
		for (int batch = 0; batch < BATCH_SIZE; ++batch)
		{
			double *nextBatchGrad = nextLayerGrads[batch];
			const BitType *batchInput = _inputBufferPtr[batch];
			double **batchDiff = _weightDiff[batch];
			for (int out = 0; out < kSettingOutDim; out++)
			{
				for (int in = 0; in < kSettingInBits; in++)
				{
					int block = GetBlockIndex(in);
					int shift = GetBitIndexInBlock(in);
					uint8_t in_bit = (batchInput[block] >> shift) & 0x01;
					_realWeight[out][in] += nextBatchGrad[out] * in_bit;
				}
			}
		}

		// 重み更新
		for (int out = 0; out < kSettingOutDim; out++)
		{
			for (int in = 0; in < kSettingInBits; in++)
			{
				int block = GetBlockIndex(in);
				int shift = GetBitIndexInBlock(in);
				uint8_t newBit = _realWeight[out][in] > 0 ? 1 : 0;
				// 左から3ビット目の重みを更新するとき 元:11111101 -> masked:11011101
				uint8_t masked_w = _weight[block] & ~(1 << shift);
				// マスクで0になった位置に新しい重みbitを入れ込む masked:11011101 | shifted:00100000
				_weight[out][block] = masked_w | newBit << shift;
			}
		}

		_prevLayer.BatchBackward(_grads);
	}
#pragma endregion

private:
	PrevLayer_t _prevLayer;
	int _outputBuffer[kSettingOutDim];
	ByteType _weight[kSettingOutDim][kPaddedInBlocks] = {0};

#pragma region Train
	double _outputBatchBuffer[BATCH_SIZE][kSettingOutDim];
	BitType **_inputBufferPtr;
	double _realWeight[kSettingOutDim][kPaddedInBlocks] = {0};
	double _weightDiff[BATCH_SIZE][kSettingOutDim][kSettingInBits] = {0};
	double _grads[BATCH_SIZE][kSettingInDim] = {0};
#pragma endregion
};

#endif