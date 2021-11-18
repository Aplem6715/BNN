#ifndef AFFINE_H_INCLUDED_
#define AFFINE_H_INCLUDED_

#include <intrin.h>
#include "../net_common.h"
#include "../util/random_real.h"

template <typename PrevLayer_t, int OutputSize>
class AffineLayer
{
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

private:
	PrevLayer_t _prevLayer;
	RealType _outputBuffer[kSettingOutDim];
	ByteType _weight[kSettingOutDim][kPaddedInBlocks] = {0};

#pragma region Train
	RealType _outputBatchBuffer[BATCH_SIZE * kSettingOutDim];
	BitType *_inputBufferPtr;
	double _realWeight[kSettingOutDim][kPaddedInBlocks] = {0};
	double _weightDiff[BATCH_SIZE][kSettingOutDim][kSettingInBits] = {0};
	double _grads[BATCH_SIZE * kSettingInDim] = {0};
#pragma endregion

public:
	const RealType *Forward(const uint8_t *netInput)
	{
		const BitType *input = _prevLayer.Forward(netInput);
		// TODO
		return _outputBuffer;
	}

	void ResetWeight()
	{
		for (int i_out = 0; i_out < kSettingOutDim; ++i_out)
		{
			for (int block = 0; block < kPaddedInBlocks; ++block)
			{
				_weight[i_out][block] = 0;
			}
		}

		for (int i_out = 0; i_out < kSettingOutDim; ++i_out)
		{
			for (int in = 0; in < kSettingInBits; ++in)
			{
				int block = GetBlockIndex(in);
				uint8_t shift = GetBitIndexInBlock(in);
				_realWeight[i_out][in] = GetRandReal() * 2 - 1;
				uint8_t bit = _realWeight[i_out][in] > 0 ? 1 : 0;
				uint8_t shifted = bit << shift;
				_weight[i_out][block] |= shifted;
			}
		}
		_prevLayer.ResetWeight();
	}

#pragma region Train
	RealType *BatchForward(const uint8_t *netInput)
	{
		_inputBufferPtr = _prevLayer.BatchForward(netInput);

		for (int b = 0; b < BATCH_SIZE; ++b)
		{
			const BitType *batchInput = &_inputBufferPtr[b * kPaddedInBlocks];
			RealType *batchOutput = &_outputBatchBuffer[b * kSettingOutDim];
			for (int i_out = 0; i_out < kSettingOutDim; ++i_out)
			{
				// パディング分も含めて±1積和演算
				int pop = 0;
				for (int block = 0; block < kPaddedInBlocks; ++block)
				{
					uint8_t xnor = ~(batchInput[block] ^ _weight[i_out][block]);
					pop += __popcnt64(xnor); // 8bitで十分だけど後々256bit演算になるので
				}

				batchOutput[i_out] = (2 * (pop - kPaddingInBits) - kSettingInDim);
			}
		}

		return _outputBatchBuffer;
	}

	void BatchBackward(const double *nextLayerGrads)
	{
		// 勾配更新
		for (int batch = 0; batch < BATCH_SIZE; ++batch)
		{
			const double *nextBatchGrad = &nextLayerGrads[batch * kSettingOutDim];
			double *grads = &_grads[batch * kSettingInDim];
			for (int in = 0; in < kSettingInDim; ++in)
			{
				int block = GetBlockIndex(in);
				uint8_t shift = GetBitIndexInBlock(in);
				double sum = 0;
				for (int out = 0; out < kSettingOutDim; ++out)
				{
					uint8_t w_bit = (_weight[out][block] >> shift) & 0x1;
					sum += nextBatchGrad[out] * (w_bit == 1 ? 1 : -1);
				}
				grads[in] = sum;
			}
		}

		for (int batch = 0; batch < BATCH_SIZE; ++batch)
		{
			const double *nextBatchGrad = &nextLayerGrads[batch * kSettingOutDim];
			const BitType *batchInput = &_inputBufferPtr[batch * kSettingInBits];

			// 重み調整幅を計算
			for (int out = 0; out < kSettingOutDim; out++)
			{
				for (int in = 0; in < kSettingInBits; in++)
				{
					int block = GetBlockIndex(in);
					uint8_t shift = GetBitIndexInBlock(in);
					uint8_t in_bit = (batchInput[block] >> shift) & 0x01;
					double realIn = (in_bit == 1) ? 1 : -1;
					_realWeight[out][in] += nextBatchGrad[out] * realIn;
				}
			}
		}

		// 重み更新
		for (int out = 0; out < kSettingOutDim; out++)
		{
			ByteType *weight = _weight[out];
			for (int in = 0; in < kSettingInBits; in++)
			{
				int block = GetBlockIndex(in);
				uint8_t shift = GetBitIndexInBlock(in);

				// Clipping
				double tmp_w = _realWeight[out][in];
				tmp_w = std::max(-1.0, std::min(1.0, tmp_w));
				_realWeight[out][in] = tmp_w;

				uint8_t newBit = tmp_w > 0 ? 1 : 0;
				newBit = (newBit << shift);
				// 左から3ビット目の重みを更新するとき 元:11111101 -> masked:11011101
				uint8_t masked_w = weight[block] & ~(1 << shift);

				// マスクで0になった位置に新しい重みbitを入れ込む masked:11011101 | shifted:00100000
				weight[block] = masked_w | newBit;
			}
		}

		_prevLayer.BatchBackward(_grads);
	}
#pragma endregion
};

#endif