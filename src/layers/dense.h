#ifndef DENSE_H_INCLUDED_
#define DENSE_H_INCLUDED_

#include <intrin.h>

#include <cassert>

#include "layer_base.h"
#include "../util/random_real.h"

template <typename PrevLayer_t, int OutputSize>
class DenseLayer {
   public:
    static constexpr int kInputBitSize       = PrevLayer_t::kOutputSize;
	static constexpr int kInputByteSize = std::ceil(kInputBitSize / (float)BIT_WIDTH);
	static constexpr int kPaddedInputBitSize = AddPaddingBitSize(kInputBitSize);
	static constexpr int kPaddedInputByteSize = std::ceil(kPaddedInputBitSize / (float)BIT_WIDTH);
	static constexpr int kPaddingBitSize = kPaddedInputBitSize - kInputBitSize;

	static constexpr int kOutputSize = OutputSize;
	static constexpr int kOutputBitSize = OutputSize;
	static constexpr int kPaddedOutputSize = AddPaddingBitSize(kOutputBitSize);
	static constexpr int kOutputByteSize = std::ceil(kPaddedOutputSize / (float)BIT_WIDTH);

	static constexpr int kSimdBlockNum = kPaddedInputBitSize / SIMD_BIT_WIDTH;

	static_assert(kOutputBitSize % BIT_WIDTH == 0,
				  "出力データはBIT_WIDTH単位でないといけない");
	// static constexpr size_t kWeightCount = kInputSize * kOutputBitSize;

	const uint8_t *Forward(uint8_t *netInput)
	{
		const auto input = _prevLayer.Forward(netInput);

		// TODO バッチ対応
		for (int i_in = 0; i_in < kPaddedInputByteSize; ++i_in)
		{
			_batchInput[0][i_in] = input[i_in];
		}

		for (int i = 0; i < kOutputByteSize; ++i)
		{
			_outputBuffer[i] = 0;
		}

		// TODO: SIMD化
		// TODO: パディング分のpop無視
		for (int i_out = 0; i_out < kPaddedOutputSize; ++i_out)
		{
			int outBlock = i_out / BIT_WIDTH;
			int outShift = BIT_WIDTH - (i_out % BIT_WIDTH) - 1;

			if (i_out >= kOutputBitSize)
			{
				// パディング分は全部0
				// _outputBuffer[outBlock] |= 0 << outShift;
				continue;
			}
			else
			{
				int sum = 0;
				for (int i_in = 0; i_in < kPaddedInputByteSize; ++i_in)
				{
					uint8_t xorI = input[i_in] ^ _weights[i_in];
					uint8_t xnorI = ~xorI;
					int p = __popcnt(xnorI);
					sum += p; // ベクトル内積
				}
				auto a = (2 * (sum - kPaddingBitSize) -
						  kInputBitSize); // 分布を0中心にシフト(0~2n を -n<0<nに)
				auto h = (a > 0);		  // Activation Function: Sign(x)
				assert((h == 0 || h == 1));

				_outputBuffer[outBlock] |= (uint8_t)h << outShift;
			}
		}
		return _outputBuffer;
	}

	void Backward(const double *nextGrads)
	{
		// for (int i_batch = 0; i_batch < BATCH_SIZE; ++i_batch)
		// {
		for (int j = 0; j < kInputBitSize; ++j)
		{
			double sum = 0;

			int bit_block = j / BIT_WIDTH;
			int bit_shift = BIT_WIDTH - (j % BIT_WIDTH) - 1;
			int w_bit = (_weights[bit_block] >> bit_shift);
			double w = (w_bit & 1) == 0 ? -1 : 1;
			for (int i = 0; i < kOutputSize; i++)
			{
				// sum += nextGrad * weight
				sum += nextGrads[i] * w /*ifブロック内でw_bitは必ず1*/;
			}

			_grads[j] = sum;
		}
		// }

		for (int i_out = 0; i_out < kOutputBitSize; ++i_out)
		{
			for (int j = 0; j < kInputBitSize; ++j)
			{
				int bit_block = j / BIT_WIDTH;
				int bit_shift = BIT_WIDTH - (j % BIT_WIDTH) - 1;
				int inputBit = ((_batchInput[0][bit_block] >> bit_shift) & 1) == 0 ? -1 : 1;
				_real_weights[j] -= nextGrads[i_out] * inputBit;
				_real_weights[j] = std::max(-1.0, std::min(1.0, _real_weights[j]));

				uint8_t newBit = rnd_prob01(mt) < ((_real_weights[j] + 1) / 2);

				uint8_t mask = ~(1 << bit_shift);
				_weights[bit_block] = (_weights[bit_block] & mask) | (newBit << bit_shift);
			}
		}

		// 手前の層へ逆伝播
		_prevLayer.Backward(_grads);
	}

	void ResetWeights()
	{
		_prevLayer.ResetWeights();
		for (int i = 0; i < kPaddedInputByteSize; i++)
		{
			_weights[i] = 0;
		}
		for (int i = 0; i < kInputBitSize; i++)
		{
			_real_weights[i] = 0;

			int bit_block = i / BIT_WIDTH;
			int bit_shift = BIT_WIDTH - (i % BIT_WIDTH) - 1;
			_weights[bit_block] |= rnd() % 2 << bit_shift;
		}
	}

   private:
    // 前のレイヤー
    PrevLayer_t _prevLayer;
    // 順伝播 出力バッファ
    uint8_t _outputBuffer[kOutputByteSize] = {0};
    // 順伝播 重み
    uint8_t _weights[kPaddedInputByteSize] = {0};

#pragma region Train
    // 前のレイヤーから受け取った入力値の履歴
    uint8_t _batchInput[BATCH_SIZE][kPaddedInputByteSize];
    // 逆伝播 重み更新用の勾配
    double _grads[kInputBitSize];
	// 逆伝播 重み更新用の実数ウェイト
	double _real_weights[kInputBitSize] = {0};
#pragma endregion
};

#endif