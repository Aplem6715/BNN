#ifndef OUTPUT_LAYER_H_INCLUDED_
#define OUTPUT_LAYER_H_INCLUDED_

#include "../util/random_real.h"
#include "layer_base.h"


template <typename PrevLayer_t, int OutputSize>
class OutputLayer {
   public:
    static constexpr int kInputBitSize       = PrevLayer_t::kOutputSize;
	static constexpr int kInputByteSize = std::ceil(kInputBitSize / (float)BIT_WIDTH);
	static constexpr int kPaddedInputBitSize = AddPaddingBitSize(kInputBitSize);
	static constexpr int kPaddingBitSize = kPaddedInputBitSize - kInputBitSize;
	static constexpr int kPaddedInputByteSize = std::ceil(kPaddedInputBitSize / (float)BIT_WIDTH);

	static constexpr int kOutputSize = OutputSize;

	static constexpr int kSimdBlockNum = std::ceil(kPaddedInputBitSize / (float)SIMD_BIT_WIDTH);

	// static constexpr size_t kWeightCount = kInputSize * kOutputBitSize;

	const int *Forward(uint8_t *netInput)
	{
		const auto input = _prevLayer.Forward(netInput);

		// TODO バッチ対応
		for (int i_in = 0; i_in < kPaddedInputByteSize; ++i_in)
		{
			_batchInput[0][i_in] = input[i_in];
		}

		// TODO: SIMD化
		// TODO: パディング分のpop無視
		for (int i_out = 0; i_out < kOutputSize; ++i_out)
		{
			int sum = 0;
			for (int i_in = 0; i_in < kPaddedInputByteSize; ++i_in)
			{
				sum += __popcnt(~(input[i_in] ^ _weights[i_in])); // ベクトル内積
			}

			_outputBuffer[i_out] =
				2 * sum -
				kInputBitSize - kPaddingBitSize; // 分布を0中心にシフト(0~2n を -n<0<nに)
		}
		return _outputBuffer;
	}

	void Backward(const double *nextGrads)
	{
		for (int j = 0; j < kInputBitSize; ++j)
		{
			_grads[j] = 0;
		}
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

			_grads[j] += sum;
		}
		// }

		for (int j = 0; j < kInputBitSize; ++j)
		{
			int bit_block = j / BIT_WIDTH;
			int bit_shift = BIT_WIDTH - (j % BIT_WIDTH) - 1;
			int inputBit = ((_batchInput[0][bit_block] >> bit_shift) & 1) == 0 ? -1 : 1;
			_real_weights[j] -= _grads[j] * inputBit;
			_real_weights[j] = std::max(-1.0, std::min(1.0, _real_weights[j]));

			uint8_t newBit = rnd_prob01(mt) < ((_real_weights[j] + 1) / 2);

			uint8_t mask = ~(1 << bit_shift);
			_weights[bit_block] = (_weights[bit_block] & mask) | (newBit << bit_shift);
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
			int bit_block = i / BIT_WIDTH;
			int bit_shift = BIT_WIDTH - (i % BIT_WIDTH) - 1;
			_weights[bit_block] |= rnd() % 2 << bit_shift;
		}
	}

   private:
    // 前のレイヤー
    PrevLayer_t _prevLayer;
    // 順伝播 重み
    uint8_t _weights[kPaddedInputByteSize] = {0};
    // 順伝播 出力バッファ(出力層はint)
    int _outputBuffer[kOutputSize] = {0};

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