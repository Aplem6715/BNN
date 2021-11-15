#include "output_layer.h"

template <typename PrevLayer_t, int OutputSize>
int *OutputLayer<PrevLayer_t, OutputSize>::Forward(uint8_t *netInput)
{
	uint8_t *input = PrevLayer_t::Forward(netInput);

	// TODO: SIMD化
	// TODO: パディング分のpop無視
	for (int i_out = 0; i_out < kOutputSize; ++i_out)
	{
		int sum = 0;
		_outputBuffer[i_out] = 0;
		for (int i_in = 0; i_in < kPaddedInputBitSize; ++i_in)
		{
			sum += __popcnt(input[i_in] ^ _weights[i_in]); // ベクトル内積
		}

		_outputBuffer[i_out] = 2 * sum - kInputBitSize; // 分布を0中心にシフト(0~2n を -n<0<nに)
	}
	return _outputBuffer;
}

template <typename PrevLayer_t, int OutputSize>
void OutputLayer<PrevLayer_t, OutputSize>::Backward(const double *nextGrads)
{
	for (int i_batch; i_batch < BATCH_SIZE; ++i_batch)
	{
		for (int j = 0; j < kInputBitSize; ++j)
		{
			double sum = 0;

			int bit_block = j / BIT_WIDTH;
			int bit_shift = BIT_WIDTH - (j % BIT_WIDTH);
			int w_bit = (_weights[bit_block] >> bit_shift) & 0x1;
			// ウェイトが0の時はsumは増加しないのでループそのものをスキップ
			if (w_bit == 1)
			{
				for (int i = 0; i < kOutputSize; i++)
				{
					// sum += nextGrad * weight
					sum += nextGrads[i] * 1 /*ifブロック内でw_bitは必ず1*/;
				}
			}

			_grads[i_batch][j] = sum;
		}
	}

	// 手前の層へ逆伝播
	_prevLayer.Backward(_grads);
}
