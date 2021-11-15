#include "dense.h"

template <typename PrevLayer_t, int OutputSize>
uint8_t *DenseLayer<PrevLayer_t, OutputSize>::Forward(uint8_t *netInput)
{
	uint8_t *input = PrevLayer_t::Forward(netInput);

	// TODO: SIMD化
	// TODO: パディング分のpop無視
	for (int i_out = 0; i_out < kPaddedOutputSize; ++i_out)
	{
		int outBlock = i_out / BIT_WIDTH;
		int outShift = BIT_WIDTH - (i_out % BIT_WIDTH);
		int sum = 0;
		_outputBuffer[outBlock] = 0;
		for (int i_in = 0; i_in < kPaddedInputByteSize; ++i_in)
		{
			sum += __popcnt(input[i_in] ^ _weights[i_in]); // ベクトル内積
		}
		auto a = (2 * sum - kInputBitSize);	// 分布を0中心にシフト(0~2n を -n<0<nに)
		auto h = (a > 0); // Activation Function: Sign(x)
		assert((h == 0 || h == 1));

		_outputBuffer[outBlock] |= h << outShift;
	}
	return _outputBuffer;
}

template <typename PrevLayer_t, int OutputSize>
void DenseLayer<PrevLayer_t, OutputSize>::Backward(const double *nextGrads)
{
	for (int i_batch; i_batch < BATCH_SIZE; ++i_batch)
	{
		for (int j = 0; j < kInputBitSize; ++j)
		{
			double grad = 0;

			int bit_block = j / BIT_WIDTH;
			int bit_shift = BIT_WIDTH - (j % BIT_WIDTH);
			int w_bit = (_weights[bit_block] >> bit_shift) & 0x1;
			// ウェイトが0の時はsumは増加しないのでループそのものをスキップ
			if (w_bit == 1)
			{
				for (int i = 0; i < kOutputBitSize; i++)
				{
					//grad += nextGrad[i] * weight(ifブロック内でw_bitは必ず1)
					grad += nextGrads[i] * 1;
				}
			}

			_grads[i_batch][j] = grad;
		}
	}

	// 手前の層へ逆伝播
	_prevLayer.Backward(_grads);
}
