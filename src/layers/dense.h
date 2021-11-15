#ifndef DENSE_H_INCLUDED_
#define DENSE_H_INCLUDED_

#include <intrin.h>
#include "layer.h"

template <typename PrevLayer_t, int OutputSize>
class DenseLayer : LayerBase
{
public:
	static constexpr int kInputSize = PrevLayer_t::kOutputSize;
	static constexpr int kPaddedInputSize = PaddingSize(kInputSize);

	static constexpr int kOutputSize = OutputSize;
	static constexpr int kPaddedOutputSize = PaddingSize(kOutputSize);
	static constexpr int kNumPopcntCycles = AlignPopcntSize(kOutputSize);

	static constexpr int kSimdBlockNum = kPaddedInputSize / SIMD_BIT_WIDTH;

	static_assert(kInputSize % BIT_WIDTH == 0, "入力データはBIT_WIDTH単位でないといけない");
	static_assert(kOutputSize % BIT_WIDTH == 0, "出力データはBIT_WIDTH単位でないといけない");
	// static constexpr size_t kWeightCount = kInputSize * kOutputSize;

	virtual uint8_t *Forward(uint8_t *netInput)
	{
		uint8_t *input = PrevLayer_t::Forward(netInput);

		// TODO: SIMD化
		// TODO: パディング分のpop無視
		for (int i_out = 0; i_out < kPaddedOutputSize : ++i_out)
		{
			int sum = 0;
			for (int i_in = 0; i_in < kPaddedInputSize; ++i_in)
			{
				sum += __popcnt(input[i_in] ^ _weights[i_in]);
			}
			_outputBuffer[i_out] = 2 * sum - kInputSize;
		}
		return _outputBuffer;
	}

	virtual void Backward();

private:
	PrevLayer_t _prevLayer;
	uint8_t _outputBuffer[kPaddedOutputSize];
	uint8_t _weights[kPaddedInputSize];
};

#endif