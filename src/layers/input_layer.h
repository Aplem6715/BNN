#ifndef INPUT_LAYER_H_INCLUDED_
#define INPUT_LAYER_H_INCLUDED_

#include "layer_base.h"

template <int InputSize>
class InputLayer
{
public:
	static constexpr int kOutputSize = InputSize;
	static constexpr int kOutputBitSize = InputSize;
	static constexpr int kPaddedOutputSize = AddPaddingBitSize(kOutputBitSize);
	static constexpr int kOutputByteSize = std::ceil(kPaddedOutputSize / (float)BIT_WIDTH);

	static constexpr int kInputBitSize  = InputSize;
	static constexpr int kInputByteSize = std::ceil(kInputBitSize / (float)BIT_WIDTH);
	static constexpr int kPaddedInputBitSize = AddPaddingBitSize(kInputBitSize);

	static_assert(kPaddedOutputSize % 8 == 0);

	uint8_t *Forward(uint8_t *netInput)
	{
		for (int i = 0; i < kInputByteSize; i++)
		{
			_outputBuffer[i] = netInput[i];
		}
		return _outputBuffer;
	}

	void Backward(const double *nextGrads)
	{
		// 終端
	}

	void ResetWeights() {}

private:
	// 順伝播 出力バッファ(出力層はint)
	uint8_t _outputBuffer[kOutputByteSize] = {0};
};

#endif