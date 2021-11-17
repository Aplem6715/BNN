#ifndef INPUT_LAYER_H_INCLUDED_
#define INPUT_LAYER_H_INCLUDED_

#include "layer_base.h"

template <int InputSize>
class BitInputLayer
{
public:
#pragma region 出力サイズ関連
	static constexpr int kSettingOutDim = InputSize;
	static constexpr int kSettingOutBytes = BitToByteSize(kSettingOutDim);
	// SIMDパディング付き出力ビット幅
	static constexpr int kPaddedOutBits = AddPaddingBitSize(kSettingOutDim);
	// SIMDパディング付き出力バイト数
	static constexpr int kPaddedOutBytes = BitToByteSize(kPaddedOutBits);
#pragma endregion

	BitType *Forward(const uint8_t *netInput)
	{
		for (int i = 0; i < kSettingOutBytes; i++)
		{
			_outputBuffer[i] = netInput[i];
		}
		return _outputBuffer;
	}

	void ResetWeight() {}

#pragma region Train
	BitType *BatchForward(const uint8_t *netInput)
	{
		for (int batch = 0; batch < BATCH_SIZE; ++batch)
		{
			BitType *outputBuffer = &_outputBatchBuffer[batch * kSettingOutBytes];
			for (int i = 0; i < kSettingOutBytes; ++i)
			{
				outputBuffer[i] = netInput[batch * kSettingOutBytes + i];
			}
		}
		return _outputBatchBuffer;
	}

	void BatchBackward(const double *nextLayerGrads)
	{
		// 終端
	}
#pragma endregion

private:
	// 順伝播 出力バッファ(出力層はint)
	BitType _outputBuffer[kPaddedOutBytes] = {0};

#pragma region Train
	BitType _outputBatchBuffer[BATCH_SIZE * kPaddedOutBytes] = {0};
#pragma endregion
};

#endif