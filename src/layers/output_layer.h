#ifndef OUTPUT_LAYER_H_INCLUDED_
#define OUTPUT_LAYER_H_INCLUDED_

#include "layer_base.h"

template <typename PrevLayer_t, int OutputSize>
class OutputLayer : LayerBase
{
public:
	static constexpr int kInputBitSize = PrevLayer_t::kOutputBitSize;
	static constexpr int kPaddedInputBitSize = AddPaddingBitSize(kInputBitSize);
	static constexpr int kPaddedInputByteSize = kPaddedInputBitSize / BIT_WIDTH;

	static constexpr int kOutputSize = OutputSize;

	static constexpr int kSimdBlockNum = kPaddedInputBitSize / SIMD_BIT_WIDTH;

	static_assert(kInputBitSize % BIT_WIDTH == 0, "入力データはBIT_WIDTH単位でないといけない");
	// static constexpr size_t kWeightCount = kInputSize * kOutputBitSize;

	virtual uint8_t *Forward(uint8_t *netInput);

	virtual void Backward(const double *nextGrads);

private:
	// 前のレイヤー
	PrevLayer_t _prevLayer;
	// 順伝播 重み
	uint8_t _weights[kPaddedInputByteSize];
	// 順伝播 出力バッファ(出力層はint)
	int _outputBuffer[kOutputSize];

#pragma region Train
	// 前のレイヤーから受け取った入力値の履歴
	uint8_t _batchInput[BATCH_SIZE][kPaddedInputByteSize];
	// 逆伝播 重み更新用の勾配
	double _grads[kInputBitSize];
#pragma endregion
};


#endif