#ifndef DENSE_H_INCLUDED_
#define DENSE_H_INCLUDED_

#include <intrin.h>
#include <cassert>
#include "layer_base.h"

template <typename PrevLayer_t, int OutputSize>
class DenseLayer : LayerBase<uint8_t>
{
public:
	static constexpr int kInputBitSize = PrevLayer_t::kOutputSize;
	static constexpr int kPaddedInputBitSize = AddPaddingBitSize(kInputBitSize);
	static constexpr int kPaddedInputByteSize = kPaddedInputBitSize / BIT_WIDTH;

	static constexpr int kOutputSize = OutputSize;
	static constexpr int kOutputBitSize = OutputSize;
	static constexpr int kPaddedOutputSize = AlignPopcntBitSize(kOutputBitSize);
	static constexpr int kOutputByteSize = kPaddedOutputSize / BIT_WIDTH;

	static constexpr int kSimdBlockNum = kPaddedInputBitSize / SIMD_BIT_WIDTH;

	static_assert(kInputBitSize % BIT_WIDTH == 0, "入力データはBIT_WIDTH単位でないといけない");
	static_assert(kOutputBitSize % BIT_WIDTH == 0, "出力データはBIT_WIDTH単位でないといけない");
	// static constexpr size_t kWeightCount = kInputSize * kOutputBitSize;

	virtual uint8_t *Forward(uint8_t *netInput);

	virtual void Backward(const double *nextGrads);

private:
	// 前のレイヤー
	PrevLayer_t _prevLayer;
	// 順伝播 出力バッファ
	uint8_t _outputBuffer[kOutputByteSize];
	// 順伝播 重み
	uint8_t _weights[kPaddedInputByteSize];

#pragma region Train
	// 前のレイヤーから受け取った入力値の履歴
	uint8_t _batchInput[BATCH_SIZE][kPaddedInputByteSize];
	// 逆伝播 重み更新用の勾配
	double _grads[kInputBitSize];
#pragma endregion
};

#endif