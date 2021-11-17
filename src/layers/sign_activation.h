#ifndef SIGN_ACTIVATION_H_INCLUDED_
#define SIGN_ACTIVATION_H_INCLUDED_

#include <algorithm>
#include "../util/random_real.h"
#include "../net_common.h"

template <typename PrevLayer_t>
class SignActivation
{
public:
#pragma region 出力サイズ関連
	static constexpr int kSettingOutDim = PrevLayer_t::kSettingOutDim;
	// 出力ビット幅（ニューロン数）
	static constexpr int kSettingOutBits = PrevLayer_t::kSettingOutDim;
	// 出力バイト数
	static constexpr int kSettingOutBytes = BitToByteSize(kSettingOutBits);
#pragma endregion

#pragma region 入力サイズ関連(前の層の値を引き継ぐ)
	// 入力次元数
	static constexpr int kSettingInDim = PrevLayer_t::kSettingOutDim;
#pragma endregion

	const BitType *Forward(const uint8_t *netInput)
	{
		const IntType *input = _prevLayer.Forward(netInput);

		ClearOutBuffer();

		for (int i = 0; i < kSettingInDim; ++i)
		{
			int block = GetBlockIndex(i);
			int shift = GetBitIndexInBlock(i);
			uint8_t sign = input[i] > 0 ? 1 : 0;
			_outputBuffer[block] |= sign << shift;
		}
		return _outputBuffer;
	}

	void ResetWeight(){
		_prevLayer.ResetWeight();
	}

#pragma region Train
	BitType *BatchForward(const uint8_t *netInput)
	{
		const RealType* _inputBufferPtr = _prevLayer.BatchForward(netInput);

		ClearOutBatchBuffer();

		for (int b = 0; b < BATCH_SIZE; b++)
		{
			const double* batchInput = &_inputBufferPtr[b*kSettingInDim];
			BitType* batchOutput = &_outputBatchBuffer[b*kSettingInDim];
			for (int i = 0; i < kSettingInDim; ++i)
			{
				int block = GetBlockIndex(i);
				int shift = GetBitIndexInBlock(i);
				// 確率論的Activation
				double probPositive = batchInput[i];
				uint8_t sign = GetRandReal() < probPositive;
				batchOutput[block] |= sign << shift;
			}
		}
		return _outputBatchBuffer;
	}

	void BatchBackward(const double *nextLayerGrads)
	{
		for (int b = 0; b < BATCH_SIZE; ++b)
		{
			int batchShift = b * kSettingInDim;
			double *grads = &_grads[batchShift];
			for (int i = 0; i < kSettingInDim; ++i)
			{
				double x = nextLayerGrads[batchShift + i];
				// Hard-tanh (straight-through estimator)
				grads[i] = std::max(-1.0, std::min(1.0, x));
			}
		}
		_prevLayer.BatchBackward(_grads);
	}
#pragma endregion

private:
	PrevLayer_t _prevLayer;
	BitType _outputBuffer[kSettingOutBytes] = {0};
	void ClearOutBuffer()
	{
		for (int i = 0; i < kSettingOutBytes; i++)
		{
			_outputBuffer[i] = 0;
		}
	}

#pragma region Train
	BitType _outputBatchBuffer[BATCH_SIZE * kSettingOutDim] = {0};
	double _grads[BATCH_SIZE * kSettingInDim];

	void ClearOutBatchBuffer()
	{
		for (int b = 0; b < BATCH_SIZE; ++b)
		{
			for (int i = 0; i < kSettingOutBytes; i++)
			{
				_outputBuffer[b * kSettingOutBytes + i] = 0;
			}
		}
	}
#pragma endregion
};

#endif