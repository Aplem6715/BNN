#ifndef SIGN_ACTIVATION_H_INCLUDED_
#define SIGN_ACTIVATION_H_INCLUDED_

#include <algorithm>
#include "../util/random_real.h"
#include "../net_common.h"

template <typename PrevLayer_t>
class SignActivation
{
public:
	static constexpr int kSettingOutDim = PrevLayer_t::kSettingOutDim;
	// 出力ビット幅
	static constexpr int kSettingOutBits = kSettingOutDim;
	// SIMDパディング付き出力ビット幅
	static constexpr int kPaddedOutBits = AddPaddingBitSize(kSettingOutDim);
	// SIMDパディング付き出力バイト数
	static constexpr int kPaddedOutBytes = BitToByteSize(kPaddedOutBits);
	// 入力次元数
	static constexpr int kSettingInDim = PrevLayer_t::kSettingOutDim;

private:
	PrevLayer_t _prevLayer;
	RealType *_inputBufferPtr;
	BitType _outputBuffer[kPaddedOutBytes] = {0};

#pragma region Train
	BitType _outputBatchBuffer[BATCH_SIZE * kPaddedOutBytes] = {0};
	double _grads[BATCH_SIZE * kSettingInDim];
#pragma endregion

public:
	const BitType *Forward(const uint8_t *netInput)
	{
		const RealType *input = _prevLayer.Forward(netInput);

		ClearOutBuffer();

		for (int i = 0; i < kSettingOutBits; ++i)
		{
			int block = GetBlockIndex(i);
			int shift = GetBitIndexInBlock(i);
			double htanh = std::max(-1.0, std::min(1.0, input[i]));
			uint8_t sign = (htanh - 0.5) > 0;
			_outputBuffer[block] |= sign << shift;
		}
		return _outputBuffer;
	}

	void ResetWeight()
	{
		_prevLayer.ResetWeight();
	}

#pragma region Train
	BitType *BatchForward(const uint8_t *netInput)
	{
		_inputBufferPtr = _prevLayer.BatchForward(netInput);

		ClearOutBatchBuffer();

		for (int b = 0; b < BATCH_SIZE; b++)
		{
			const RealType *batchInput = &_inputBufferPtr[b * kSettingInDim];
			BitType *batchOutput = &_outputBatchBuffer[b * kPaddedOutBytes];
			for (int i = 0; i < kSettingOutBits; ++i)
			{
				int block = GetBlockIndex(i);
				int shift = GetBitIndexInBlock(i);

				// hard-tanh
				double htanh = std::max(-1.0, std::min(1.0, batchInput[i]));

				// binalized neurons
				double probPositive = (htanh + 1.0) / 2.0;
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
			int batchShift = b * kSettingOutDim;
			for (int i = 0; i < kSettingOutDim; ++i)
			{
				double g = nextLayerGrads[batchShift + i];

				// d_Hard-tanh
				if (std::abs(_inputBufferPtr[batchShift + i]) <= 1)
				{
					_grads[batchShift + i] = g;
				}
				else
				{
					_grads[batchShift + i] = 0;
				}
			}
		}
		_prevLayer.BatchBackward(_grads);
	}

private:
#pragma endregion

	void ClearOutBuffer()
	{
		for (int i = 0; i < kPaddedOutBytes; i++)
		{
			_outputBuffer[i] = 0;
		}
	}

#pragma region Train
	void ClearOutBatchBuffer()
	{
		for (int b = 0; b < BATCH_SIZE; ++b)
		{
			for (int i = 0; i < kPaddedOutBytes; i++)
			{
				_outputBuffer[b * kPaddedOutBytes + i] = 0;
			}
		}
	}
#pragma endregion
};

#endif