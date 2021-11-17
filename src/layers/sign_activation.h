#ifndef SIGN_ACTIVATION_H_INCLUDED_
#define SIGN_ACTIVATION_H_INCLUDED_

#include <algorithm>
#include "../util/random_real.h"
#include "../net_common.h"

template <typename PrevLayer_t>
class SignActivation
{
public:
	using InType = typename PrevLayer_t::OutType;
	using OutType = BitType;

	static_assert(std::is_same<InType, ByteType>::value);

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

	const OutType *Forward(const uint8_t *netInput)
	{
		const BitType *input = _prevLayer.Forward(netInput);

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

	// void Backward(const double *nextLayerGrads)
	// {
	// 	for (int i = 0; i < kSettingInDim; i++)
	// 	{
	// 		double x = nextLayerGrads[i];
	// 		// Hard-tanh
	// 		_grads[i] = std::max(-1.0, std::min(1.0, x));
	// 	}
	// 	_prevLayer.Backward(_grads);
	// }

#pragma region Train
	OutType **BatchForward(const uint8_t **netInput)
	{
		_inputBufferPtr = _prevLayer.BatchForward(netInput);

		ClearOutBatchBuffer();

		for (int b = 0; b < BATCH_SIZE; b++)
		{
			InType* batchInput = _inputBufferPtr[b];
			OutType* batchOutput = _outputBatchBuffer[b];
			for (int i = 0; i < kSettingInDim; ++i)
			{
				int block = GetBlockIndex(i);
				int shift = GetBitIndexInBlock(i);
				uint8_t sign = batchInput[i] > 0 ? 1 : 0;
				batchOutput[block] |= sign << shift;
			}
		}
		return _outputBatchBuffer;
	}

	void BatchBackward(const double **nextLayerGrads)
	{
		for (int b = 0; b < BATCH_SIZE; ++b)
		{
			for (int i = 0; i < kSettingInDim; ++i)
			{
				double x = nextLayerGrads[b][i];
				// Hard-tanh
				_grads[b][i] = std::max(-1.0, std::min(1.0, x));
			}
		}
		_prevLayer.BatchBackward(_grads);
	}
#pragma endregion

private:
	PrevLayer_t _prevLayer;
	OutType _outputBuffer[kSettingOutBytes];
	void ClearOutBuffer()
	{
		for (int i = 0; i < kSettingOutBytes; i++)
		{
			_outputBuffer[i] = 0;
		}
	}

#pragma region Train
	OutType _outputBatchBuffer[BATCH_SIZE][kSettingOutDim];
	InType **_inputBufferPtr;
	double _grads[BATCH_SIZE][kSettingInDim];

	void ClearOutBatchBuffer()
	{
		for (int b = 0; b < BATCH_SIZE; ++b)
		{
			for (int i = 0; i < kSettingOutBytes; i++)
			{
				_outputBuffer[b][i] = 0;
			}
		}
	}
#pragma endregion
};

#endif