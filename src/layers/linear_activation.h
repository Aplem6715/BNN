#ifndef LINEAR_ACTIVATION_H_INCLUDED_
#define LINEAR_ACTIVATION_H_INCLUDED_

#include "../util/random_real.h"
#include "../net_common.h"

template <typename PrevLayer_t>
class LinearActivation
{
public:
#pragma region 入力サイズ関連(前の層の値を引き継ぐ)
	// 入力次元数
	static constexpr int kSettingInDim = PrevLayer_t::kSettingOutDim;
#pragma endregion

	const RealType *Forward(const uint8_t *netInput)
	{
		const RealType *input = _prevLayer.Forward(netInput);

		for (int i = 0; i < kSettingInDim; ++i)
		{
			_outputBuffer[i] = input[i] > 0 ? 1 : -1;
		}
		return _outputBuffer;
	}

	void ResetWeight()
	{
		_prevLayer.ResetWeight();
	}

#pragma region Train
	RealType *BatchForward(const uint8_t *netInput)
	{
		_inputBufferPtr = _prevLayer.BatchForward(netInput);

		for (int b = 0; b < BATCH_SIZE; b++)
		{
			const RealType *batchInput = &_inputBufferPtr[b * kSettingInDim];
			RealType *batchOutput = &_outputBatchBuffer[b * kSettingInDim];
			for (int i = 0; i < kSettingInDim; ++i)
			{
				double htanh = std::max(-1.0, std::min(1.0, batchInput[i]));

				// 決定的
				double tmp = (htanh + 1.0) / 2.0 - 0.5;
				double sign = (tmp > 0) ? 1 : -1;
				batchOutput[i] = sign;
			}
		}
		return _outputBatchBuffer;
	}

	void BatchBackward(const double *nextLayerGrads)
	{
		for (int b = 0; b < BATCH_SIZE; ++b)
		{
			int batchShift = b * kSettingInDim;
			for (int i = 0; i < kSettingInDim; ++i)
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
		_prevLayer.BatchBackward(nextLayerGrads);
	}
#pragma endregion

private:
	PrevLayer_t _prevLayer;
	RealType _outputBuffer[kSettingInDim];

#pragma region Train
	RealType *_inputBufferPtr;
	RealType _outputBatchBuffer[BATCH_SIZE * kSettingInDim];
	double _grads[BATCH_SIZE * kSettingInDim];
#pragma endregion
};

#endif