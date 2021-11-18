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

	const IntType *Forward(const uint8_t *netInput)
	{
		const IntType *input = _prevLayer.Forward(netInput);

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
	IntType *BatchForward(const uint8_t *netInput)
	{
		_inputBufferPtr = _prevLayer.BatchForward(netInput);

		for (int b = 0; b < BATCH_SIZE; b++)
		{
			double *batchInput = &_inputBufferPtr[b * kSettingInDim];
			IntType *batchOutput = &_outputBatchBuffer[b * kSettingInDim];
			for (int i = 0; i < kSettingInDim; ++i)
			{
				batchOutput[i] = batchInput[i] > 0.5 ? 1 : -1;
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
				double x = nextLayerGrads[batchShift + i];
				if (std::abs(_inputBufferPtr[batchShift + i]) <= 1)
				{
					// Hard-tanh (straight-through estimator)
					_grads[batchShift + i] = std::max(-1.0, std::min(1.0, x));
				}
				else
				{
					// こっち側に来ることはないのでは？
					_grads[batchShift + i] = 0;
				}
			}
		}
		_prevLayer.BatchBackward(nextLayerGrads);
	}
#pragma endregion

private:
	PrevLayer_t _prevLayer;
	IntType _outputBuffer[kSettingInDim];

#pragma region Train
	double *_inputBufferPtr;
	IntType _outputBatchBuffer[BATCH_SIZE * kSettingInDim];
	double _grads[BATCH_SIZE * kSettingInDim];
#pragma endregion
};

#endif