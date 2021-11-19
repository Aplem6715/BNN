#ifndef INT_SIGN_ACTIVATION_H_INCLUDED_
#define INT_SIGN_ACTIVATION_H_INCLUDED_

#include "../util/random_real.h"
#include "../net_common.h"

template <typename PrevLayer_t>
class IntSignActivation
{
public:
	static constexpr int kSettingOutDim = PrevLayer_t::kSettingOutDim;

private:
    PrevLayer_t _prevLayer;
    int *_inputBufferPtr;
    int8_t _outputBuffer[kSettingOutDim] = {0};

#pragma region Train
    int8_t _outputBatchBuffer[kSettingOutDim] = {0};
	double _grads[kSettingOutDim];
#pragma endregion

public:
    const int8_t *Forward(const uint8_t *netInput)
    {
        // const int8_t *input = _prevLayer.Forward(netInput);

        // for (int i = 0; i < kSettingOutBits; ++i)
        // {
        //     double htanh = std::max(-1.0, std::min(1.0, input[i]));
        //     uint8_t sign = (htanh - 0.5) > 0;
        //     _outputBuffer[block] |= sign << shift;
        // }
        return _outputBuffer;
    }

    void ResetWeight()
    {
        _prevLayer.ResetWeight();
    }

#pragma region Train
	int8_t *BatchForward(const int8_t *netInput)
	{
		_inputBufferPtr = _prevLayer.BatchForward(netInput);

		for (int out = 0; out < kSettingOutDim; ++out)
		{
			// hard-tanh
			double htanh = std::max(-1, std::min(1, _inputBufferPtr[out]));

			// binalized neurons
			double probPositive = (htanh + 1.0) / 2.0;
			bool sign = GetRandReal() < probPositive;
			_outputBatchBuffer[out] = sign ? 1 : -1;
		}

		return _outputBatchBuffer;
	}

	void BatchBackward(const double *nextLayerGrads)
    {
		for (int i = 0; i < kSettingOutDim; ++i)
		{
			double g = nextLayerGrads[i];

			// d_Hard-tanh
			if (std::abs(_inputBufferPtr[i]) <= 1)
			{
				_grads[i] = g;
			}
			else
			{
				_grads[i] = 0;
			}
		}
		_prevLayer.BatchBackward(_grads);
	}
};

#endif