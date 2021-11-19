#ifndef INT_AFFINE_H_INCLUDED_
#define INT_AFFINE_H_INCLUDED_

#include "../net_common.h"
#include "../util/random_real.h"

template <typename PrevLayer_t, int OutputSize>
class IntAffineLayer
{
public:
    static constexpr int kSettingOutDim = OutputSize;
    // 入力次元数
    static constexpr int kSettingInDim = PrevLayer_t::kSettingOutDim;

private:
    PrevLayer_t _prevLayer;
    int _outputBuffer[kSettingOutDim];
    int8_t _weight[kSettingOutDim][kSettingInDim] = {0};
	RealType _bias[kSettingOutDim] = {0};

#pragma region Train
    int _outputBatchBuffer[kSettingOutDim];
    int8_t *_inputBufferPtr;
    double _realWeight[kSettingOutDim][kSettingInDim] = {0};
    double _grads[kSettingInDim] = {0};
#pragma endregion

public:
    const int *Forward(const uint8_t *netInput)
    {
        const int8_t *input = _prevLayer.Forward(netInput);
        // TODO
        return _outputBuffer;
    }

    void ResetWeight()
    {
        for (int i_out = 0; i_out < kSettingOutDim; ++i_out)
        {
			_bias[i_out] = 0;
			for (int in = 0; in < kSettingInDim; ++in)
			{
                _weight[i_out][in] = GetRandReal() > 0.5 ? 1 : -1;
            }
        }
        _prevLayer.ResetWeight();
    }

#pragma region Train
	int *BatchForward(const int8_t *netInput)
	{
		_inputBufferPtr = _prevLayer.BatchForward(netInput);

		for (int i_out = 0; i_out < kSettingOutDim; ++i_out)
		{
			// パディング分も含めて±1積和演算
			int sum = _bias[i_out];
			for (int in = 0; in < kSettingInDim; ++in)
			{
				sum += _inputBufferPtr[in] * _weight[i_out][in];
			}

			_outputBatchBuffer[i_out] = sum;
		}

		return _outputBatchBuffer;
	}

	void BatchBackward(const double *nextLayerGrads)
	{
		// 勾配更新
		for (int in = 0; in < kSettingInDim; ++in)
		{
			double sum = 0;
			for (int out = 0; out < kSettingOutDim; ++out)
			{
				sum += nextLayerGrads[out] * _weight[out][in];
			}
			_grads[in] = sum;
		}

		// 重み調整幅を計算
		for (int out = 0; out < kSettingOutDim; out++)
		{
			_bias[out] += nextLayerGrads[out];
			for (int in = 0; in < kSettingInDim; in++)
			{
				_realWeight[out][in] += nextLayerGrads[out] * _inputBufferPtr[in];
			}
		}

		// 重み更新
		for (int out = 0; out < kSettingOutDim; out++)
		{
			for (int in = 0; in < kSettingInDim; in++)
			{
				// Clipping
				double tmp_w = std::max(-1.0, std::min(1.0, _realWeight[out][in]));
				_realWeight[out][in] = tmp_w;

				_weight[out][in] = (tmp_w > 0) ? 1 : -1;
			}
		}

		_prevLayer.BatchBackward(_grads);
	}
#pragma endregion
};

#endif