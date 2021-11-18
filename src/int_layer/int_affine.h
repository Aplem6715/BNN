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

#pragma region Train
    int _outputBatchBuffer[BATCH_SIZE * kSettingOutDim];
    int8_t *_inputBufferPtr;
    double _realWeight[kSettingOutDim][kSettingInDim] = {0};
    double _grads[BATCH_SIZE * kSettingInDim] = {0};
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
            for (int in = 0; in < kSettingInDim; ++in)
            {
                _weight[i_out][in] = GetRandReal() > 0.5 ? 1 : -1;
            }
        }
        _prevLayer.ResetWeight();
    }

#pragma region Train
    int *BatchForward(const uint8_t *netInput)
    {
        _inputBufferPtr = _prevLayer.BatchForward(netInput);

        for (int b = 0; b < BATCH_SIZE; ++b)
        {
            const int8_t *batchInput = &_inputBufferPtr[b * kSettingInDim];
            int *batchOutput = &_outputBatchBuffer[b * kSettingOutDim];
            for (int i_out = 0; i_out < kSettingOutDim; ++i_out)
            {
                // パディング分も含めて±1積和演算
                int sum = 0;
                for (int in = 0; in < kSettingInDim; ++in)
                {
                    sum += batchInput[in] * _weight[i_out][in];
                }

                batchOutput[i_out] = sum;
            }
        }

        return _outputBatchBuffer;
    }

    void BatchBackward(const double *nextLayerGrads)
    {
        // 勾配更新
        for (int batch = 0; batch < BATCH_SIZE; ++batch)
        {
            const double *nextBatchGrad = &nextLayerGrads[batch * kSettingOutDim];
            double *grads = &_grads[batch * kSettingInDim];
            for (int in = 0; in < kSettingInDim; ++in)
            {
                double sum = 0;
                for (int out = 0; out < kSettingOutDim; ++out)
                {
                    sum += nextBatchGrad[out] * _weight[out][in];
                }
                grads[in] = sum;
            }
        }

        for (int batch = 0; batch < BATCH_SIZE; ++batch)
        {
            const double *nextBatchGrad = &nextLayerGrads[batch * kSettingOutDim];
            const int8_t *batchInput = &_inputBufferPtr[batch * kSettingInDim];

            // 重み調整幅を計算
            for (int out = 0; out < kSettingOutDim; out++)
            {
                for (int in = 0; in < kSettingInDim; in++)
                {
                    _realWeight[out][in] += nextBatchGrad[out] * batchInput[in];
                }
            }
        }

        // 重み更新
        for (int out = 0; out < kSettingOutDim; out++)
        {
            int8_t *weight = _weight[out];
            for (int in = 0; in < kSettingInDim; in++)
            {
                // Clipping
                double tmp_w = _realWeight[out][in];
                tmp_w = std::max(-1.0, std::min(1.0, tmp_w));
                _realWeight[out][in] = tmp_w;

                weight[in] = (tmp_w > 0) ? 1 : -1;
            }
        }

        _prevLayer.BatchBackward(_grads);
    }
#pragma endregion
};

#endif