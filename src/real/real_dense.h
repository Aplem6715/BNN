#ifndef REAL_DENSE_H_
#define REAL_DENSE_H_

#include <intrin.h>

#include "../net_common.h"
#include "../util/random_real.h"

template <typename PrevLayer_t, int OutputSize>
class RealDenseLayer
{
public:
    static constexpr int kSettingOutDim = OutputSize;
    // 入力次元数
    static constexpr int kSettingInDim = PrevLayer_t::kSettingOutDim;

private:
    PrevLayer_t _prevLayer;
    RealType _outputBuffer[kSettingOutDim];
    RealType _weight[kSettingOutDim][kSettingInDim] = {0};
	RealType _bias[kSettingOutDim] = {0};

#pragma region Train
    RealType _outputBatchBuffer[BATCH_SIZE * kSettingOutDim];
    RealType *_inputBufferPtr;
    double _weightDiff[BATCH_SIZE][kSettingOutDim][kSettingInDim] = {0};
    double _grads[BATCH_SIZE * kSettingInDim] = {0};
#pragma endregion

public:
    const int *Forward(const double *netInput)
    {
        const BitType *input = _prevLayer.Forward(netInput);
        // TODO
        return _outputBuffer;
    }

    void ResetWeight()
    {
        for (int i_out = 0; i_out < kSettingOutDim; ++i_out)
        {
            for (int block = 0; block < kSettingInDim; ++block)
            {
                _weight[i_out][block] = GetRandReal() * 2 - 1;
            }
        }
        _prevLayer.ResetWeight();
    }

#pragma region Train
    RealType *BatchForward(const double *netInput)
    {
        _inputBufferPtr = _prevLayer.BatchForward(netInput);

        for (int b = 0; b < BATCH_SIZE; ++b)
        {
            const RealType *batchInput = &_inputBufferPtr[b * BATCH_SIZE];
            RealType *batchOutput = &_outputBatchBuffer[b * BATCH_SIZE];
            for (int i_out = 0; i_out < kSettingOutDim; ++i_out)
            {
				double sum = _bias[i_out];
				for (int block = 0; block < kSettingInDim; ++block)
                {
                    sum += batchInput[block] * _weight[i_out][block];
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

        // 重み調整幅を計算
        for (int batch = 0; batch < BATCH_SIZE; ++batch)
        {
            const double *nextBatchGrad = &nextLayerGrads[batch * kSettingOutDim];
            const RealType *batchInput = &_inputBufferPtr[batch * kSettingInDim];
            for (int out = 0; out < kSettingOutDim; out++)
            {
				_bias[out] += nextBatchGrad[out];
				for (int in = 0; in < kSettingInDim; in++)
                {
                    _weight[out][in] += nextBatchGrad[out] * batchInput[in];
                }
            }
        }

        // // 重み更新
        // for (int out = 0; out < kSettingOutDim; out++)
        // {
        //     RealType *weight = _weight[out];
        //     for (int in = 0; in < kSettingInDim; in++)
        //     {
        //         // Clipping
        //         double tmp_w = _realWeight[out][in];
        //         tmp_w = std::max(-1.0, std::min(1.0, tmp_w));
        //         _realWeight[out][in] = tmp_w;

        //         uint8_t newBit = tmp_w > 0 ? 1 : 0;
        //         newBit = (newBit << shift);
        //         // 左から3ビット目の重みを更新するとき 元:11111101 ->
        //         // masked:11011101
        //         uint8_t masked_w = weight[block] & ~(1 << shift);

        //         // マスクで0になった位置に新しい重みbitを入れ込む
        //         // masked:11011101 | shifted:00100000
        //         weight[block] = masked_w | newBit;
        //     }
        // }

        _prevLayer.BatchBackward(_grads);
    }
#pragma endregion
};

#endif