#ifndef REAL_RELU_H_
#define REAL_RELU_H_

#include <algorithm>
#include "../util/random_real.h"
#include "../net_common.h"

template <typename PrevLayer_t>
class ReLULayer
{
public:
    static constexpr int kSettingOutDim = PrevLayer_t::kSettingOutDim;
    // 入力次元数
    static constexpr int kSettingInDim = PrevLayer_t::kSettingOutDim;

private:
    PrevLayer_t _prevLayer;
    RealType _outputBuffer[kSettingOutDim] = {0};

#pragma region Train
    RealType *_inputBufferPtr;
    RealType _outputBatchBuffer[BATCH_SIZE * kSettingOutDim] = {0};
    double _grads[BATCH_SIZE * kSettingInDim];
#pragma endregion

public:
    const RealType *Forward(const double *netInput)
    {
        const RealType *input = _prevLayer.Forward(netInput);

        for (int i = 0; i < kSettingOutDim; ++i)
        {
            _outputBuffer[i] = std::min(0.0, input[i]);
        }
        return _outputBuffer;
    }

    void ResetWeight()
    {
        _prevLayer.ResetWeight();
    }

#pragma region Train
    RealType *BatchForward(const double *netInput)
    {
        _inputBufferPtr = _prevLayer.BatchForward(netInput);

        for (int b = 0; b < BATCH_SIZE; b++)
        {
            const double *batchInput = &_inputBufferPtr[b * kSettingInDim];
            RealType *batchOutput = &_outputBatchBuffer[b * kSettingInDim];
            for (int i = 0; i < kSettingInDim; ++i)
            {
                batchOutput[i] = std::max(0.0, batchInput[i]);
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
                // Hard-tanh (straight-through estimator)
                if (_inputBufferPtr[batchShift + i] < 0)
                {
                    _grads[batchShift + i] = 0;
                }
                else
                {
                    _grads[batchShift + i] = std::max(0.0, x);
                }
            }
        }
        _prevLayer.BatchBackward(_grads);
    }
#pragma endregion
};

#endif