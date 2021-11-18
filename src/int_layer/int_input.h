#ifndef INT_INPUT_LAYER_H_INCLUDED_
#define INT_INPUT_LAYER_H_INCLUDED_

#include "../net_common.h"

template <int InputSize>
class IntInputLayer
{
public:
    static constexpr int kSettingOutDim = InputSize;

private:
    // 順伝播 出力バッファ(出力層はint)
    int8_t _outputBuffer[kSettingOutDim] = {0};

#pragma region Train
    int8_t _outputBatchBuffer[BATCH_SIZE * kSettingOutDim] = {0};
#pragma endregion

public:
    int8_t *Forward(const uint8_t *netInput)
    {
        for (int i = 0; i < kSettingOutDim; i++)
        {
            _outputBuffer[i] = netInput[i] == 0 ? -1 : 1;
        }
        return _outputBuffer;
    }

    void ResetWeight() {}

#pragma region Train
    int8_t *BatchForward(const uint8_t *netInput)
    {
        for (int batch = 0; batch < BATCH_SIZE; ++batch)
        {
            int8_t *outputBuffer = &_outputBatchBuffer[batch * kSettingOutDim];
            for (int i = 0; i < kSettingOutDim; ++i)
            {
                outputBuffer[i] = netInput[batch * kSettingOutDim + i];
            }
        }
        return _outputBatchBuffer;
    }

    void BatchBackward(const double *nextLayerGrads)
    {
        // 終端
    }
#pragma endregion
};

#endif