#ifndef REAL_INPUT_H_
#define REAL_INPUT_H_

template <int InputSize>
class RealInputLayer
{
public:
#pragma region 出力サイズ関連
    static constexpr int kSettingOutDim = InputSize;
#pragma endregion

private:
    // 順伝播 出力バッファ(出力層はint)
    RealType _outputBuffer[kSettingOutDim] = {0};

#pragma region Train
    RealType _outputBatchBuffer[BATCH_SIZE * kSettingOutDim] = {0};
#pragma endregion

public:
    RealType *Forward(const double *netInput)
    {
        for (int i = 0; i < kSettingOutDim; i++)
        {
            _outputBuffer[i] = netInput[i];
        }
        return _outputBuffer;
    }

    void ResetWeight() {}

#pragma region Train
    RealType *BatchForward(const double *netInput)
    {
        for (int batch = 0; batch < BATCH_SIZE; ++batch)
        {
            RealType *outputBuffer = &_outputBatchBuffer[batch * kSettingOutDim];
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