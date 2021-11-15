#ifndef SOFTMAX_LAYER_H_INCLUDED_
#define SOFTMAX_LAYER_H_INCLUDED_

#include "layer_base.h"

template <typename PrevLayer_t>
class SoftmaxLayer : LayerBase<double> {
   public:
    static constexpr int kInputSize  = PrevLayer_t::kOutputSize;
    static constexpr int kOutputSize = kInputSize;

    const double *Forward(uint8_t *netInput) {
        const int *input = _prevLayer.Forward(netInput);

        int in_max = INT_MIN;
        for (int i_out = 0; i_out < kOutputSize; ++i_out) {
            if (input[i_out] > in_max) {
                in_max = input[i_out];
            }
        }

        double sum = 0;
        for (int i_out = 0; i_out < kOutputSize; ++i_out) {
            double exp = std::exp(input[i_out] - in_max);
            sum += exp;
            _outputBuffer[i_out] = exp;
        }

        for (int i_out = 0; i_out < kOutputSize; ++i_out) {
            _outputBuffer[i_out] /= sum;
        }

        return _outputBuffer;
    }

    void Backward(const double *nextGrads) {
        // 手前の層へ逆伝播(softmaxの逆伝播は誤差（y[k] - t[k]）そのまま)
        _prevLayer.Backward(nextGrads);
    }

    void ResetWeights() { _prevLayer.ResetWeights(); }

   private:
    // 前のレイヤー
    PrevLayer_t _prevLayer;
    // 順伝播 出力バッファ(0~1の実数値)
    double _outputBuffer[kOutputSize];
};

#endif