#ifndef INPUT_LAYER_H_INCLUDED_
#define INPUT_LAYER_H_INCLUDED_

#include "layer_base.h"

template <int InputSize, int OutputSize>
class InputLayer {
   public:
    static constexpr int kOutputSize    = OutputSize;
    static constexpr int kInputBitSize  = InputSize;
    static constexpr int kOutputBitSize = OutputSize;
    static constexpr int kSimdBlockNum  = kInputBitSize / SIMD_BIT_WIDTH;

    uint8_t *Forward(uint8_t *netInput) { return netInput; }

    void Backward(const double *nextGrads) {
        // 終端
    }

    void ResetWeights() {}
};

#endif