#ifndef OUTPUT_LAYER_H_INCLUDED_
#define OUTPUT_LAYER_H_INCLUDED_

#include "layer_base.h"

template <typename PrevLayer_t, int OutputSize>
class OutputLayer {
   public:
    static constexpr int kInputBitSize       = PrevLayer_t::kOutputSize;
    static constexpr int kPaddedInputBitSize = AddPaddingBitSize(kInputBitSize);
    static constexpr int kPaddedInputByteSize = kPaddedInputBitSize / BIT_WIDTH;

    static constexpr int kOutputSize = OutputSize;

    static constexpr int kSimdBlockNum = kPaddedInputBitSize / SIMD_BIT_WIDTH;

    static_assert(kInputBitSize % BIT_WIDTH == 0,
                  "入力データはBIT_WIDTH単位でないといけない");
    // static constexpr size_t kWeightCount = kInputSize * kOutputBitSize;

    const int *Forward(uint8_t *netInput) {
        const auto input = _prevLayer.Forward(netInput);

        // TODO: SIMD化
        // TODO: パディング分のpop無視
        for (int i_out = 0; i_out < kOutputSize; ++i_out) {
            int sum              = 0;
            _outputBuffer[i_out] = 0;
            for (int i_in = 0; i_in < kPaddedInputByteSize; ++i_in) {
                sum += __popcnt(input[i_in] ^ _weights[i_in]);  // ベクトル内積
            }

            _outputBuffer[i_out] =
                2 * sum -
                kInputBitSize;  // 分布を0中心にシフト(0~2n を -n<0<nに)
        }
        return _outputBuffer;
    }

    void Backward(const double *nextGrads) {
        for (int i_batch; i_batch < BATCH_SIZE; ++i_batch) {
            for (int j = 0; j < kInputBitSize; ++j) {
                double sum = 0;

                int bit_block = j / BIT_WIDTH;
                int bit_shift = BIT_WIDTH - (j % BIT_WIDTH);
                int w_bit     = (_weights[bit_block] >> bit_shift) & 0x1;
                // ウェイトが0の時はsumは増加しないのでループそのものをスキップ
                if (w_bit == 1) {
                    for (int i = 0; i < kOutputSize; i++) {
                        // sum += nextGrad * weight
                        sum += nextGrads[i] * 1 /*ifブロック内でw_bitは必ず1*/;
                    }
                }

                _grads[j] += sum;
            }
        }

        // 手前の層へ逆伝播
        _prevLayer.Backward(_grads);
    }

    void ResetWeights() {
        _prevLayer.ResetWeights();
        for (int i = 0; i < kPaddedInputByteSize; i++) {
            _weights[i] = 0;
        }
    }

   private:
    // 前のレイヤー
    PrevLayer_t _prevLayer;
    // 順伝播 重み
    uint8_t _weights[kPaddedInputByteSize] = {0};
    // 順伝播 出力バッファ(出力層はint)
    int _outputBuffer[kOutputSize] = {0};

#pragma region Train
    // 前のレイヤーから受け取った入力値の履歴
    uint8_t _batchInput[BATCH_SIZE][kPaddedInputByteSize];
    // 逆伝播 重み更新用の勾配
    double _grads[kInputBitSize];
#pragma endregion
};

#endif