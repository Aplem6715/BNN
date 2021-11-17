#ifndef AFFINE_H_INCLUDED_
#define AFFINE_H_INCLUDED_

#include "../net_common.h"
#include "../util/random_real.h"

template <typename PrevLayer_t, int OutputSize>
class AffineLayer {
   public:
    using InType  = typename PrevLayer_t::OutType;
    using OutType = ByteType;

    static_assert(std::is_same<InType, BitType>::value);

#pragma region 出力サイズ関連
    static constexpr int kSettingOutDim = OutputSize;
    // SIMDパディング付き出力ビット幅
    static constexpr int kPaddedOutBitWidth = AddPaddingBitSize(kSettingOutDim);
    // SIMDパディング付き出力バイト数
    static constexpr int kPaddedOutBytes = BitToByteSize(kPaddedOutBitWidth);
#pragma endregion

#pragma region 入力サイズ関連(前の層の値を引き継ぐ)
    // 入力次元数
    static constexpr int kSettingInDim = PrevLayer_t::kOutDim;
    // 入力ビット幅（前の層のニューロン数）
    static constexpr int kSettingInBitWidth = PrevLayer_t::kOutDim;
    // 入力バイト数
    static constexpr int kSettingNumInBytes = BitToByteSize(kSettingInBitWidth);
    // SIMDパディング付き入力ビット幅
    static constexpr int kPaddedInBitWidth =
        AddPaddingBitSize(kSettingInBitWidth);
    // SIMDパディング付き入力バイト数
    static constexpr int kPaddedInBytes = BitToByteSize(kPaddedInBitWidth);
#pragma endregion

    const OutType *Forward(const uint8_t *netInput) {
        const InType *input = _prevLayer.Forward(netInput);

        return _outputBuffer;
    }

#pragma region Train
    OutType **BatchForward(const uint8_t **netInput) {
        _inputBufferPtr = _prevLayer.BatchForward(netInput);
        // TODO

        for (int b = 0; b < BATCH_SIZE; ++b) {
            for (int i_out = 0; i_out < kPaddedOutBytes; ++i_out) {
                for (int i = 0; i < kSettingInBitWidth; ++i) {
                }
            }
        }

        return _outputBatchBuffer;
    }

    void BatchBackward(const double **nextLayerGrads) {
        // TODO
        _prevLayer.BatchBackward(_grads);
    }
#pragma endregion

   private:
    PrevLayer_t _prevLayer;
    OutType _outputBuffer[kSettingOutDim];
    ByteType _weight[kPaddedInBytes][kPaddedOutBytes];

#pragma region Train
    OutType _outputBatchBuffer[BATCH_SIZE][kSettingOutDim];
    InType **_inputBufferPtr;
    double _grads[BATCH_SIZE][kSettingInDim];
#pragma endregion
};

#endif