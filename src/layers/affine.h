#ifndef AFFINE_H_INCLUDED_
#define AFFINE_H_INCLUDED_

#include "../util/random_real.h"
#include "../net_common.h"

template <typename PrevLayer_t, int OutputSize>
class AffineLayer
{
public:
	using InType = typename PrevLayer_t::OutType;
	using OutType = ByteType;

	static_assert(std::is_same<InType, BitType>::value);

#pragma region 出力サイズ関連
	static constexpr int kOutDim = OutputSize;
	// SIMDパディング付き出力ビット幅
	static constexpr int kPaddedOutBitWidth = AddPaddingBitSize(kOutDim);
	// SIMDパディング付き出力バイト数
	static constexpr int kNumPaddedOutBytes = BitToByteSize(kPaddedOutBitWidth);
#pragma endregion

#pragma region 入力サイズ関連(前の層の値を引き継ぐ)
	// 入力次元数
	static constexpr int kInDim = PrevLayer_t::kOutDim;
	// 入力ビット幅（前の層のニューロン数）
	static constexpr int kInBitWidth = PrevLayer_t::kOutDim;
	// 入力バイト数
	static constexpr int kNumInBytes = BitToByteSize(kInBitWidth);
	// SIMDパディング付き入力ビット幅
	static constexpr int kPaddedInBitWidth = AddPaddingBitSize(kInBitWidth);
	// SIMDパディング付き入力バイト数
	static constexpr int kNumPaddedInBytes = BitToByteSize(kPaddedInBitWidth);
#pragma endregion

	const OutType *Forward(const uint8_t *netInput)
	{
		const InType *input = _prevLayer.Forward(netInput);

		return _outputBuffer;
	}

#pragma region Train
	OutType **BatchForward(const uint8_t **netInput)
	{
		_inputBufferPtr = _prevLayer.BatchForward(netInput);
		// TODO
		return _outputBatchBuffer;
	}

	void BatchBackward(const double **nextLayerGrads)
	{
		// TODO
		_prevLayer.BatchBackward(_grads);
	}
#pragma endregion

private:
	PrevLayer_t _prevLayer;
	OutType _outputBuffer[kOutDim];

#pragma region Train
	OutType _outputBatchBuffer[BATCH_SIZE][kOutDim];
	InType **_inputBufferPtr;
	double _grads[BATCH_SIZE][kInDim];
#pragma endregion
};

#endif