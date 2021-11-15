#ifndef SOFTMAX_LAYER_H_INCLUDED_
#define SOFTMAX_LAYER_H_INCLUDED_

#include "layer_base.h"

template <typename PrevLayer_t>
class SoftmaxLayer : LayerBase<double>
{
public:
	static constexpr int kInputSize = PrevLayer_t::kOutputSize;
	static constexpr int kOutputSize = kInputSize;

	virtual double *Forward(uint8_t *netInput);

	virtual void Backward(const double *nextGrads)
	{
		// 手前の層へ逆伝播(softmaxの逆伝播は誤差（y[k] - t[k]）そのまま)
		_prevLayer.Backward(nextGrads);
	}

private:
	// 前のレイヤー
	PrevLayer_t _prevLayer;
	// 順伝播 出力バッファ(0~1の実数値)
	double _outputBuffer[kOutputSize];

};

#endif