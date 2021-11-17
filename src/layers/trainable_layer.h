#ifndef TRAINABLE_LAYER_H_INCLUDED_
#define TRAINABLE_LAYER_H_INCLUDED_

#include "layer_base.h"

template <typename PrevLayer_t, typename OutputType>
struct TrainableLayer : LayerBase<PrevLayer_t, OutputType>
{
	using InType = PrevLayer_t::OutType;
protected:
	OutputType _outputBatchBuffer[BATCH_SIZE][kOutDim];
	InType *_inputBufferPtr;
	double _grads[BATCH_SIZE][kInDim];
};

#endif