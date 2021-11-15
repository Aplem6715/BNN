#ifndef INPUT_LAYER_H_INCLUDED_
#define INPUT_LAYER_H_INCLUDED_

#include "layer.h"

template <int OutputSize>
class InputLayer : LayerBase
{
	virtual uint8_t* Forward();
	virtual void Backward();
};

#endif