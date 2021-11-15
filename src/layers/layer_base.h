#ifndef LAYER_H_INCLUDED_
#define LAYER_H_INCLUDED_

#include "../net_common.h"

template<typename OutputType>
struct LayerBase
{
	virtual OutputType *Forward(uint8_t* netInput) = 0;
	virtual void Backward(const double *nextGrads) = 0;
};

#endif