#ifndef LAYER_H_INCLUDED_
#define LAYER_H_INCLUDED_

#include "../net_common.h"

struct LayerBase
{
	virtual uint8_t *Forward(uint8_t* netInput) = 0;
	virtual void Backward() = 0;
};

#endif