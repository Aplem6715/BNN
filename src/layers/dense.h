#ifndef DENSE_H_INCLUDED_
#define DENSE_H_INCLUDED_

#include "layer.h"

template <typename PrevLayer_t, int OutputSize>
class DenseLayer : LayerBase
{
public:
	virtual uint8_t *Forward();
	virtual void Backward();

private:
	PrevLayer_t _prevLayer;
};

#endif