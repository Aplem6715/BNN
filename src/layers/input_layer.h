#ifndef INPUT_LAYER_H_INCLUDED_
#define INPUT_LAYER_H_INCLUDED_

#include "layer.h"

template <int InputSize, int OutputSize>
class InputLayer : LayerBase
{
	static constexpr int kInputSize = InputSize;
	static constexpr int kOutputSize = OutputSize;
	static constexpr int kSimdBlockNum = kInputSize / SIMD_BIT_WIDTH;
	
	virtual uint8_t *Forward(uint8_t *netInput){
		return netInput;
	}

	virtual void Backward(){
		// 終端
	}
};

#endif