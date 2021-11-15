#include "softmax_layer.h"
#include <cmath>

template <typename PrevLayer_t>
double *SoftmaxLayer<PrevLayer_t>::Forward(uint8_t *netInput)
{
	int *input = PrevLayer_t::Forward(netInput);

	double sum = 0;
	for (int i_out = 0; i_out < kOutputSize; ++i_out)
	{
		double exp = std::exp(input[i_out]);
		sum += exp;
		_outputBuffer[i_out] = exp;
	}

	for (int i_out = 0; i_out < kOutputSize; ++i_out)
	{
		_outputBuffer[i_out] /= sum;
	}
	
	return _outputBuffer;
}