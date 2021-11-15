#include <iostream>

#include "util/mnist_trans.h"
#include "layers.h"

using Input = InputLayer<784, 4096>;
using Hidden1 = DenseLayer<Input, 4096>;
using Hidden2 = DenseLayer<Hidden1, 4096>;
using Network = SoftmaxLayer<OutputLayer<Hidden2, 10>>;

int main(int, char**) {
	Network net;
	uint8_t *inputImage = new uint8_t[net.kOutputSize];

	double *pred = net.Forward(inputImage);

	for (int i = 0; i < net.kOutputSize; i++)
	{
		std::cout << pred[i] << std::endl;
	}

	return 0;
}
