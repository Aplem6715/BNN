#include <iostream>
#include <random>

#include "layers.h"
#include "util/mnist_trans.h"

using Input = InputLayer<2>;
using Hidden1 = DenseLayer<Input, 32>;
using Hidden2 = DenseLayer<Hidden1, 32>;
using Network = OutputLayer<Hidden2, 1>;

void Train(Network *net, int nbTrain)
{
	std::random_device rnd;
	uint8_t *input = new uint8_t[1]{0};
	double diff[1];
	double lr = 0.001;
	for (int i = 0; i < nbTrain; i++)
	{
		int x1 = rnd() % 2;
		int x2 = rnd() % 2;
		double teach = x1 ^ x2;
		input[0] = (x1 << 1) + x2;

		const int *pred = net->Forward(input);

		double d = (teach - *pred);
		diff[0] = lr * d;
		net->Backward(diff);
		// std::cout << d << std::endl;
	}
}

double Test(Network *net, int nbTest)
{
	std::random_device rnd;
	uint8_t *input = new uint8_t[1]{0};
	double error = 0;
	for (int i = 0; i < nbTest; i++)
	{
		int x1 = rnd() % 2;
		int x2 = rnd() % 2;
		double teach = x1 ^ x2;
		input[0] = (x1 << 1) + x2;

		const int *pred = net->Forward(input);

		error += std::abs(teach - pred[0]);
	}
	return error / nbTest;
}

// TODO: アクティベーション層の分離：softmax, popsum
int main(int, char **)
{
	Network net;
	net.ResetWeights();

	for (int i = 0; i < 100; i++)
	{
		Train(&net, 10000);
		std::cout << Test(&net, 100) << std::endl;
	}

	return 0;
}
