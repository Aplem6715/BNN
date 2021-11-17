#include <iostream>
#include <random>

#include "layers.h"
#include "util/mnist_trans.h"

using Input = BitInputLayer<2>;
using Hidden1 = SignActivation<AffineLayer<Input, 64>>;
using Hidden2 = SignActivation<AffineLayer<Hidden1, 32>>;
using OutLayer = AffineLayer<Hidden2, 2>;
using Network = LinearActivation<OutLayer>;

void Train(Network *net, int nbTrain)
{
	uint8_t input[BATCH_SIZE * 1] = {0};
	double diff[BATCH_SIZE * 2];
	double lr = 0.01;
	for (int i = 0; i < nbTrain; i++)
	{
		int x1 = GetRandUInt() % 2;
		int x2 = GetRandUInt() % 2;
		double teach[BATCH_SIZE * 2];
		teach[0] = (x1 ^ x2) == 0 ? 1 : -1;
		teach[1] = (x1 ^ x2) == 1 ? 1 : -1;
		input[0] = (uint8_t)(x1 << 7) + (uint8_t)(x2 << 6);
		const int *pred = net->BatchForward(input);

		double loss = 0;
		for (int i = 0; i < 2; i++)
		{
			double t = teach[i];
			double y = pred[i];
			if (1 - t * y <= 0)
			{
				diff[i] = 0;
			}
			else
			{
				double d = (2 * t * (t * y - 1));
				d = std::max(-1.0, std::min(1.0, d));
				diff[i] = lr * d;
			}
			loss += std::max(0.0, 1 - t * y);
		}
		loss /= 2.0;
		net->BatchBackward(diff);
		std::cout << pred[0] << " - " << loss << std::endl;
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
	RandomSeed(42);
	Network net;
	net.ResetWeight();

	for (int i = 0; i < 100; i++)
	{
		Train(&net, 10000);
		// std::cout << Test(&net, 100) << std::endl;
	}

	return 0;
}
