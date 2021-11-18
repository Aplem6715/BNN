#include <iostream>
#include <random>

#include "layers.h"
#include "util/mnist_trans.h"

using Input = IntInputLayer<2>;
using Hidden1 = IntSignActivation<IntAffineLayer<Input, 512>>;
using Hidden2 = IntSignActivation<IntAffineLayer<Hidden1, 256>>;
using OutLayer = IntAffineLayer<Hidden2, 2>;
using Network = OutLayer;

// using Input = BitInputLayer<2>;
// using Hidden1 = SignActivation<AffineLayer<Input, 16>>;
// using Hidden2 = SignActivation<AffineLayer<Hidden1, 8>>;
// using OutLayer = AffineLayer<Hidden2, 2>;
// using Network = OutLayer;

using RealInput = RealInputLayer<2>;
using RealHidden1 = HTanhActivationLayer<RealDenseLayer<RealInput, 64>>;
using RealHidden2 = HTanhActivationLayer<RealDenseLayer<RealHidden1, 32>>;
using RealNetwork = RealDenseLayer<RealHidden2, 2>;

void TrainReal(RealNetwork *net, int nbTrain)
{
	double input[BATCH_SIZE * 2] = {0};
	double diff[BATCH_SIZE * 2];
	double lr = 0.001;
	for (int i = 0; i < nbTrain; i++)
	{
		int x1 = GetRandUInt() % 2;
		int x2 = GetRandUInt() % 2;
		double teach[BATCH_SIZE * 2];
		teach[0] = x1 ^ x2;
		teach[1] = x1 ^ x2;
		input[0] = x1;
		input[1] = x2;
		const double *pred = net->BatchForward(input);

		double loss = 0;
		for (int i = 0; i < 2; i++)
		{
			double t = teach[i];
			double y = pred[i];
			diff[i] = lr * (t - y);
			loss += std::abs(t - y);
		}
		loss /= 2.0;
		net->BatchBackward(diff);
		std::cout << pred[0] << " - " << loss << std::endl;
	}
}

void Train(Network *net, int nbTrain)
{
	uint8_t input[BATCH_SIZE * 2] = {0};
	double teach[BATCH_SIZE * 2] = {0};
	double diff[BATCH_SIZE * 2];
	double lr = 0.001;
	double total_loss = 0;
	for (int i = 0; i < nbTrain; i++)
	{
		for (int b = 0; b < BATCH_SIZE; b++)
		{
			int x1 = GetRandUInt() % 2;
			int x2 = GetRandUInt() % 2;
			// input[b] = (uint8_t)(x1 << 7) + (uint8_t)(x2 << 6);
			input[b * 2] = x1;
			input[b * 2 + 1] = x2;
			teach[b * 2] = (x1 ^ x2) == 0 ? 128 : -128;
			teach[b * 2 + 1] = (x1 ^ x2) == 1 ? 128 : -128;
		}

		const int *pred = net->BatchForward(input);

		for (int b = 0; b < BATCH_SIZE; b++)
		{
			double loss = 0;
			for (int j = 0; j < 2; j++)
			{
				int idx = b * 2 + j;
				double y = pred[idx];
				double t = teach[idx];
				diff[idx] = t - y;
				loss += (y - t) * (y - t) * 0.5;
				// std::cout << y << " : " << t << std::endl;
			}
			loss /= 2.0;
			total_loss += loss;
		}
		total_loss /= BATCH_SIZE;

		net->BatchBackward(diff);
	}
	std::cout << total_loss << std::endl;
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
		Train(&net, 100);
		// std::cout << Test(&net, 100) << std::endl;
	}

	return 0;
}
