#include <iostream>
#include <random>

#include "layers.h"
#include "util/mnist_trans.h"

using Input   = InputLayer<2, 4096>;
using Hidden1 = DenseLayer<Input, 4096>;
using Hidden2 = DenseLayer<Hidden1, 4096>;
using Network = SoftmaxLayer<OutputLayer<Input, 1>>;

int main(int, char **) {
    Network net;
    net.ResetWeights();
    uint8_t *input = new uint8_t[2]{0};
    std::random_device rnd;
    double diff[1];
    double lr = 0.001;

    for (int i = 0; i < 1000; i++) {
        input[0] = rnd() % 1;
        input[1] = rnd() % 1;

        const double *pred = net.Forward(input);
        double teach = input[0] ^ input[1];

        double d = (teach - *pred);
        diff[0]  = lr * d;
        net.Backward(diff);
        std::cout << d << std::endl;
    }

    return 0;
}
