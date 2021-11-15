#include <iostream>

#include "layers.h"
#include "util/mnist_trans.h"

using Input   = InputLayer<2, 4096>;
using Hidden1 = DenseLayer<Input, 4096>;
using Hidden2 = DenseLayer<Hidden1, 4096>;
using Network = SoftmaxLayer<OutputLayer<Input, 1>>;

int main(int, char **) {
    Network net;
    net.ResetWeights();
    uint8_t *inputImage = new uint8_t[2]{0};

    auto *pred = net.Forward(inputImage);

    for (int i = 0; i < net.kOutputSize; i++) {
        std::cout << pred[i] << std::endl;
    }

    return 0;
}
