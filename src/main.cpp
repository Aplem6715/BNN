#include <iostream>

#include "util/input_trans.h"

int main(int, char**) {
	std::vector<uint8_t> bnnIn[8];
	std::vector<uint8_t> raw = {
		0xff,
		0,
		0,
		0,
		0,
		0,
		0,
		0};

	TransformBinToBNNInput(raw, bnnIn);

	std::cout << bnnIn;
}
