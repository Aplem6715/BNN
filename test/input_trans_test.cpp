
#include "../src/util/mnist_trans.h"
#include "gtest/gtest.h"
#include <bitset>

TEST(Input_Trans, TransformBinToBNNInput8)
{
	std::vector<uint8_t> bnnIn[8];
	std::vector<uint8_t> raw = {
		0b10000000,
		0b11000000,
		0b11100000,
		0b11110000,
		0b11111000,
		0b11111100,
		0b11111110,
		0b11111111,
	};

	EXPECT_TRUE(TransformBinToBNNInput(raw, bnnIn));
	for (int i = 0; i < 8; i++)
	{
		EXPECT_EQ(bnnIn[i][0], ((uint64_t)0b11111111) >> i);
	}

	for (auto b : raw)
	{
		std::cout << std::bitset<8>(b) << std::endl;
	}
	std::cout << "   ||   \n";
	std::cout << "   \\/   \n";
	for (int i = 0; i < 8; i++)
	{
		std::cout << std::bitset<8>(bnnIn[i][0]) << std::endl;
	}
}

TEST(Input_Trans, TransformBinToBNNInput16)
{
	std::vector<uint8_t> bnnIn[8];
	std::vector<uint8_t> raw = {
		0b10000000,
		0b11000000,
		0b11100000,
		0b11110000,
		0b11111000,
		0b11111100,
		0b11111110,
		0b11111111,
		0b10000000,
		0b11000000,
		0b11100000,
		0b11110000,
		0b11111000,
		0b11111100,
		0b11111110,
		0b11111111,
	};

	EXPECT_TRUE(TransformBinToBNNInput(raw, bnnIn));
	for (int i = 0; i < 8; i++)
	{
		EXPECT_EQ(bnnIn[i][0], ((uint64_t)0b11111111) >> i);
		EXPECT_EQ(bnnIn[i][1], ((uint64_t)0b11111111) >> i);
	}

	for (auto b : raw)
	{
		std::cout << std::bitset<8>(b) << std::endl;
	}
	std::cout << "   ||   \n";
	std::cout << "   \\/   \n";
	for (int i = 0; i < 8; i++)
	{
		std::cout << std::bitset<8>(bnnIn[i][0]) << std::endl;
	}
	for (int i = 0; i < 8; i++)
	{
		std::cout << std::bitset<8>(bnnIn[i][1]) << std::endl;
	}
}