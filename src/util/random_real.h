#ifndef RANDOM_REAL_H_
#define RANDOM_REAL_H_

#include <random>

namespace Random
{
	std::random_device rnd;
	std::mt19937 mt(rnd());
	std::uniform_real_distribution<double> rnd_prob01(0.0, 1.0);
}

void RandomSeed(int seed){
	Random::mt = std::mt19937(seed);
}

// [0~1)の実数乱数を取得
double GetRandReal()
{
	return Random::rnd_prob01(Random::mt);
}

uint32_t GetRandUInt(){
	return Random::mt();
}

#endif