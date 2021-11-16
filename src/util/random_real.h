#ifndef RANDOM_REAL_H_
#define RANDOM_REAL_H_

#include <random>

std::random_device rnd;
std::mt19937 mt(rnd());
std::uniform_real_distribution<double> rnd_prob01(0.0, 1.0);

#endif