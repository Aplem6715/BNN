#ifndef MNIST_TRANS_H_INCLUDED_
#define MNIST_TRANS_H_INCLUDED_

#include <vector>
#include <cstdint>

/**
 * @brief バイナリデータ列をBNNの入力形式ビット列に変換する
 * 
 * @param raw 入力生データ
 * @param out BNN用ビット列の出力
 * @return true 変換成功
 * @return false 変換失敗
 */
bool TransformBinToBNNInput(std::vector<uint8_t> &raw, std::vector<uint8_t> *out);

#endif