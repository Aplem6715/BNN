#ifndef NORMALIZATION_H_INCLUDED_
#define NORMALIZATION_H_INCLUDED_

#include <algorithm>
#include "../util/random_real.h"
#include "../net_common.h"

template <typename PrevLayer_t, int MomentumMilli = 900 /*double渡せないので1000倍して整数で*/>
class BatchNormalization
{
public:
	static constexpr int kSettingOutDim = PrevLayer_t::kSettingOutDim;
	// 入力次元数
	static constexpr int kSettingInDim = PrevLayer_t::kSettingOutDim;

private:
	static constexpr double kLearningRate = 0.001;
	static constexpr double kSmallValue = 0.00001;
	static constexpr double kMomentum = MomentumMilli / 1000.0;

	PrevLayer_t _prevLayer;
	RealType *_inputBufferPtr;
	RealType _outputBuffer[kSettingOutDim] = {0};

	// スケール補正
	double _gamma[kSettingInDim] = {0};
	// バイアス
	double _beta[kSettingInDim] = {0};

	// 訓練用保持変数
	double _normed_in[BATCH_SIZE][kSettingInDim] = {0};
	double _centered_in[BATCH_SIZE][kSettingInDim] = {0};
	double _std[kSettingInDim] = {0};

	// 訓練済み平均
	double _trained_mean[kSettingInDim] = {0};
	// 訓練済み分散
	double _trained_var[kSettingInDim] = {0};

#pragma region Train
	RealType _outputBatchBuffer[BATCH_SIZE * kSettingOutDim] = {0};
	double _grads[BATCH_SIZE * kSettingInDim];
#pragma endregion

public:
	const RealType *Forward(const uint8_t *netInput)
	{
		const RealType *input = _prevLayer.Forward(netInput);

		for (int i = 0; i < kSettingInDim; ++i)
		{
			double x = input[i];
			double x_hat = (x - _trained_mean[i]) / _trained_var[i];
			_outputBuffer[i] = _gamma[i] * x_hat + _beta[i];
		}
		return _outputBuffer;
	}

	void ResetWeight()
	{
		for (int i = 0; i < kSettingInDim; ++i)
		{
			_gamma[i] = 1;
			_beta[i] = 0;
		}
		_prevLayer.ResetWeight();
	}

#pragma region Train
	RealType *BatchForward(const uint8_t *netInput)
	{
		_inputBufferPtr = _prevLayer.BatchForward(netInput);

		for (int i = 0; i < kSettingInDim; ++i)
		{
			// バッチ平均を計算
			double mu = 0;
			for (int b = 0; b < BATCH_SIZE; b++)
			{
				mu += _inputBufferPtr[i + b * kSettingInDim];
			}
			mu /= BATCH_SIZE;

			// 中央シフト入力値と バッチ分散>偏差を計算
			double var = 0;
			for (int b = 0; b < BATCH_SIZE; b++)
			{
				const int idx = b * kSettingInDim + i;
				const double xc = _inputBufferPtr[idx] - mu;
				_centered_in[b][i] = xc;
				var += xc * xc;
			}
			var /= BATCH_SIZE;
			_std[i] = std::sqrt(var + kSmallValue);

			// 入力値をノーマライズ
			for (int b = 0; b < BATCH_SIZE; b++)
			{
				double normed = _centered_in[b][i] / _std[i];
				_normed_in[b][i] = normed;

				// 出力をセット
				_outputBatchBuffer[b * kSettingInDim + i] = _gamma[i] * normed + _beta[i];
			}

			// 平均・分散を更新
			_trained_mean[i] = kMomentum * _trained_mean[i] + (1 - kMomentum) * mu;
			_trained_var[i] = kMomentum * _trained_var[i] + (1 - kMomentum) * var;
		}

		return _outputBatchBuffer;
	}

	void BatchBackward(const double *nextLayerGrads)
	{
		for (int i = 0; i < kSettingInDim; ++i)
		{
			double d_beta = 0;
			double d_gamma = 0;
			double d_std = 0;
			double d_centered_in;
			for (int b = 0; b < BATCH_SIZE; b++)
			{
				int idx = i + b * kSettingInDim;
				double grad = nextLayerGrads[idx];
				d_beta += grad;
				d_gamma += _normed_in[b][i] * grad;

				double d_normed_in = _gamma[i] * grad;
				d_centered_in = d_normed_in / _std[i];
				d_std -= (d_normed_in * _centered_in[b][i]) / (_std[i] * _std[i]);
			}

			double d_mu = 0;
			for (int b = 0; b < BATCH_SIZE; b++)
			{
				double d_var = 0.5 * d_std / _std[i];
				d_centered_in += (2.0 / BATCH_SIZE) * _centered_in[b][i] * d_var;
				d_mu += d_centered_in;
			}

			for (int b = 0; b < BATCH_SIZE; b++)
			{
				_grads[b * kSettingInDim + i] = d_centered_in - d_mu / BATCH_SIZE;
			}

			// _gamma[i] += kLearningRate * d_gamma;
			// _beta[i] += kLearningRate * d_beta;
		}
		_prevLayer.BatchBackward(_grads);
	}

#pragma endregion
};
#endif