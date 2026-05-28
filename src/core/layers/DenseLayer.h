#pragma once

#include "Layer.h"
#include "core/math/Init.h"
#include "core/activations/Activation.h"
#include "core/activations/ReLU.h"

namespace NN::Layers
{

	class DenseLayer : public Layer {
	private:
		Eigen::MatrixXd W;
		Eigen::VectorXd a;
		Eigen::VectorXd z;
		Eigen::VectorXd b;

		Eigen::VectorXd input;  // кэш вход слоя

		Eigen::MatrixXd dW;     // градиент по весам
		Eigen::VectorXd db;		// градиент по bias

		std::unique_ptr<Activation::Activation> activation;
	public:
		DenseLayer(int inputSize, int outputSize,
			NN::Math::InitType initType = NN::Math::InitType::He,
			std::unique_ptr<Activation::Activation> activation = nullptr);

		~DenseLayer() = default;

		Eigen::VectorXd Forward(const Eigen::VectorXd& input) override;
		Eigen::VectorXd Backward(const Eigen::VectorXd& grad) override;

		void UpdateWeights(double learningRate) override;

		int InputSize() const override;
		int OutputSize() const override;

		const Eigen::MatrixXd& GetW() const;
		const Eigen::VectorXd& GetB() const;
		void SetW(const Eigen::MatrixXd& w);
		void SetB(const Eigen::VectorXd& b);
		const Eigen::VectorXd& GetA() const { return a; }

		std::string GetActivationName() const;
	};

}  // namespace NN::Layers