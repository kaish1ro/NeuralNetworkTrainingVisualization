#pragma once

#include <Eigen/Dense>

// «¿ BROOD— Œ≈

namespace NN::Layers
{

	class Layer {
	public:
		virtual ~Layer() = default;

		virtual Eigen::VectorXd Forward(const Eigen::VectorXd& input) = 0;
		virtual Eigen::VectorXd Backward(const Eigen::VectorXd& input) = 0;

		virtual void UpdateWeights(double learningRate) = 0;

		virtual int InputSize()  const = 0;
		virtual int OutputSize() const = 0;
	};

}  // namespace NN::Layers