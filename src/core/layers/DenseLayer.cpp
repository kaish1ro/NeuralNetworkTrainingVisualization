#include "DenseLayer.h"
#include "core/activations/ReLU.h"

namespace NN::Layers
{

	DenseLayer::DenseLayer(int inputSize, int outputSize,
		NN::Math::InitType initType,
		std::unique_ptr<Activation::Activation> activation)
	{
		W = Eigen::MatrixXd(outputSize, inputSize);
		b = Eigen::VectorXd::Zero(outputSize);

		NN::Math::Init(W, initType);

		if (!activation)
			this->activation = std::make_unique<NN::Activation::ReLU>();
		else
			this->activation = std::move(activation);
	}

	Eigen::VectorXd DenseLayer::Forward(const Eigen::VectorXd& input)
	{
		this->input = input;

		z = W * input + b;
		a = z;

		activation->Forward(a);

		return a;
	}

	Eigen::VectorXd DenseLayer::Backward(const Eigen::VectorXd& grad)
	{
		Eigen::VectorXd delta;

		if (activation->NeedsDerivative()) {
			auto activationDerivative = z;
			activation->ApplyDerivative(activationDerivative);
			delta = grad.array() * activationDerivative.array();
		}
		else {
			delta = grad;
		}

		delta = delta.cwiseMax(-1.0).cwiseMin(1.0);

		dW = delta * input.transpose();
		db = delta;
		return W.transpose() * delta;
	}

	void DenseLayer::UpdateWeights(double learningRate)
	{
		W -= learningRate * dW;
		b -= learningRate * db;
	}

	int DenseLayer::InputSize() const
	{
		return W.cols();
	}

	int DenseLayer::OutputSize() const
	{
		return W.rows();
	}

	const Eigen::MatrixXd& DenseLayer::GetW() const { return W; }
	const Eigen::VectorXd& DenseLayer::GetB() const { return b; }
	void DenseLayer::SetW(const Eigen::MatrixXd& w) { W = w; }
	void DenseLayer::SetB(const Eigen::VectorXd& b) { this->b = b; }

	std::string DenseLayer::GetActivationName() const { return activation->Name(); }

} // namespace NN::Layers