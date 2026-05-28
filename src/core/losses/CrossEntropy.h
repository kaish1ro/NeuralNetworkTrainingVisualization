#pragma once

#include "Loss.h"

namespace NN::Loss
{

	class CrossEntropyLoss : public Loss {
	public:
		double Forward(const Eigen::VectorXd& output,
			const Eigen::VectorXd& target) const override {
			return -(target.array() * (output.array() + 1e-9).log()).sum();
		}

		Eigen::VectorXd Backward(const Eigen::VectorXd& output,
			const Eigen::VectorXd& target) const override {
			return output - target;
		}

		std::string Name() const override {
			return "CrossEntropy";
		}
	};

}  // namespace NN::Loss