#pragma once

#include "Loss.h"

namespace NN::Loss 
{

	class MSELoss : public Loss {
	public:
		double Forward(const Eigen::VectorXd& output,
			 const Eigen::VectorXd& target) const override {
			return 0.5 * (target - output).squaredNorm();
		}

		Eigen::VectorXd Backward(const Eigen::VectorXd& output,
			const Eigen::VectorXd& target) const override {
			return output - target;
		}

		std::string Name() const override {
			return "MSE";
		}
	};

}  // namespace NN::Loss