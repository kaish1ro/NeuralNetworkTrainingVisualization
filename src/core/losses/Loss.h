#pragma once

#include <string>
#include <Eigen/Dense>

// почему...
namespace NN::Loss
{

	class Loss {
	public:
		virtual ~Loss() = default;

		virtual double Forward(const Eigen::VectorXd& output,
								const Eigen::VectorXd& target)
								const = 0;

		virtual Eigen::VectorXd Backward(const Eigen::VectorXd& output,
										const Eigen::VectorXd& target)
										const = 0;

		virtual std::string Name() const = 0;
	};

}  // namespace NN::Loss