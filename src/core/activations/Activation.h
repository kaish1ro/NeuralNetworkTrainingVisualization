#pragma once
#include <Eigen/Dense>
#include <string>
#include <stdexcept>

namespace NN::Activation
{

    class Activation {
    public:
        virtual bool NeedsDerivative() const { return true; }

        virtual ~Activation() = default;

        virtual void Forward(Eigen::VectorXd& v) const = 0;

        virtual void ApplyDerivative(Eigen::VectorXd& v) const = 0;

        virtual std::string Name() const = 0;
    };

}