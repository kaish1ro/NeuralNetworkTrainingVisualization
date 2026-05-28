#pragma once
#include "Activation.h"

namespace NN::Activation
{

    class Softmax : public Activation {
    public:
        bool NeedsDerivative() const override { return false; }

        void Forward(Eigen::VectorXd& v) const override {
            v = (v.array() - v.maxCoeff()).exp();
            v /= v.sum();
        }

        void ApplyDerivative(Eigen::VectorXd& v) const override {
            // градиент считаетс€ в CrossEntropyLoss, не здесь.
            // пр€мой вызов этого метода - ошибка.
            throw std::logic_error(
                "Softmax::Backward called directly Ч gradient is computed in CrossEntropyLoss"
            );
        }

        static Eigen::MatrixXd Jacobian(const Eigen::VectorXd& s) {
            Eigen::MatrixXd J = -s * s.transpose();
            J.diagonal() += s;
            return J;
        }

        std::string Name() const override { return "Softmax"; }
    };

}