#pragma once
#include "Activation.h"

namespace NN::Activation
{

    class Sigmoid : public Activation {
    public:
        static double Apply(double x) {
            return 1.0 / (1.0 + std::exp(-x));
        }

        static double Derivative(double x) {
            double s = Apply(x);
            return s * (1.0 - s);
        }

        void Forward(Eigen::VectorXd& v) const override {
            v = ((-v).array().exp() + 1.0).inverse();
        }

        void ApplyDerivative(Eigen::VectorXd& v) const override {
            v = v.array() * (1.0 - v.array());
        }

        std::string Name() const override { return "Sigmoid"; }
    };

}