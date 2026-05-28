#pragma once
#include "Activation.h"

namespace NN::Activation
{

    class Tanh : public Activation {
    public:
        static double Apply(double x) { return std::tanh(x); }
        static double Derivative(double x) { return 1.0 - x * x; }

        void Forward(Eigen::VectorXd& v) const override {
            v = v.array().tanh();
        }

        void ApplyDerivative(Eigen::VectorXd& v) const override {
            v = 1.0 - v.array().square();
        }

        std::string Name() const override { return "Tanh"; }
    };

}