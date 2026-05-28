#pragma once
#include "Activation.h"

namespace NN::Activation
{

    class ReLU : public Activation {
    public:
        static double Apply(double x) { return x > 0.0 ? x : 0.0; }
        static double Derivative(double x) { return x > 0.0 ? 1.0 : 0.0; }

        void Forward(Eigen::VectorXd& v) const override {
            v = v.cwiseMax(0.0);
        }

        void ApplyDerivative(Eigen::VectorXd& v) const override {
            v = (v.array() > 0.0).cast<double>();
        }

        std::string Name() const override { return "ReLU"; }
    };

}