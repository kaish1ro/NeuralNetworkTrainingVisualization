// Нужен ли?????

/*#pragma once

#include <Eigen/Dense>

namespace NN::Math
{

    inline Eigen::MatrixXd Sigmoid(const Eigen::MatrixXd& m)
    {
        return m.unaryExpr([](double x)
            {
                return 1.0 / (1.0 + std::exp(-x));
            });
    }

    inline Eigen::MatrixXd ReLU(const Eigen::MatrixXd& m)
    {
        return m.cwiseMax(0.0);
    }

}
*/