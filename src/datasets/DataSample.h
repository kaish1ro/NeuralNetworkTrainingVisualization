#pragma once

#include <Eigen/Dense>

namespace NN::Datasets
{

    struct DataSample
    {
        Eigen::VectorXd input;
        Eigen::VectorXd target;
    };

} // namespace NN::Datasets