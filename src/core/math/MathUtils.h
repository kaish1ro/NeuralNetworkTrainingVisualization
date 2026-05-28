#pragma once

#include <algorithm>
#include <cmath>

namespace NN::Math
{

    inline double clamp(double x, double min, double max)
    {
        return std::max(min, std::min(x, max));
    }

    inline double sigmoid_scalar(double x)
    {
        return 1.0 / (1.0 + std::exp(-x));
    }

} // namespace NN::Math