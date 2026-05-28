#pragma once

#include <random>
#include <Eigen/Dense>
#include <cmath>

namespace NN::Math
{
    enum class InitType
    {
        He,
        Xavier,
        Random
    };

    inline void Init(Eigen::MatrixXd& m, InitType type)
    {
        static std::mt19937 gen(std::random_device{}());

        switch (type) {
        case InitType::He:
        {
            auto fan_in = m.cols();

            double stddev = std::sqrt(2.0 / fan_in);
            std::normal_distribution<double> dist(0.0, stddev);

            for (int i = 0; i < m.size(); i++)
                m.data()[i] = dist(gen);

            break;
        }

        case InitType::Xavier:
        {
            auto fan_in = m.cols();
            auto fan_out = m.rows();

            double limit = std::sqrt(6.0 / (fan_in + fan_out));
            std::uniform_real_distribution<double> dist(-limit, limit);
            for (int i = 0; i < m.size(); i++)
                m.data()[i] = dist(gen);

            break;
        }
        case InitType::Random:
        {
            std::uniform_real_distribution<double> dist(-1.0, 1.0);

            for (int i = 0; i < m.size(); i++)
                m.data()[i] = dist(gen);

            break;
        }
        }
    
    }

}  // namespace NN::Math
