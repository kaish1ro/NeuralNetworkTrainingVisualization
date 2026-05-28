#pragma once

#include "DataSample.h"

#include <vector>

namespace NN::Datasets
{

    struct Batch
    {
        std::vector<DataSample> samples;
        int Size() const { return (int)samples.size(); }
    };

} // namespace NN::Datasets