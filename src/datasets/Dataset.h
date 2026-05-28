#pragma once

#include "DataSample.h"
#include "Batch.h"

namespace NN::Datasets
{

    class Dataset {
    public:
        virtual ~Dataset() = default;

        virtual size_t Size() const = 0;

        virtual DataSample GetSample(size_t index) const = 0;

        virtual void Shuffle() = 0;

        virtual std::vector<Batch> GetBatches(int batchSize) const = 0;
    };

} // namespace NN::Datasets