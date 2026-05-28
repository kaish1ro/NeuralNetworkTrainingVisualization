#pragma once

#include "TrainingHistory.h"
#include "core/model/SequentialModel.h"
#include "core/losses/Loss.h"
#include "datasets/Dataset.h"
#include "datasets/Batch.h"

#include <functional>

namespace NN::Training 
{

	class Trainer {
	private:
		Models::SequentialModel* model;
		Loss::Loss* loss;
		double learningRate;
		int epochs;
		int batchSize;

		TrainingHistory history;
		std::function<void(int, double)> onEpochEnd;

		void TrainEpoch(const std::vector<Datasets::Batch>& batches);
	public:
		Trainer(Models::SequentialModel* model,
			Loss::Loss* loss,
			double learningRate,
			int epochs,
			int batchSize);

		~Trainer() = default;

		void Train(Datasets::Dataset& dataset);
		const TrainingHistory& GetHistory() const;
		void SetOnEpochEnd(std::function<void(int, double)> callback);

		double ComputeAccuracy(Datasets::Dataset& dataset);
		double ComputeLoss(Datasets::Dataset& dataset);
	};

} // namespace NN::Training