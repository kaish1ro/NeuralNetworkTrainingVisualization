#include "Trainer.h"

namespace NN::Training
{

	Trainer::Trainer(Models::SequentialModel* model,
		Loss::Loss* loss,
		double learningRate,
		int epochs,
		int batchSize)
		: model(model),
		loss(loss),
		learningRate(learningRate),
		epochs(epochs),
		batchSize(batchSize) {}

	void Trainer::TrainEpoch(const std::vector<Datasets::Batch>& batches)
	{
		for (auto& batch : batches) {
			for (auto& sample : batch.samples) {
				model->Train(sample.input, sample.target, *loss, learningRate);
			}
		}
	}

	double Trainer::ComputeLoss(Datasets::Dataset& dataset)
	{
		double totalLoss = 0.0;
		for (size_t i = 0; i < dataset.Size(); ++i) {
			auto sample = dataset.GetSample(i);
			auto output = model->Predict(sample.input);
			totalLoss += loss->Forward(output, sample.target);
		}
		return totalLoss / dataset.Size();
	}

	void Trainer::Train(Datasets::Dataset& dataset)
	{
		for (int i = 0; i < epochs; ++i) {
			dataset.Shuffle();
			auto batches = dataset.GetBatches(batchSize);
			TrainEpoch(batches);

			double accuracy = ComputeAccuracy(dataset);
			double lossValue = ComputeLoss(dataset);

			Metrics metrics;
			metrics.epoch = i + 1;
			metrics.accuracy = accuracy;
			metrics.loss = lossValue;
			history.Add(metrics);

			if (onEpochEnd)
				onEpochEnd(i + 1, metrics.loss);
		}
	}

	double Trainer::ComputeAccuracy(Datasets::Dataset& dataset)
	{
		int correct = 0;

		for (size_t i = 0; i < dataset.Size(); ++i) {
			auto sample = dataset.GetSample(i);
			Eigen::VectorXd output = model->Predict(sample.input);

			int predicted = 0;
			output.maxCoeff(&predicted);

			int actual = 0;
			sample.target.maxCoeff(&actual);

			if (predicted == actual)
				correct++;
		}

		return (double)correct / dataset.Size();
	}

	const TrainingHistory& Trainer::GetHistory() const {
		return this->history;
	}

	void Trainer::SetOnEpochEnd(std::function<void(int, double)> callback) {
		onEpochEnd = std::move(callback);
	}

} // namespace NN::Training