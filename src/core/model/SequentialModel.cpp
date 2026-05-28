#include "SequentialModel.h"
#include <ranges>

#include "core/layers/DenseLayer.h"

namespace NN::Models
{
	SequentialModel::SequentialModel()
	{
	}

	void SequentialModel::AddLayer(std::unique_ptr<Layers::Layer> Layer) {
		this->layers.push_back(std::move(Layer));
	}

	Eigen::VectorXd SequentialModel::Predict(const Eigen::VectorXd& input)
	{
		Eigen::VectorXd output = input;
		for (auto& layer : layers) {
			output = layer->Forward(output);
		}

		return output;
	}

	void SequentialModel::Train(const Eigen::VectorXd& input,
		const Eigen::VectorXd& target,
		Loss::Loss& loss,
		double learningRate)
	{

		Eigen::VectorXd output = input;
		for (auto& layer : layers) {
			output = layer->Forward(output);
		}

		auto grad = loss.Backward(output, target);

		for (auto& layer : layers | std::views::reverse) {
			grad = layer->Backward(grad);
		}

		for (auto& layer : layers) {
			layer->UpdateWeights(learningRate);
		}
	}

	double SequentialModel::ComputeLoss(const Eigen::VectorXd& output,
		const Eigen::VectorXd& target,
		Loss::Loss& loss) {
		return loss.Forward(output, target);
	}

	Layers::Layer* SequentialModel::GetLayer(int index) const {
		if (index < 0 || index >= (int)layers.size())
			return nullptr;
		return layers[index].get();
	}

	int SequentialModel::GetLayersCount() const {
		return layers.size();
	}

	QVector<Eigen::MatrixXd> SequentialModel::GetWeights() const
	{
		QVector<Eigen::MatrixXd> weights;

		for (int i = 0; i < GetLayersCount(); ++i) {
			auto* layer = dynamic_cast<Layers::DenseLayer*>(GetLayer(i));
			if (layer) weights.append(layer->GetW());
		}

		return weights;
	}

	QVector<int> SequentialModel::GetLayerSizes() const
	{
		QVector<int> sizes;

		for (int i = 0; i < GetLayersCount(); ++i)
			sizes.append(GetLayer(i)->InputSize());
		sizes.append(GetLayer(GetLayersCount() - 1)->OutputSize());

		return sizes;
	}

	QVector<Eigen::VectorXd> SequentialModel::GetActivations() const
	{
		QVector<Eigen::VectorXd> result;
		for (int i = 0; i < GetLayersCount(); ++i) {
			auto* layer = dynamic_cast<Layers::DenseLayer*>(GetLayer(i));
			if (layer) result.append(layer->GetA());
		}
		return result;
	}

} // namespace NN::Models