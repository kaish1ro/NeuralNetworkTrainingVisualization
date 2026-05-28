#pragma once

#include <vector>
#include <memory>
#include <Qvector>
#include <Eigen/Dense>

#include "core/layers/Layer.h"
#include "core/losses/Loss.h"

namespace NN::Models {
	class SequentialModel {
	private:
		std::vector<std::unique_ptr<Layers::Layer>> layers;
	public:
		SequentialModel();

		//SequentialModel(std::unique_ptr<Layers::Layer> Layers);

		~SequentialModel() = default;

		void AddLayer(std::unique_ptr<Layers::Layer> Layer);

		Eigen::VectorXd Predict(const Eigen::VectorXd& input);

		void Train(const Eigen::VectorXd& input,
			const Eigen::VectorXd& target,
			Loss::Loss& loss,
			double learningRate);

		double ComputeLoss(const Eigen::VectorXd& input,
			const Eigen::VectorXd& target,
			Loss::Loss& loss);

		Layers::Layer* GetLayer(int index) const;

		int GetLayersCount() const;

		QVector<Eigen::MatrixXd> GetWeights() const;
		QVector<int> GetLayerSizes() const;
		QVector<Eigen::VectorXd> GetActivations() const;
	};

} // namespace NN::Models