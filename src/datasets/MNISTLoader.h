#pragma once

#include <vector>
#include <filesystem>
#include "Dataset.h"

namespace NN::Datasets 
{

	class MNISTLoader : public Dataset {
	private:
		std::vector<DataSample> samples;
		std::filesystem::path imagesPath;
		std::filesystem::path labelsPath;

		std::vector<Eigen::VectorXd> LoadImages();
		std::vector<Eigen::VectorXd> LoadLabels();
		void LoadFiles();
	public:
		MNISTLoader(const std::filesystem::path& imagesPath,
			const std::filesystem::path& labelsPath);

		size_t Size() const override;

		DataSample GetSample(size_t index) const override;

		void Shuffle() override;

		std::vector<Batch> GetBatches(int batchSize) const override;
	};

} // namespace NN::Datasets