#pragma once

#include <map>
#include <vector>
#include <filesystem>

#include "Dataset.h"

namespace NN::Datasets
{

	class CSVLoader : public Dataset {
	private:
		std::vector<DataSample> samples;
		std::filesystem::path path;
		std::map<std::string, int> labelMap;
		std::vector<double> minVals;
		std::vector<double> maxVals;
		std::vector<std::string> featureNames;

		int inputSize = 0;
		int outputSize = 0;

		void LoadFile();

		bool hasHeader(const std::string& firstLine) const;
	public:
		CSVLoader(const std::filesystem::path& path);

		size_t Size() const override;

		DataSample GetSample(size_t index) const override;

		void Shuffle() override;

		std::vector<Batch> GetBatches(int batchSize) const override;

		int InputSize() const { return inputSize; }
		int OutputSize() const { return outputSize; }
		const std::map<std::string, int>&    GetLabelMap()     const { return labelMap; }
		const std::vector<double>&           GetMinVals()      const { return minVals; }
		const std::vector<double>&           GetMaxVals()      const { return maxVals; }
		const std::vector<std::string>&      GetFeatureNames() const { return featureNames; }
	};

} // namespace NN::Datasets