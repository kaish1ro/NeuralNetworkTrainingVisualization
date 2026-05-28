#pragma once

#include "Metrics.h"

#include <vector>

namespace NN::Training 
{

	struct TrainingHistory {
		std::vector<Metrics> metrics;

		void Add(const Metrics& metric) {
			metrics.push_back(metric);
		}

		std::vector<double> GetLossHistory() const
		{
			std::vector<double> lossHistory;

			for (const Metrics& metric : metrics) {
				lossHistory.push_back(metric.loss);
			}

			return lossHistory;
		}

		std::vector<double> GetAccuracyHistory() const
		{
			std::vector<double> accuracyHistory;

			for (const Metrics& metric : metrics) {
				accuracyHistory.push_back(metric.accuracy);
			}

			return accuracyHistory;
		}

		const Metrics& GetLast() const {
			return metrics.back();
		}

		size_t Size() const {
			return metrics.size();
		}
	};

} //namespace NN::Training