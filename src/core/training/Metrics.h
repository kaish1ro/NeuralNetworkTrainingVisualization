#pragma once

namespace NN::Training 
{

	struct Metrics {
		double loss;
		double accuracy;
		int epoch;
	};

} // namespace NN::Training