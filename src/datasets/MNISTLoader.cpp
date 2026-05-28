#include <random>
#include <fstream>
#include <algorithm>
#include <Eigen/Dense>

#include "MNISTLoader.h"

namespace NN::Datasets {

	static int32_t reverseBytes(int32_t n) {
		return ((n & 0xFF) << 24) |
			(((n >> 8) & 0xFF) << 16) |
			(((n >> 16) & 0xFF) << 8) |
			((n >> 24) & 0xFF);
	}

	std::vector<Eigen::VectorXd> MNISTLoader::LoadImages() {
		std::ifstream imagesFile(imagesPath, std::ios::binary);

		if (!imagesFile.is_open())
			throw std::runtime_error("Failed to open file: " + imagesPath.string());

		int32_t magic, count, rows, cols;

		imagesFile.read(reinterpret_cast<char*>(&magic), 4);
		imagesFile.read(reinterpret_cast<char*>(&count), 4);
		imagesFile.read(reinterpret_cast<char*>(&rows), 4);
		imagesFile.read(reinterpret_cast<char*>(&cols), 4);

		magic = reverseBytes(magic);
		count = reverseBytes(count);
		rows = reverseBytes(rows);
		cols = reverseBytes(cols);

		std::vector<uint8_t> imageBuffer(rows * cols);
		std::vector<Eigen::VectorXd> inputs;

		for (size_t i = 0; i < count; i++) {
			imagesFile.read(reinterpret_cast<char*>(imageBuffer.data()), rows * cols);

			Eigen::VectorXd input(rows * cols);
			for (size_t j = 0; j < rows * cols; j++)
				input[j] = imageBuffer[j] / 255.0;

			inputs.push_back(input);
		}

		imagesFile.close();

		return inputs;
	}

	std::vector<Eigen::VectorXd> MNISTLoader::LoadLabels() {
		std::ifstream labelsFile(labelsPath, std::ios::binary);

		if (!labelsFile.is_open())
			throw std::runtime_error("Failed to open file: " + labelsPath.string());

		int32_t magic, count;

		labelsFile.read(reinterpret_cast<char*>(&magic), 4);
		labelsFile.read(reinterpret_cast<char*>(&count), 4);

		magic = reverseBytes(magic);
		count = reverseBytes(count);

		std::vector<uint8_t> labelBuffer(count);
		labelsFile.read(reinterpret_cast<char*>(labelBuffer.data()), count);

		std::vector<Eigen::VectorXd> targets;
		for (size_t i = 0; i < count; i++) {
			Eigen::VectorXd target = Eigen::VectorXd::Zero(10);
			target[labelBuffer[i]] = 1.0;
			targets.push_back(target);
		}

		labelsFile.close();

		return targets;
	}

	void MNISTLoader::LoadFiles() {
		auto inputs = LoadImages();
		auto targets = LoadLabels();

		for (size_t i = 0; i < inputs.size(); i++)
			samples.push_back({ inputs[i], targets[i] });
	}

	MNISTLoader::MNISTLoader(const std::filesystem::path& imagesPath,
		const std::filesystem::path& labelsPath) : imagesPath(imagesPath), labelsPath(labelsPath) {

		if (!std::filesystem::exists(imagesPath))
			throw std::runtime_error("Images file not found: " + imagesPath.string());
		if (!std::filesystem::exists(labelsPath))
			throw std::runtime_error("Labels file not found: " + labelsPath.string());

		LoadFiles();
	}

	size_t MNISTLoader::Size() const {
		return samples.size();
	}

	DataSample MNISTLoader::GetSample(size_t index) const {
		return samples[index];
	}

	void MNISTLoader::Shuffle() {
		std::shuffle(samples.begin(), samples.end(),
			std::mt19937{ std::random_device{}() });
	}

	std::vector<Batch> MNISTLoader::GetBatches(int batchSize) const {
		std::vector<Batch> batches;
		for (size_t i = 0; i < samples.size(); i += batchSize) {
			Batch batch;
			size_t end = std::min(i + batchSize, samples.size());
			batch.samples = std::vector<DataSample>(samples.begin() + i,
				samples.begin() + end);
			batches.push_back(batch);
		}
		return batches;
	}

} // namespace NN::Datasets