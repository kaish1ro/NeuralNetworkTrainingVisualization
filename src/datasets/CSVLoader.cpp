#include "CSVLoader.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <stdexcept>

namespace NN::Datasets {

    static std::vector<std::string> splitLine(const std::string& line, char delim = ',')
    {
        std::vector<std::string> tokens;
        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, delim)) {
            token.erase(0, token.find_first_not_of(" \t\r"));
            token.erase(token.find_last_not_of(" \t\r") + 1);
            tokens.push_back(token);
        }
        return tokens;
    }

    static bool isNumber(const std::string& s)
    {
        if (s.empty()) return false;
        char* end = nullptr;
        std::strtod(s.c_str(), &end); // подумать
        return end != s.c_str() && *end == '\0';
    }

    bool CSVLoader::hasHeader(const std::string& firstLine) const
    {
        auto tokens = splitLine(firstLine);

        for (int i = 0; i < (int)tokens.size() - 1; ++i)
            if (!isNumber(tokens[i]))
                return true;

        return false;
    }

    void CSVLoader::LoadFile()
    {
        std::ifstream f(path);
        if (!f.is_open())
            throw std::runtime_error("Cannot open file: " + path.string());

        std::string line;
        std::vector<std::vector<std::string>> rawData;

        if (!std::getline(f, line))
            throw std::runtime_error("CSV file is empty");

        if (hasHeader(line)) {
            auto cols = splitLine(line);
            for (int i = 0; i + 1 < (int)cols.size(); ++i)
                featureNames.push_back(cols[i]);
        } else {
            rawData.push_back(splitLine(line));
        }

        while (std::getline(f, line)) {
            if (line.empty()) continue;
            rawData.push_back(splitLine(line));
        }
        f.close();

        if (rawData.empty())
            throw std::runtime_error("CSV file has no data");

        inputSize = (int)rawData[0].size() - 1;
        if (inputSize <= 0)
            throw std::runtime_error("CSV file must have at least 2 columns");

        if (featureNames.empty())
            for (int i = 0; i < inputSize; ++i)
                featureNames.push_back("Feature " + std::to_string(i + 1));

        for (auto& row : rawData) {
            if (row.empty()) continue;
            const std::string& label = row.back();
            if (labelMap.find(label) == labelMap.end()) {
                int idx = (int)labelMap.size();
                labelMap[label] = idx;
            }
        }
        outputSize = (int)labelMap.size();

        minVals.assign(inputSize, 1e18);
        maxVals.assign(inputSize, -1e18);

        for (auto& row : rawData) {
            for (int i = 0; i < inputSize; ++i) {
                if (i >= (int)row.size()) continue;
                if (!isNumber(row[i])) continue;
                double val = std::stod(row[i]);
                minVals[i] = std::min(minVals[i], val);
                maxVals[i] = std::max(maxVals[i], val);
            }
        }

        samples.reserve(rawData.size());
        for (auto& row : rawData) {
            if ((int)row.size() < inputSize + 1) continue;

            Eigen::VectorXd input(inputSize);
            for (int i = 0; i < inputSize; ++i) {
                double val = isNumber(row[i]) ? std::stod(row[i]) : 0.0;
                double range = maxVals[i] - minVals[i];
                input[i] = (range < 1e-9) ? 0.0 : (val - minVals[i]) / range;
            }

            Eigen::VectorXd target = Eigen::VectorXd::Zero(outputSize);
            target[labelMap[row.back()]] = 1.0;

            samples.push_back({ input, target });
        }

        if (samples.empty())
            throw std::runtime_error("No valid samples found in CSV");
    }

    CSVLoader::CSVLoader(const std::filesystem::path& path)
        : path(path)
    {
        if (!std::filesystem::exists(path))
            throw std::runtime_error("File not found: " + path.string());

        LoadFile();
    }

    size_t CSVLoader::Size() const {
        return samples.size();
    }

    DataSample CSVLoader::GetSample(size_t index) const
    {
        return samples[index];
    }

    void CSVLoader::Shuffle()
    {
        std::shuffle(samples.begin(), samples.end(),
            std::mt19937{ std::random_device{}() });
    }

    std::vector<Batch> CSVLoader::GetBatches(int batchSize) const
    {
        std::vector<Batch> batches;
        for (size_t i = 0; i < samples.size(); i += batchSize) {
            Batch batch;
            size_t end = std::min(i + (size_t)batchSize, samples.size());
            batch.samples = std::vector<DataSample>(
                samples.begin() + i, samples.begin() + end);
            batches.push_back(batch);
        }
        return batches;
    }

} // namespace NN::Datasets