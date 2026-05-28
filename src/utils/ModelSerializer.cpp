#include "ModelSerializer.h"
#include "core/layers/DenseLayer.h"
#include "core/activations/ReLU.h"
#include "core/activations/Sigmoid.h"
#include "core/activations/Tanh.h"
#include "core/activations/Softmax.h"

#include <QDebug>
#include <fstream>
#include <stdexcept>

namespace NN {

    static constexpr uint32_t MAGIC = 0x4E4E564D; // "NNVM"
    static constexpr uint32_t VERSION = 1;

    static void writeString(std::ofstream& f, const std::string& s) {
        uint32_t len = (uint32_t)s.size();
        f.write(reinterpret_cast<const char*>(&len), sizeof(len));
        f.write(s.data(), len);
    }

    static std::string readString(std::ifstream& f) {
        uint32_t len = 0;
        f.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string s(len, '\0');
        f.read(s.data(), len);
        return s;
    }

    static void writeMatrix(std::ofstream& f, const Eigen::MatrixXd& m) {
        int32_t rows = (int32_t)m.rows();
        int32_t cols = (int32_t)m.cols();
        f.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        f.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
        f.write(reinterpret_cast<const char*>(m.data()), rows * cols * sizeof(double));
    }

    static Eigen::MatrixXd readMatrix(std::ifstream& f) {
        int32_t rows = 0, cols = 0;
        f.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        f.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        Eigen::MatrixXd m(rows, cols);
        f.read(reinterpret_cast<char*>(m.data()), rows * cols * sizeof(double));
        return m;
    }

    static void writeVector(std::ofstream& f, const Eigen::VectorXd& v) {
        int32_t size = (int32_t)v.size();
        f.write(reinterpret_cast<const char*>(&size), sizeof(size));
        f.write(reinterpret_cast<const char*>(v.data()), size * sizeof(double));
    }

    static Eigen::VectorXd readVector(std::ifstream& f) {
        int32_t size = 0;
        f.read(reinterpret_cast<char*>(&size), sizeof(size));
        Eigen::VectorXd v(size);
        f.read(reinterpret_cast<char*>(v.data()), size * sizeof(double));
        return v;
    }

    void ModelSerializer::Save(const Models::SequentialModel& model,
        const App::TrainingParams& params,
        const std::filesystem::path& path)
    {
        std::ofstream f(path, std::ios::binary);
        if (!f.is_open())
            throw std::runtime_error("Cannot open file for writing: " + path.string());

        // заголовок
        f.write(reinterpret_cast<const char*>(&MAGIC), sizeof(MAGIC));
        f.write(reinterpret_cast<const char*>(&VERSION), sizeof(VERSION));
        writeString(f, params.datasetType.toStdString()); // "MNIST", "Iris", "CSV"
        f.write(reinterpret_cast<const char*>(&params.inputSize), sizeof(int));
        f.write(reinterpret_cast<const char*>(&params.outputSize), sizeof(int));

        // параметры обучения
        f.write(reinterpret_cast<const char*>(&params.learningRate), sizeof(double));
        f.write(reinterpret_cast<const char*>(&params.epochs), sizeof(int));
        f.write(reinterpret_cast<const char*>(&params.batchSize), sizeof(int));
        writeString(f, params.lossFunction.toStdString());

        // архитектура скрытых слоёв
        int32_t hiddenCount = (int32_t)params.hiddenLayers.size();
        f.write(reinterpret_cast<const char*>(&hiddenCount), sizeof(hiddenCount));
        for (auto& [size, activation] : params.hiddenLayers) {
            f.write(reinterpret_cast<const char*>(&size), sizeof(int));
            writeString(f, activation.toStdString());
        }

        // веса слоёв
        int32_t layerCount = (int32_t)model.GetLayersCount();
        f.write(reinterpret_cast<const char*>(&layerCount), sizeof(layerCount));

        for (int i = 0; i < layerCount; ++i) {
            auto* layer = dynamic_cast<Layers::DenseLayer*>(model.GetLayer(i));
            if (!layer)
                throw std::runtime_error("Only DenseLayer is supported for serialization");

            writeString(f, layer->GetActivationName());
            writeMatrix(f, layer->GetW());
            writeVector(f, layer->GetB());
        }

        f.close();
    }

    std::pair<std::unique_ptr<Models::SequentialModel>, App::TrainingParams>
        ModelSerializer::Load(const std::filesystem::path& path)
    {
        //qDebug() << "Load: opening" << QString::fromStdString(path.string());

        std::ifstream f(path, std::ios::binary);
        if (!f.is_open())
            throw std::runtime_error("Cannot open file: " + path.string());

        App::TrainingParams params;

        // заголовок
        uint32_t magic = 0, version = 0;
        f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        f.read(reinterpret_cast<char*>(&version), sizeof(version));
        //qDebug() << "magic:" << magic << "version:" << version;

        if (magic != MAGIC)
            throw std::runtime_error("Invalid model file format");
        if (version != VERSION)
            throw std::runtime_error("Unsupported model version");

        // версия датасета
        std::string savedDataset = readString(f);
        int savedInputSize = 0, savedOutputSize = 0;
        f.read(reinterpret_cast<char*>(&savedInputSize), sizeof(int));
        f.read(reinterpret_cast<char*>(&savedOutputSize), sizeof(int));
        params.datasetType = QString::fromStdString(savedDataset);
        params.inputSize = savedInputSize;
        params.outputSize = savedOutputSize;

        // параметры обучения
        f.read(reinterpret_cast<char*>(&params.learningRate), sizeof(double));
        f.read(reinterpret_cast<char*>(&params.epochs), sizeof(int));
        f.read(reinterpret_cast<char*>(&params.batchSize), sizeof(int));
        //qDebug() << "lr:" << params.learningRate << "epochs:" << params.epochs;

        params.lossFunction = QString::fromStdString(readString(f));
        //qDebug() << "loss:" << params.lossFunction;

        // архитектура скрытых слоёв
        int32_t hiddenCount = 0;
        f.read(reinterpret_cast<char*>(&hiddenCount), sizeof(hiddenCount));
        //qDebug() << "hiddenCount:" << hiddenCount;

        if (hiddenCount < 0 || hiddenCount > 100)
            throw std::runtime_error("Invalid hidden layer count");

        for (int i = 0; i < hiddenCount; ++i) {
            int size = 0;
            f.read(reinterpret_cast<char*>(&size), sizeof(int));
            QString activation = QString::fromStdString(readString(f));
           //qDebug() << "hidden layer" << i << "size:" << size << "act:" << activation;
            params.hiddenLayers.append({ size, activation });
        }

        // веса слоёв
        int32_t layerCount = 0;
        f.read(reinterpret_cast<char*>(&layerCount), sizeof(layerCount));
        //qDebug() << "layerCount:" << layerCount;

        if (layerCount <= 0 || layerCount > 100)
            throw std::runtime_error("Invalid layer count");

        auto model = std::make_unique<Models::SequentialModel>();

        for (int i = 0; i < layerCount; ++i) {
            //qDebug() << "Loading layer" << i;

            std::string activationName = readString(f);
            //qDebug() << "activation:" << QString::fromStdString(activationName);

            Eigen::MatrixXd W = readMatrix(f);
            Eigen::VectorXd b = readVector(f);

            std::unique_ptr<Activation::Activation> activation;
            if (activationName == "ReLU")    activation = std::make_unique<Activation::ReLU>();
            else if (activationName == "Sigmoid") activation = std::make_unique<Activation::Sigmoid>();
            else if (activationName == "Tanh")    activation = std::make_unique<Activation::Tanh>();
            else if (activationName == "Softmax") activation = std::make_unique<Activation::Softmax>();
            else                                  activation = std::make_unique<Activation::ReLU>();

            auto layer = std::make_unique<Layers::DenseLayer>(
                (int)W.cols(), (int)W.rows(),
                Math::InitType::He,
                std::move(activation)
            );

            layer->SetW(W);
            layer->SetB(b);

            model->AddLayer(std::move(layer));
            //qDebug() << "Layer" << i << "loaded OK";
        }

       // qDebug() << "Load complete";
        f.close();
        return { std::move(model), params };
    }

} // namespace NN