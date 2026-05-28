#include "TrainingWorker.h"
#include "datasets/CSVLoader.h"
#include "core/math/Init.h"

namespace App {

TrainingWorker::TrainingWorker(const TrainingParams& params, QObject* parent)
    : QThread(parent), params(params)
{
}

void TrainingWorker::requestStop()
{
    stopRequested = true;
}

void TrainingWorker::run()
{
    try {
        stopRequested = false;

        std::unique_ptr<NN::Datasets::Dataset> dataset;

        if (params.datasetType == "MNIST") {
            dataset = std::make_unique<NN::Datasets::MNISTLoader>(
                "resources/datasets/mnist/train-images.idx3-ubyte",
                "resources/datasets/mnist/train-labels.idx1-ubyte"
            );
            params.inputSize = 784;
            params.outputSize = 10;
        }
        else if (params.datasetType == "Iris") {
            dataset = std::make_unique<NN::Datasets::CSVLoader>(
                "resources/datasets/iris/iris.csv"
            );
            params.inputSize = 4;
            params.outputSize = 3;
        }
        else if (params.datasetType == "Custom CSV...") {
            if (params.csvPath.isEmpty()) {
                emit trainingFinished();
                return;
            }
            dataset = std::make_unique<NN::Datasets::CSVLoader>(
                std::filesystem::path(params.csvPath.toStdWString()));
            auto* csv = dynamic_cast<NN::Datasets::CSVLoader*>(dataset.get());
            params.inputSize = csv->InputSize();
            params.outputSize = csv->OutputSize();
        }

        buildModel();
        emit architectureReady(model->GetLayerSizes());

        NN::Training::Trainer trainer(
            model.get(),
            loss.get(),
            params.learningRate,
            params.epochs,
            params.batchSize
        );

        //qDebug() << "Model input size:" << model->GetLayer(0)->InputSize();
        //qDebug() << "Dataset input size:" << dataset->GetSample(0).input.size();

        for (int i = 0; i < params.epochs; ++i) {
            if (stopRequested)
                break;
      
            auto batches = dataset->GetBatches(params.batchSize);

            int batchNum = 0;
            int totalBatches = (int)batches.size();
            for (auto& batch : batches) {
                if (stopRequested) break;
                for (auto& sample : batch.samples)
                    model->Train(sample.input, sample.target, *loss, params.learningRate);

                if (++batchNum % 50 == 0)  // каждые 50 батчей
                    emit batchProgress(batchNum, totalBatches);
            }

            if (stopRequested) break;

            double lossValue = trainer.ComputeLoss(*dataset);
            double accuracy = trainer.ComputeAccuracy(*dataset);
            emit epochFinished(i + 1, lossValue, accuracy);
        }

        {
            const size_t analysisSize = std::min(dataset->Size(), (size_t)2000);
            AnalysisResult result;
            result.numClasses     = params.outputSize;
            result.hasMnistImages = (params.datasetType == "MNIST");
            result.confusionMatrix =
                QVector<QVector<int>>(params.outputSize, QVector<int>(params.outputSize, 0));

            for (size_t i = 0; i < analysisSize; ++i) {
                auto sample = dataset->GetSample(i);
                auto pred   = model->Predict(sample.input);

                Eigen::Index actual, predicted;
                sample.target.maxCoeff(&actual);
                pred.maxCoeff(&predicted);

                result.confusionMatrix[(int)actual][(int)predicted]++;

                if (actual != predicted
                    && result.misclassified.size() < 20
                    && result.hasMnistImages)
                {
                    MisclassifiedSample s;
                    s.actual    = (int)actual;
                    s.predicted = (int)predicted;
                    s.input     = QVector<double>(
                        sample.input.data(),
                        sample.input.data() + sample.input.size());
                    result.misclassified.push_back(s);
                }
            }
            emit analysisReady(result);
        }

        //qDebug() << "Emitting trainingFinished";
        emit trainingFinished();
    }
    catch (const std::exception& e) {
        //qDebug() << "Training error:" << e.what();
        emit trainingFinished();
    }
    catch (...) {
        //qDebug() << "Unknown error in TrainingWorker";
        emit trainingFinished();
    }
}

void TrainingWorker::buildModel()
{
    model = std::make_unique<NN::Models::SequentialModel>();

    int prevSize = params.inputSize;
    for (auto& [size, activation] : params.hiddenLayers) {
        NN::Math::InitType initType = (activation == "ReLU")
            ? NN::Math::InitType::He
            : NN::Math::InitType::Xavier;

        model->AddLayer(std::make_unique<NN::Layers::DenseLayer>(
            prevSize, size,
            initType,
            makeActivation(activation)
        ));
        prevSize = size;
    }

    model->AddLayer(std::make_unique<NN::Layers::DenseLayer>(prevSize, params.outputSize,
        NN::Math::InitType::Xavier,
        std::make_unique<NN::Activation::Softmax>()
    ));

    if (params.lossFunction == "CrossEntropy")
        loss = std::make_unique<NN::Loss::CrossEntropyLoss>();
    else
        loss = std::make_unique<NN::Loss::MSELoss>();
}

std::unique_ptr<NN::Activation::Activation>
TrainingWorker::makeActivation(const QString& name)
{
    if (name == "Sigmoid")
        return std::make_unique<NN::Activation::Sigmoid>();
    if (name == "Tanh")
        return std::make_unique<NN::Activation::Tanh>();
    return std::make_unique<NN::Activation::ReLU>(); // default
}

} // namespace App
