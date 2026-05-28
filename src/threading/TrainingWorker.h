#pragma once

#include <QThread>
#include <atomic>

#include "app/panels/ControlPanel.h"
#include "core/model/SequentialModel.h"
#include "core/losses/CrossEntropy.h"
#include "core/losses/MSELoss.h"
#include "core/layers/DenseLayer.h"
#include "core/activations/ReLU.h"
#include "core/activations/Sigmoid.h"
#include "core/activations/Tanh.h"
#include "core/activations/Softmax.h"
#include "core/training/Trainer.h"
#include "datasets/MNISTLoader.h"
#include "threading/AnalysisResult.h"

namespace App {

class TrainingWorker : public QThread {
    Q_OBJECT

public:
    explicit TrainingWorker(const TrainingParams& params, QObject* parent = nullptr);
    void requestStop();

    NN::Models::SequentialModel* GetModel() const { return model.get(); }
    std::unique_ptr<NN::Models::SequentialModel> ReleaseModel() {
        return std::move(model);
    }

signals:
    void epochFinished(int epoch, double loss, double accuracy);
    void trainingFinished();
    void batchProgress(int current, int total);
    void analysisReady(App::AnalysisResult result);
    void architectureReady(QVector<int> sizes);

protected:
    void run() override;

private:
    void buildModel();
    std::unique_ptr<NN::Activation::Activation> makeActivation(const QString& name);

    TrainingParams params;
    std::atomic<bool> stopRequested = false;

    std::unique_ptr<NN::Models::SequentialModel> model;
    std::unique_ptr<NN::Loss::Loss>              loss;
};

} // namespace App
