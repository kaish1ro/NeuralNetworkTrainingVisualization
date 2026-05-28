#pragma once

#include <QMainWindow>
#include <QSplitter>
#include <QSettings>

#include "panels/ControlPanel.h"
#include "panels/MetricsPanel.h"
#include "utils/ModelSerializer.h"
#include "widgets/NeuralNetworkView.h"
#include "threading/TrainingWorker.h"

namespace App
{

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() = default;

private slots:
    void onStartTraining();
    void onStopTraining();
    void onResetModel();
    void onEpochFinished(int epoch, double loss, double accuracy);
    void onTrainingFinished();
    bool onSaveModel();
    void onLoadModel();
    void onDatasetChanged(const QString& dataset);
    void onSettingsRequested();

private:
    void setupLayout();
    void setupConnections();
    void buildModel();
    void showPredictWidgets(const TrainingParams& params,
                            NN::Models::SequentialModel* model);
    static DatasetMeta metaFromParams(const TrainingParams& params);

    ControlPanel*       controlPanel  = nullptr;
    NeuralNetworkView*  networkView   = nullptr;
    MetricsPanel*       metricsPanel  = nullptr;

    TrainingWorker*     worker        = nullptr;
    std::unique_ptr<NN::Models::SequentialModel> loadedModel;
    std::unique_ptr<NN::Models::SequentialModel> trainedModel;
    App::TrainingParams lastParams;
};

} // namespace App
