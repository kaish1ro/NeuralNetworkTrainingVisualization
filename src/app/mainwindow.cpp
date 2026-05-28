#include "MainWindow.h"
#include "SettingsDialog.h"

#include <QHBoxLayout>
#include <QMessageBox>
#include <QFileDialog>
#include <QDialog>
#include <QDialogButtonBox>
#include <QCheckBox>
#include <QLabel>
#include <QVBoxLayout>
#include <QSettings>
#include "datasets/CSVLoader.h"
#include "datasets/MNISTLoader.h"

namespace App
{

    MainWindow::MainWindow(QWidget* parent)
        : QMainWindow(parent)
    {
        setWindowTitle("Neural Network Visualizer");
        setMinimumSize(1100, 700);
        resize(1280, 760);

        setupLayout();
        setupConnections();
    }

    void MainWindow::setupLayout()
    {
        auto* splitter = new QSplitter(Qt::Horizontal, this);
        splitter->setChildrenCollapsible(false);

        controlPanel = new ControlPanel(this);
        networkView = new NeuralNetworkView(this);
        metricsPanel = new MetricsPanel(this);

        splitter->addWidget(controlPanel);
        splitter->addWidget(networkView);
        splitter->addWidget(metricsPanel);

        splitter->setSizes({ 300, 720, 260 });

        setCentralWidget(splitter);
    }

    void MainWindow::setupConnections()
    {
        connect(controlPanel, &ControlPanel::startRequested, this, &MainWindow::onStartTraining);
        connect(controlPanel, &ControlPanel::stopRequested, this, &MainWindow::onStopTraining);
        connect(controlPanel, &ControlPanel::resetRequested, this, &MainWindow::onResetModel);
        connect(controlPanel, &ControlPanel::saveRequested, this, &MainWindow::onSaveModel);
        connect(controlPanel, &ControlPanel::loadRequested, this, &MainWindow::onLoadModel);
        connect(controlPanel, &ControlPanel::datasetChanged, this, &MainWindow::onDatasetChanged);
        connect(controlPanel, &ControlPanel::settingsRequested, this, &MainWindow::onSettingsRequested);
    }

    void MainWindow::onStartTraining()
    {
        if (worker && worker->isRunning())
            return;

        if (trainedModel) {
            QSettings s;
            const QString dataset = controlPanel->getParams().datasetType;
            const bool skipAll     = s.value("warnings/skipAll", false).toBool();
            const QString skipFor  = s.value("warnings/skipForDataset", "").toString();
            const bool skipThis    = (skipFor == dataset);

            if (!skipAll && !skipThis) {
                QDialog dlg(this);
                dlg.setWindowTitle(tr("Overwrite model?"));

                auto* layout = new QVBoxLayout(&dlg);
                layout->setSpacing(12);

                auto* msg = new QLabel(
                    tr("You have a trained or loaded model.\n"
                       "Starting new training will overwrite it.\n\n"
                       "Save it first?"), &dlg);
                layout->addWidget(msg);

                auto* cbDataset = new QCheckBox(
                    tr("Don't ask again for %1").arg(dataset), &dlg);
                auto* cbAll = new QCheckBox(tr("Don't ask again"), &dlg);
                layout->addWidget(cbDataset);
                layout->addWidget(cbAll);

                auto* buttons = new QDialogButtonBox(&dlg);
                auto* saveBtn_  = buttons->addButton(tr("Save"),      QDialogButtonBox::AcceptRole);
                auto* discardBtn = buttons->addButton(tr("Overwrite"), QDialogButtonBox::DestructiveRole);
                auto* cancelBtn  = buttons->addButton(tr("Cancel"),    QDialogButtonBox::RejectRole);
                layout->addWidget(buttons);

                int result = -1; // -1=cancel, 0=discard, 1=save
                connect(saveBtn_,   &QPushButton::clicked, &dlg, [&]{ result = 1; dlg.accept(); });
                connect(discardBtn, &QPushButton::clicked, &dlg, [&]{ result = 0; dlg.accept(); });
                connect(cancelBtn,  &QPushButton::clicked, &dlg, [&]{ result = -1; dlg.reject(); });

                dlg.exec();

                if (result == -1) return;
                if (result == 1 && !onSaveModel()) return;

                if (cbAll->isChecked())
                    s.setValue("warnings/skipAll", true);
                else if (cbDataset->isChecked())
                    s.setValue("warnings/skipForDataset", dataset);
            } else if (!skipAll && skipThis) {
                // пропуск
            }
        }

        metricsPanel->setTrainingState(true);
        metricsPanel->nextRun();

        auto params = controlPanel->getParams();

        worker = new TrainingWorker(params, nullptr);

        connect(worker, &TrainingWorker::epochFinished,
            this, &MainWindow::onEpochFinished, Qt::QueuedConnection);
        connect(worker, &TrainingWorker::trainingFinished,
            this, &MainWindow::onTrainingFinished, Qt::QueuedConnection);
        connect(worker, &TrainingWorker::finished,
            worker, &QObject::deleteLater,
            Qt::QueuedConnection);
        connect(worker, &TrainingWorker::batchProgress,
            metricsPanel, &MetricsPanel::updateProgress,
            Qt::QueuedConnection);
        connect(worker, &TrainingWorker::analysisReady,
            metricsPanel, &MetricsPanel::showAnalysis,
            Qt::QueuedConnection);
        connect(worker, &TrainingWorker::architectureReady,
            this, [this](const QVector<int>& sizes) {
                networkView->setArchitecture(sizes);
            }, Qt::QueuedConnection);

        controlPanel->setTrainingState(true);
        worker->start();
    }

    void MainWindow::onStopTraining()
    {
        if (worker && worker->isRunning()) {
            worker->requestStop();
            controlPanel->setTrainingState(false);
        }
    }

    void MainWindow::onResetModel()
    {
        onStopTraining();
        metricsPanel->clear();
        networkView->clear();
    }

    void MainWindow::onEpochFinished(int epoch, double loss, double accuracy)
    {
        metricsPanel->resetProgress();
        metricsPanel->addMetrics(epoch, loss, accuracy);

        if (worker && worker->GetModel()) {
            networkView->setWeights(worker->GetModel()->GetWeights());
            networkView->setArchitecture(worker->GetModel()->GetLayerSizes());
            networkView->setActivations(worker->GetModel()->GetActivations());
        }

        networkView->update();
    }

    void MainWindow::onTrainingFinished()
    {
        metricsPanel->setTrainingState(false);
        //qDebug() << "onTrainingFinished called, worker:" << worker;

        if (worker)
            trainedModel = worker->ReleaseModel();

        //qDebug() << "trainedModel:" << trainedModel.get();

        lastParams = controlPanel->getParams();
        controlPanel->setTrainingState(false);
        controlPanel->setSaveEnabled(true);
        worker = nullptr;

        if (trainedModel)
            showPredictWidgets(lastParams, trainedModel.get());
    }

    bool MainWindow::onSaveModel()
    {
        QString path = QFileDialog::getSaveFileName(
            this,
            "Save Model",
            "model.nnvm",
            "Neural Network Model (*.nnvm)"
        );

        if (path.isEmpty())
            return false;

        try {
            NN::ModelSerializer::Save(*trainedModel, lastParams, path.toStdString());
            QMessageBox::information(this, "Saved", "Model saved successfully!");
            return true;
        }
        catch (const std::exception& e) {
            QMessageBox::critical(this, "Error", e.what());
            return false;
        }
    }

    void MainWindow::onLoadModel()
    {
        QString path = QFileDialog::getOpenFileName(
            this, "Load Model", "", "Neural Network Model (*.nnvm)");
        if (path.isEmpty())
            return;

        try {
            auto [model, params] = NN::ModelSerializer::Load(path.toStdString());
            auto currentParams = controlPanel->getParams();

            if (params.datasetType != currentParams.datasetType) {
                auto btn = QMessageBox::warning(this, "Dataset Mismatch",
                    QString("Model was trained on %1 but %2 is selected.\nSwitch to %1?")
                    .arg(params.datasetType).arg(currentParams.datasetType),
                    QMessageBox::Yes | QMessageBox::Cancel);
                if (btn == QMessageBox::Cancel)
                    return;
                controlPanel->setDataset(params.datasetType);
            }

            trainedModel = std::move(model);
            lastParams = params;
            controlPanel->setParams(params);

            networkView->setWeights(trainedModel->GetWeights());
            networkView->setArchitecture(trainedModel->GetLayerSizes());

            showPredictWidgets(params, trainedModel.get());

            QMessageBox::information(this, "Loaded", "Model loaded successfully!");
        }
        catch (const std::exception& e) {
            QMessageBox::critical(this, "Error", e.what());
        }
    }

    void MainWindow::showPredictWidgets(const TrainingParams& params,
                                        NN::Models::SequentialModel* model)
    {
        if (params.datasetType == "MNIST") {
            metricsPanel->showDigitCanvas(true);
            metricsPanel->setPredictor([model](const Eigen::VectorXd& input) {
                return model->Predict(input);
            });
        } else {
            auto meta = metaFromParams(params);
            if (meta.featureNames.isEmpty()) return;
            metricsPanel->setupFeatureInput(meta, [model](const Eigen::VectorXd& input) {
                return model->Predict(input);
            });
        }
    }

    DatasetMeta MainWindow::metaFromParams(const TrainingParams& params)
    {
        DatasetMeta meta;
        try {
            std::filesystem::path path;
            if (params.datasetType == "Iris")
                path = "../resources/datasets/iris/iris.csv";
            else if (params.datasetType == "Custom CSV..." && !params.csvPath.isEmpty())
                path = params.csvPath.toStdWString();
            else
                return meta;

            NN::Datasets::CSVLoader loader(path);

            for (auto& s : loader.GetFeatureNames())
                meta.featureNames.push_back(QString::fromStdString(s));
            for (double v : loader.GetMinVals()) meta.minVals.push_back(v);
            for (double v : loader.GetMaxVals()) meta.maxVals.push_back(v);

            meta.classNames.resize(loader.OutputSize());
            for (auto& [name, idx] : loader.GetLabelMap())
                if (idx < meta.classNames.size())
                    meta.classNames[idx] = QString::fromStdString(name);
        }
        catch (...) {}
        return meta;
    }

    void MainWindow::onDatasetChanged(const QString& /*dataset*/)
    {
        metricsPanel->hidePredictWidgets();
    }

    void MainWindow::onSettingsRequested()
    {
        SettingsDialog dlg(this);
        dlg.exec();
    }

} // namespace App
