#pragma once

#include <QWidget>
#include <QVBoxLayout>
#include <QLabel>
#include <QTabWidget>
#include <QScrollArea>
#include <QProgressBar>
#include <QEvent>
#include <functional>

#include "app/widgets/LossGraphWidget.h"
#include "app/widgets/DigitCanvasWidget.h"
#include "app/widgets/ConfusionMatrixWidget.h"
#include "app/widgets/MisclassifiedWidget.h"
#include "app/widgets/FeatureInputWidget.h"
#include "threading/AnalysisResult.h"

namespace App {

class MetricsPanel : public QWidget {
    Q_OBJECT
public:
    explicit MetricsPanel(QWidget* parent = nullptr);

    void addMetrics(int epoch, double loss, double accuracy);
    void clear();
    void nextRun();
    void setPredictor(std::function<Eigen::VectorXd(const Eigen::VectorXd&)> pred);
    void showDigitCanvas(bool show);
    void setupFeatureInput(const DatasetMeta& meta,
                           std::function<Eigen::VectorXd(const Eigen::VectorXd&)> pred);
    void updateProgress(int current, int total);
    void resetProgress();
    void setTrainingState(bool running);
    void showAnalysis(const AnalysisResult& result);
    void hidePredictWidgets();

private:
    void setupUI();

    QTabWidget* tabs = nullptr;

    QLabel*            epochLabel    = nullptr;
    QLabel*            lossLabel     = nullptr;
    QLabel*            accuracyLabel = nullptr;
    LossGraphWidget*   lossGraph     = nullptr;
    LossGraphWidget*   accuracyGraph = nullptr;
    DigitCanvasWidget*   digitCanvas   = nullptr;
    FeatureInputWidget*  featureInput  = nullptr;
    QProgressBar*      progressBar   = nullptr;
    QLabel*            progressLabel = nullptr;

    ConfusionMatrixWidget* confMatrix           = nullptr;
    MisclassifiedWidget*   misclassified        = nullptr;
    QLabel*                analysisPlaceholder  = nullptr;
    QWidget*               analysisContent      = nullptr;
    QLabel*                cmLabel              = nullptr;
    QLabel*                mcLabel              = nullptr;

    void retranslateUi();

protected:
    void changeEvent(QEvent* e) override;
};

} // namespace App
