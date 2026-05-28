#include "MetricsPanel.h"

namespace App {

MetricsPanel::MetricsPanel(QWidget* parent)
    : QWidget(parent)
{
    setupUI();
}

void MetricsPanel::setupUI()
{
    auto* mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(4, 4, 4, 4);
    mainLayout->setSpacing(0);

    tabs = new QTabWidget(this);
    mainLayout->addWidget(tabs);

    auto* metricsTab    = new QWidget(tabs);
    auto* metricsLayout = new QVBoxLayout(metricsTab);
    metricsLayout->setSpacing(8);
    metricsLayout->setContentsMargins(6, 8, 6, 8);

    epochLabel    = new QLabel("Epoch: —",    metricsTab);
    lossLabel     = new QLabel("Loss: —",     metricsTab);
    accuracyLabel = new QLabel("Accuracy: —", metricsTab);
    metricsLayout->addWidget(epochLabel);
    metricsLayout->addWidget(lossLabel);
    metricsLayout->addWidget(accuracyLabel);

    lossGraph     = new LossGraphWidget("Loss",     metricsTab);
    accuracyGraph = new LossGraphWidget("Accuracy", metricsTab);
    metricsLayout->addWidget(lossGraph);
    metricsLayout->addWidget(accuracyGraph);
    metricsLayout->addStretch();

    digitCanvas = new DigitCanvasWidget(metricsTab);
    digitCanvas->setVisible(false);
    metricsLayout->addWidget(digitCanvas);

    featureInput = new FeatureInputWidget(metricsTab);
    featureInput->setVisible(false);
    metricsLayout->addWidget(featureInput);

    progressLabel = new QLabel("", metricsTab);
    progressLabel->setVisible(false);
    progressLabel->setAlignment(Qt::AlignCenter);
    progressLabel->setStyleSheet("color: gray; font-size: 8pt;");

    progressBar = new QProgressBar(metricsTab);
    progressBar->setVisible(false);
    progressBar->setRange(0, 100);
    progressBar->setValue(0);
    progressBar->setTextVisible(false);
    progressBar->setFixedHeight(6);
    metricsLayout->addWidget(progressLabel);
    metricsLayout->addWidget(progressBar);

    tabs->addTab(metricsTab, tr("Metrics"));

    auto* analysisTab    = new QWidget(tabs);
    auto* analysisLayout = new QVBoxLayout(analysisTab);
    analysisLayout->setContentsMargins(4, 4, 4, 4);
    analysisLayout->setSpacing(0);

    analysisPlaceholder = new QLabel("Train the model\nto see analysis", analysisTab);
    analysisPlaceholder->setAlignment(Qt::AlignCenter);
    analysisPlaceholder->setStyleSheet("color: gray;");
    analysisLayout->addWidget(analysisPlaceholder);

    analysisContent = new QWidget(analysisTab);
    auto* contentLayout = new QVBoxLayout(analysisContent);
    contentLayout->setContentsMargins(0, 0, 0, 0);
    contentLayout->setSpacing(4);

    cmLabel = new QLabel("Confusion Matrix", analysisContent);
    cmLabel->setAlignment(Qt::AlignCenter);
    QFont boldFont = cmLabel->font();
    boldFont.setBold(true);
    cmLabel->setFont(boldFont);
    contentLayout->addWidget(cmLabel);

    confMatrix = new ConfusionMatrixWidget(analysisContent);
    contentLayout->addWidget(confMatrix);

    mcLabel = new QLabel("Misclassified Examples", analysisContent);
    mcLabel->setAlignment(Qt::AlignCenter);
    mcLabel->setFont(boldFont);
    contentLayout->addWidget(mcLabel);

    auto* scrollArea = new QScrollArea(analysisContent);
    scrollArea->setWidgetResizable(true);
    scrollArea->setFrameShape(QFrame::NoFrame);
    misclassified = new MisclassifiedWidget(nullptr);
    scrollArea->setWidget(misclassified);
    contentLayout->addWidget(scrollArea, 1);

    analysisContent->setVisible(false);
    analysisLayout->addWidget(analysisContent);

    tabs->addTab(analysisTab, tr("Analysis"));

    setMinimumWidth(240);
}

void MetricsPanel::addMetrics(int epoch, double loss, double accuracy)
{
    epochLabel->setText(QString("Epoch: %1").arg(epoch));
    lossLabel->setText(QString("Loss: %1").arg(loss, 0, 'f', 6));
    accuracyLabel->setText(QString("Accuracy: %1%").arg(accuracy * 100.0, 0, 'f', 2));
    lossGraph->addValue(loss);
    accuracyGraph->addValue(accuracy);
}

void MetricsPanel::clear()
{
    epochLabel->setText("Epoch: —");
    lossLabel->setText("Loss: —");
    accuracyLabel->setText("Accuracy: —");
    lossGraph->clear();
    accuracyGraph->clear();
    confMatrix->clear();
    misclassified->clear();
    analysisContent->setVisible(false);
    analysisPlaceholder->setVisible(true);
}

void MetricsPanel::nextRun()
{
    lossGraph->nextRun();
    accuracyGraph->nextRun();
}

void MetricsPanel::setPredictor(std::function<Eigen::VectorXd(const Eigen::VectorXd&)> pred)
{
    if (digitCanvas) digitCanvas->setPredictor(pred);
}

void MetricsPanel::showDigitCanvas(bool show)
{
    if (digitCanvas) digitCanvas->setVisible(show);
    if (show && featureInput) featureInput->setVisible(false);
}

void MetricsPanel::setupFeatureInput(
    const DatasetMeta& meta,
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> pred)
{
    if (digitCanvas) digitCanvas->setVisible(false);
    featureInput->configure(meta);
    featureInput->setPredictor(std::move(pred));
    featureInput->setVisible(true);
}

void MetricsPanel::updateProgress(int current, int total)
{
    progressBar->setValue((int)(100.0 * current / total));
    progressLabel->setText(QString("Batch %1 / %2").arg(current).arg(total));
}

void MetricsPanel::resetProgress()
{
    progressBar->setValue(0);
    progressLabel->setText("");
}

void MetricsPanel::setTrainingState(bool running)
{
    progressBar->setVisible(running);
    progressLabel->setVisible(running);
    if (!running) {
        progressBar->setValue(0);
        progressLabel->setText("");
    }
}

void MetricsPanel::hidePredictWidgets()
{
    if (digitCanvas)  digitCanvas->setVisible(false);
    if (featureInput) featureInput->setVisible(false);
}

void MetricsPanel::showAnalysis(const AnalysisResult& result)
{
    confMatrix->setData(result.confusionMatrix, result.numClasses);

    if (result.hasMnistImages && !result.misclassified.isEmpty()) {
        misclassified->setData(result.misclassified);
        misclassified->setVisible(true);
    } else {
        misclassified->setVisible(false);
    }

    analysisPlaceholder->setVisible(false);
    analysisContent->setVisible(true);
}

void MetricsPanel::retranslateUi()
{
    tabs->setTabText(0, tr("Metrics"));
    tabs->setTabText(1, tr("Analysis"));
    epochLabel->setText(tr("Epoch: —"));
    lossLabel->setText(tr("Loss: —"));
    accuracyLabel->setText(tr("Accuracy: —"));
    analysisPlaceholder->setText(tr("Train the model\nto see analysis"));
    cmLabel->setText(tr("Confusion Matrix"));
    mcLabel->setText(tr("Misclassified Examples"));
}

void MetricsPanel::changeEvent(QEvent* e)
{
    if (e->type() == QEvent::LanguageChange)
        retranslateUi();
    QWidget::changeEvent(e);
}

} // namespace App
