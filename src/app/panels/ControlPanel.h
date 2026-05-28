#pragma once

#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QLabel>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QPushButton>
#include <QScrollArea>
#include <QFrame>
#include <QTimer>
#include <QVector>
#include <QEvent>

namespace App
{

struct TrainingParams {
    QVector<QPair<int, QString>> hiddenLayers;
    double learningRate = 0.01;
    int    epochs = 5;
    int    batchSize = 32;
    QString lossFunction = "CrossEntropy";
    QString datasetType = "MNIST";  // "MNIST", "Iris", "CSV"
    QString csvPath = "";
    int inputSize = 784;
    int outputSize = 10;
};

class ControlPanel : public QWidget {
    Q_OBJECT

public:
    explicit ControlPanel(QWidget* parent = nullptr);

    TrainingParams getParams() const;
    void setTrainingState(bool running);
    void setParams(const App::TrainingParams& params);
    void setSaveEnabled(bool enabled) { saveBtn->setEnabled(enabled); }
    void setDataset(const QString& dataset);

signals:
    void startRequested();
    void stopRequested();
    void resetRequested();
    void saveRequested();
    void loadRequested();
    void datasetChanged(const QString& dataset);
    void settingsRequested();

private slots:
    void addLayer();
    void removeLayer();
    void removeLastLayer();

private:
    void setupUI();
    void setupArchitectureGroup();
    void setupButtons();

    QGroupBox*   archGroup    = nullptr;
    QVBoxLayout* layersLayout = nullptr;
    QPushButton* addLayerBtn  = nullptr;

    struct LayerRow {
        QWidget*    widget     = nullptr;
        QSpinBox*   sizeBox    = nullptr;
        QComboBox*  activation = nullptr;
    };
    QVector<LayerRow> layerRows;

    QDoubleSpinBox* lrBox        = nullptr;
    QSpinBox*       epochsBox    = nullptr;
    QSpinBox*       batchSizeBox = nullptr;
    QComboBox*      lossBox      = nullptr;

    QPushButton* startBtn    = nullptr;
    QPushButton* stopBtn     = nullptr;
    QPushButton* resetBtn    = nullptr;
    QPushButton* saveBtn     = nullptr;
    QPushButton* loadBtn     = nullptr;
    QPushButton* settingsBtn = nullptr;

    QComboBox* datasetBox = nullptr;
    QPushButton* loadCsvBtn = nullptr;
    QLabel* csvPathLabel = nullptr;
    QString csvPath = "";

    QWidget* buildParamsWidget();
    QWidget* buildDatasetWidget();
    QFrame*  makeSeparator();

    QLabel* inputLabel  = nullptr;
    QLabel* outputLabel = nullptr;

    QGroupBox* datasetGroup = nullptr;
    QLabel*    hiddenLabel  = nullptr;
    QLabel*    paramsLabel  = nullptr;
    QLabel*    lrLabel      = nullptr;
    QLabel*    epochsLabel  = nullptr;
    QLabel*    batchLabel   = nullptr;
    QLabel*    lossLabel    = nullptr;

    QTimer* pulseTimer = nullptr;
    bool    pulseState = false;

    void retranslateUi();

protected:
    void changeEvent(QEvent* e) override;
};

} // namespace App
