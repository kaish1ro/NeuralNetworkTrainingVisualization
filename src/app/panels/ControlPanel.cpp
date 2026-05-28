#include "datasets/CSVLoader.h"
#include "ControlPanel.h"

#include <QMessageBox>
#include <QFileDialog>
#include <QFileInfo>

namespace App {

    ControlPanel::ControlPanel(QWidget* parent)
        : QWidget(parent)
    {
        setupUI();

        // добавить два скрытых слоя по умолчанию
        addLayer();
        addLayer();
        saveBtn->setEnabled(false);
    }

    void ControlPanel::setupUI()
    {
        setupArchitectureGroup();
        setupButtons();

        // ── outer layout: scroll area (stretchy) + pinned button panel (bottom) ──
        auto* outerLayout = new QVBoxLayout(this);
        outerLayout->setContentsMargins(0, 0, 0, 0);
        outerLayout->setSpacing(0);

        auto* scrollArea = new QScrollArea(this);
        scrollArea->setWidgetResizable(true);
        scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        scrollArea->setFrameShape(QFrame::NoFrame);

        auto* scrollContent = new QWidget();
        auto* mainLayout = new QVBoxLayout(scrollContent);
        mainLayout->setSpacing(6);
        mainLayout->setContentsMargins(8, 8, 8, 8);

        // title row: label + ⚙ settings button
        auto* titleRow = new QHBoxLayout();
        auto* title = new QLabel("Neural Network Visualizer", scrollContent);
        QFont f = title->font();
        f.setBold(true);
        f.setPointSize(11);
        title->setFont(f);
        titleRow->addWidget(title, 1);
        titleRow->addWidget(settingsBtn);
        mainLayout->addLayout(titleRow);

        mainLayout->addWidget(makeSeparator());
        mainLayout->addWidget(archGroup);
        mainLayout->addWidget(makeSeparator());
        mainLayout->addWidget(buildDatasetWidget());
        mainLayout->addWidget(makeSeparator());
        paramsLabel = new QLabel("Training Parameters", scrollContent);
        mainLayout->addWidget(paramsLabel);
        mainLayout->addWidget(buildParamsWidget());
        mainLayout->addStretch();

        scrollArea->setWidget(scrollContent);
        outerLayout->addWidget(scrollArea, 1);

        // pinned button panel
        auto* btnWidget = new QWidget(this);
        btnWidget->setObjectName("btnPanel");
        auto* btnLayout = new QVBoxLayout(btnWidget);
        btnLayout->setContentsMargins(8, 8, 8, 8);
        btnLayout->setSpacing(4);
        btnLayout->addWidget(startBtn);
        btnLayout->addWidget(stopBtn);
        btnLayout->addWidget(resetBtn);
        btnLayout->addWidget(saveBtn);
        btnLayout->addWidget(loadBtn);
        outerLayout->addWidget(btnWidget);

        // pulse timer
        pulseTimer = new QTimer(this);
        pulseTimer->setInterval(600);
        connect(pulseTimer, &QTimer::timeout, this, [this]() {
            pulseState = !pulseState;
            startBtn->setProperty("pulse", pulseState);
            startBtn->style()->unpolish(startBtn);
            startBtn->style()->polish(startBtn);
            });

        setMinimumWidth(280);
    }

    void ControlPanel::setupArchitectureGroup()
    {
        archGroup = new QGroupBox("Network Architecture", this);
        auto* layout = new QVBoxLayout(archGroup);

        // фиксированные слои (не редактируются)
        inputLabel = new QLabel("Input layer:  784 neurons", archGroup);
        outputLabel = new QLabel("Output layer: 10 neurons (Softmax)", archGroup);
        inputLabel->setStyleSheet("color: gray;");
        outputLabel->setStyleSheet("color: gray;");

        layout->addWidget(inputLabel);

        hiddenLabel = new QLabel("Hidden layers:", archGroup);
        layout->addWidget(hiddenLabel);

        layersLayout = new QVBoxLayout();
        layout->addLayout(layersLayout);

        addLayerBtn = new QPushButton("+ Add Layer", archGroup);
        connect(addLayerBtn, &QPushButton::clicked, this, &ControlPanel::addLayer);
        layout->addWidget(addLayerBtn);
        layout->addWidget(outputLabel);
    }

    QWidget* ControlPanel::buildParamsWidget()
    {
        auto* w = new QWidget();
        auto* layout = new QVBoxLayout(w);
        layout->setContentsMargins(0, 0, 0, 0);

        // learning rate
        auto* lrRow = new QHBoxLayout();
        lrBox = new QDoubleSpinBox(w);
        lrBox->setRange(0.0001, 1.0);
        lrBox->setSingleStep(0.001);
        lrBox->setDecimals(4);
        lrBox->setValue(0.01);
        lrLabel = new QLabel("Learning rate:", w);
        lrRow->addWidget(lrLabel);
        lrRow->addWidget(lrBox);
        layout->addLayout(lrRow);

        // epochs
        auto* epochRow = new QHBoxLayout();
        epochsBox = new QSpinBox(w);
        epochsBox->setRange(1, 100);
        epochsBox->setValue(5);
        epochsLabel = new QLabel("Epochs:", w);
        epochRow->addWidget(epochsLabel);
        epochRow->addWidget(epochsBox);
        layout->addLayout(epochRow);

        // batch size
        auto* batchRow = new QHBoxLayout();
        batchSizeBox = new QSpinBox(w);
        batchSizeBox->setRange(1, 512);
        batchSizeBox->setValue(32);
        batchLabel = new QLabel("Batch size:", w);
        batchRow->addWidget(batchLabel);
        batchRow->addWidget(batchSizeBox);
        layout->addLayout(batchRow);

        // loss function
        auto* lossRow = new QHBoxLayout();
        lossBox = new QComboBox(w);
        lossBox->addItems({ "CrossEntropy", "MSE" });
        lossLabel = new QLabel("Loss:", w);
        lossRow->addWidget(lossLabel);
        lossRow->addWidget(lossBox);
        layout->addLayout(lossRow);

        return w;
    }

    void ControlPanel::setupButtons()
    {
        startBtn = new QPushButton("▶  Start Training", this);
        stopBtn = new QPushButton("Stop", this);
        resetBtn = new QPushButton("Reset", this);
        saveBtn = new QPushButton("Save Model", this);
        loadBtn = new QPushButton("Load Model", this);
        settingsBtn = new QPushButton("⚙", this);

        startBtn->setObjectName("startBtn");
        stopBtn->setObjectName("stopBtn");
        settingsBtn->setObjectName("settingsBtn");
        settingsBtn->setToolTip("Settings");

        stopBtn->setEnabled(false);

        connect(startBtn, &QPushButton::clicked, this, &ControlPanel::startRequested);
        connect(stopBtn, &QPushButton::clicked, this, &ControlPanel::stopRequested);
        connect(resetBtn, &QPushButton::clicked, this, &ControlPanel::resetRequested);
        connect(saveBtn, &QPushButton::clicked, this, &ControlPanel::saveRequested);
        connect(loadBtn, &QPushButton::clicked, this, &ControlPanel::loadRequested);
        connect(settingsBtn, &QPushButton::clicked, this, &ControlPanel::settingsRequested);
    }

    void ControlPanel::addLayer()
    {
        auto* row = new QWidget(archGroup);
        auto* layout = new QHBoxLayout(row);
        layout->setContentsMargins(0, 0, 0, 0);

        auto* sizeBox = new QSpinBox(row);
        sizeBox->setRange(1, 1024);
        sizeBox->setValue(128);

        auto* activation = new QComboBox(row);
        activation->addItems({ "ReLU", "Sigmoid", "Tanh" });

        auto* removeBtn = new QPushButton("✕", row);
        removeBtn->setObjectName("removeLayerBtn");
        removeBtn->setFixedWidth(24);
        connect(removeBtn, &QPushButton::clicked, this, &ControlPanel::removeLayer);

        layout->addWidget(sizeBox);
        layout->addWidget(activation);
        layout->addWidget(removeBtn);

        layersLayout->addWidget(row);
        layerRows.push_back({ row, sizeBox, activation });
    }

    void ControlPanel::removeLayer()
    {
        if (layerRows.isEmpty())
            return;

        // найти какую строку удалить по sender
        auto* btn = qobject_cast<QPushButton*>(sender());
        for (int i = 0; i < layerRows.size(); ++i) {
            if (layerRows[i].widget->findChild<QPushButton*>() == btn) {
                layerRows[i].widget->deleteLater();
                layerRows.remove(i);
                break;
            }
        }
    }

    TrainingParams ControlPanel::getParams() const
    {
        TrainingParams p;
        p.learningRate = lrBox->value();
        p.epochs = epochsBox->value();
        p.batchSize = batchSizeBox->value();
        p.lossFunction = lossBox->currentText();
        p.datasetType = datasetBox->currentText();
        p.csvPath = csvPath;
        if (p.datasetType == "MNIST") {
            p.inputSize = 784;
            p.outputSize = 10;
        }
        else if (p.datasetType == "Iris") {
            p.inputSize = 4;
            p.outputSize = 3;
        }

        for (auto& row : layerRows)
            p.hiddenLayers.append({ row.sizeBox->value(), row.activation->currentText() });

        return p;
    }

    void ControlPanel::setTrainingState(bool running)
    {
        startBtn->setEnabled(!running);
        stopBtn->setEnabled(running);
        addLayerBtn->setEnabled(!running);
        lrBox->setEnabled(!running);
        epochsBox->setEnabled(!running);
        batchSizeBox->setEnabled(!running);
        lossBox->setEnabled(!running);
        saveBtn->setEnabled(!running);

        if (running) {
            pulseTimer->start();
        }
        else {
            pulseTimer->stop();
            pulseState = false;
            startBtn->setProperty("pulse", false);
            startBtn->style()->unpolish(startBtn);
            startBtn->style()->polish(startBtn);
        }
    }

    QFrame* ControlPanel::makeSeparator()
    {
        auto* sep = new QFrame(this);
        sep->setFrameShape(QFrame::HLine);
        sep->setFrameShadow(QFrame::Plain);
        sep->setFixedHeight(1);
        sep->setStyleSheet("background-color: #2A2A4A; border: none;");
        return sep;
    }

    void ControlPanel::removeLastLayer()
    {
        if (layerRows.isEmpty())
            return;
        layerRows.last().widget->deleteLater();
        layerRows.removeLast();
    }

    void ControlPanel::setParams(const App::TrainingParams& params) {
        lrBox->setValue(params.learningRate);
        epochsBox->setValue(params.epochs);
        batchSizeBox->setValue(params.batchSize);
        lossBox->setCurrentText(params.lossFunction);

        // пересоздать строки скрытых слоёв
        while (!layerRows.isEmpty()) removeLastLayer();
        for (auto& [size, activation] : params.hiddenLayers) {
            addLayer();
            layerRows.last().sizeBox->setValue(size);
            layerRows.last().activation->setCurrentText(activation);
        }
    }

    QWidget* ControlPanel::buildDatasetWidget()
    {
        datasetGroup = new QGroupBox("Dataset", this);
        auto* group = datasetGroup;
        auto* layout = new QVBoxLayout(group);

        datasetBox = new QComboBox(group);
        datasetBox->addItems({ "MNIST", "Iris", "Custom CSV..." });
        layout->addWidget(datasetBox);

        loadCsvBtn = new QPushButton("Browse CSV...", group);
        loadCsvBtn->setVisible(false);
        layout->addWidget(loadCsvBtn);

        csvPathLabel = new QLabel("", group);
        csvPathLabel->setWordWrap(true);
        csvPathLabel->setStyleSheet("color: gray; font-size: 8pt;");
        csvPathLabel->setVisible(false);
        layout->addWidget(csvPathLabel);

        connect(datasetBox, &QComboBox::currentTextChanged,
            [this](const QString& text) {
                bool isCSV = (text == "Custom CSV...");
                loadCsvBtn->setVisible(isCSV);
                csvPathLabel->setVisible(isCSV);

                if (text == "MNIST") {
                    inputLabel->setText("Input layer:  784 neurons");
                    outputLabel->setText("Output layer: 10 neurons (Softmax)");
                }
                else if (text == "Iris") {
                    inputLabel->setText("Input layer:  4 neurons");
                    outputLabel->setText("Output layer: 3 neurons (Softmax)");
                }
                else {
                    inputLabel->setText("Input layer:  ? neurons");
                    outputLabel->setText("Output layer: ? neurons (Softmax)");
                }
                emit datasetChanged(text);
            });

        connect(loadCsvBtn, &QPushButton::clicked, [this]() {
            QString path = QFileDialog::getOpenFileName(
                this, "Open CSV", "", "CSV files (*.csv)");
            if (!path.isEmpty()) {
                try {
                    std::filesystem::path fsPath(path.toStdWString());
                    NN::Datasets::CSVLoader loader(fsPath);
                    csvPath = path;
                    csvPathLabel->setText(QFileInfo(path).fileName());
                    inputLabel->setText(QString("Input layer:  %1 neurons")
                        .arg(loader.InputSize()));
                    outputLabel->setText(QString("Output layer: %1 neurons (Softmax)")
                        .arg(loader.OutputSize()));
                }
                catch (const std::exception& e) {
                    QMessageBox::critical(this, "Error", e.what());
                }
            }
            });

        return group;
    }

    void ControlPanel::setDataset(const QString& dataset) {
        datasetBox->setCurrentText(dataset);
    }

    void ControlPanel::retranslateUi()
    {
        archGroup->setTitle(tr("Network Architecture"));
        hiddenLabel->setText(tr("Hidden layers:"));
        addLayerBtn->setText(tr("+ Add Layer"));
        datasetGroup->setTitle(tr("Dataset"));
        loadCsvBtn->setText(tr("Browse CSV..."));
        paramsLabel->setText(tr("Training Parameters"));
        lrLabel->setText(tr("Learning rate:"));
        epochsLabel->setText(tr("Epochs:"));
        batchLabel->setText(tr("Batch size:"));
        lossLabel->setText(tr("Loss:"));
        startBtn->setText(tr("▶  Start Training"));
        stopBtn->setText(tr("Stop"));
        resetBtn->setText(tr("Reset"));
        saveBtn->setText(tr("Save Model"));
        loadBtn->setText(tr("Load Model"));

        // inputLabel / outputLabel текст зависит от текущего датасета —
        // он обновится при следующем изменении датасета, поэтому триггерим вручную
        const QString ds = datasetBox->currentText();
        if (ds == "MNIST") {
            inputLabel->setText(tr("Input layer:  784 neurons"));
            outputLabel->setText(tr("Output layer: 10 neurons (Softmax)"));
        }
        else if (ds == "Iris") {
            inputLabel->setText(tr("Input layer:  4 neurons"));
            outputLabel->setText(tr("Output layer: 3 neurons (Softmax)"));
        }
        else {
            inputLabel->setText(tr("Input layer:  ? neurons"));
            outputLabel->setText(tr("Output layer: ? neurons (Softmax)"));
        }
    }

    void ControlPanel::changeEvent(QEvent* e)
    {
        if (e->type() == QEvent::LanguageChange)
            retranslateUi();
        QWidget::changeEvent(e);
    }

} // namespace App