cmake_minimum_required(VERSION 3.19)
project(NeuralNetworkVisualizer)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Qt
find_package(Qt6 REQUIRED COMPONENTS Core Widgets LinguistTools)
qt_standard_project_setup()

# Eigen (header-only)
add_library(Eigen INTERFACE)
target_include_directories(Eigen INTERFACE ${CMAKE_SOURCE_DIR}/external)

set(SOURCES
    src/main.cpp

    src/app/ThemeManager.cpp
    src/app/LanguageManager.cpp
    src/app/SettingsDialog.cpp
    src/app/panels/MetricsPanel.cpp
    src/app/panels/ControlPanel.cpp
    src/app/MainWindow.cpp
    src/app/widgets/NeuralNetworkView.cpp
    src/app/widgets/LossGraphWidget.cpp
    src/app/widgets/DigitCanvasWidget.cpp
    src/app/widgets/ConfusionMatrixWidget.cpp
    src/app/widgets/MisclassifiedWidget.cpp
    src/app/widgets/FeatureInputWidget.cpp

    src/core/layers/DenseLayer.cpp
    src/core/model/SequentialModel.cpp
    src/core/training/Trainer.cpp

    src/datasets/MNISTLoader.cpp
    src/datasets/CSVLoader.cpp

    src/threading/TrainingWorker.cpp

    src/utils/ModelSerializer.cpp
)

set(HEADERS
    src/app/ThemeManager.h
    src/app/LanguageManager.h
    src/app/SettingsDialog.h
    src/app/MainWindow.h
    src/app/panels/ControlPanel.h
    src/app/panels/MetricsPanel.h
    src/app/widgets/LossGraphWidget.h
    src/app/widgets/NeuralNetworkView.h
    src/app/widgets/DigitCanvasWidget.h
    src/app/widgets/ConfusionMatrixWidget.h
    src/app/widgets/MisclassifiedWidget.h
    src/app/widgets/FeatureInputWidget.h
    src/threading/AnalysisResult.h

    src/core/math/Init.h
    src/core/math/MathUtils.h
    src/core/math/Transformer.h
    src/core/activations/Activation.h
    src/core/activations/ReLU.h
    src/core/activations/Sigmoid.h
    src/core/activations/Softmax.h
    src/core/activations/Tanh.h
    src/core/layers/Layer.h
    src/core/layers/DenseLayer.h
    src/core/losses/Loss.h
    src/core/losses/MSELoss.h
    src/core/losses/CrossEntropy.h
    src/core/model/SequentialModel.h
    src/core/training/Trainer.h
    src/core/training/Metrics.h
    src/core/training/TrainingHistory.h

    src/datasets/Dataset.h
    src/datasets/MNISTLoader.h
    src/datasets/CSVLoader.h
    src/datasets/Batch.h
    src/datasets/DataSample.h

    src/threading/TrainingWorker.h

    src/utils/ModelSerializer.h
)

set(UI_FILES
    src/app/MainWindow.ui
)

qt_add_executable(NeuralNetworkVisualizer WIN32
    ${SOURCES}
    ${HEADERS}
    ${UI_FILES}
)

qt_add_resources(NeuralNetworkVisualizer "resources"
    PREFIX "/"
    FILES
        ../resources/styles/dark.qss
        ../resources/styles/light.qss
        ../resources/styles/blue.qss
        docs/README.md
)

qt_add_resources(NeuralNetworkVisualizer "icons"
    PREFIX "/icons"
    FILES ../resources/icons/app.ico
)

qt_add_translations(NeuralNetworkVisualizer
    TS_FILES translations/nn_ru.ts
)

target_link_libraries(NeuralNetworkVisualizer PRIVATE
    Qt6::Core
    Qt6::Widgets
    Eigen
)

target_include_directories(NeuralNetworkVisualizer PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_precompile_headers(NeuralNetworkVisualizer PRIVATE
    <vector>
    <string>
    <cmath>
    <algorithm>
    <memory>
    <Eigen/Dense>
)

set_target_properties(NeuralNetworkVisualizer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build
)
