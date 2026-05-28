#pragma once

#include <QVector>
#include <QMetaType>

namespace App {

struct MisclassifiedSample {
    QVector<double> input;
    int actual    = 0;
    int predicted = 0;
};

struct AnalysisResult {
    QVector<QVector<int>>        confusionMatrix;
    QVector<MisclassifiedSample> misclassified;
    int  numClasses     = 0;
    bool hasMnistImages = false;
};

} // namespace App

Q_DECLARE_METATYPE(App::AnalysisResult)
