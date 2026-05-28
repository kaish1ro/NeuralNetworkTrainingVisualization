#pragma once
#include <QWidget>
#include <QVector>
#include <QString>
#include <functional>
#include <Eigen/Dense>

namespace App {

struct DatasetMeta {
    QVector<QString> featureNames;
    QVector<double>  minVals;
    QVector<double>  maxVals;
    QVector<QString> classNames;   // indexed by class index
};

class FeatureInputWidget : public QWidget {
    Q_OBJECT
public:
    explicit FeatureInputWidget(QWidget* parent = nullptr);

    void configure(const DatasetMeta& meta);
    void setPredictor(std::function<Eigen::VectorXd(const Eigen::VectorXd&)> pred);
    void reset();

private slots:
    void onPredict();

private:
    void rebuildForm();
    void updateBars();

    DatasetMeta meta_;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> predictor_;
    QVector<double> probs_;

    class FormWidget;
    class BarsWidget;

    FormWidget* form_  = nullptr;
    BarsWidget* bars_  = nullptr;
};

} // namespace App
