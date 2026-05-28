#pragma once
#include <QWidget>
#include "threading/AnalysisResult.h"

namespace App {

class MisclassifiedWidget : public QWidget {
    Q_OBJECT
public:
    explicit MisclassifiedWidget(QWidget* parent = nullptr);
    void setData(const QVector<MisclassifiedSample>& samples);
    void clear();

protected:
    void paintEvent(QPaintEvent*) override;

private:
    QVector<MisclassifiedSample> samples_;

    static constexpr int kImgSize = 36;
    static constexpr int kCols    = 5;
    static constexpr int kLabelH  = 16;
    static constexpr int kPad     = 4;
};

} // namespace App
