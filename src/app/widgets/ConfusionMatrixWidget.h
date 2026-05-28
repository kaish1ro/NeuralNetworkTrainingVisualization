#pragma once
#include <QWidget>
#include <QVector>

namespace App {

class ConfusionMatrixWidget : public QWidget {
    Q_OBJECT
public:
    explicit ConfusionMatrixWidget(QWidget* parent = nullptr);
    void setData(const QVector<QVector<int>>& matrix, int numClasses);
    void clear();
    QSize sizeHint() const override;

protected:
    void paintEvent(QPaintEvent*) override;

private:
    QVector<QVector<int>> matrix_;
    int numClasses_  = 0;
    int maxDiag_     = 1;
    int maxOffDiag_  = 1;
};

} // namespace App
