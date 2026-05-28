#include "ConfusionMatrixWidget.h"
#include <QPainter>
#include <algorithm>
#include <cmath>

namespace App {

ConfusionMatrixWidget::ConfusionMatrixWidget(QWidget* parent)
    : QWidget(parent)
{
    setMinimumSize(200, 200);
}

void ConfusionMatrixWidget::setData(const QVector<QVector<int>>& matrix, int numClasses)
{
    matrix_     = matrix;
    numClasses_ = numClasses;
    maxDiag_    = 1;
    maxOffDiag_ = 1;

    for (int r = 0; r < numClasses; ++r)
        for (int c = 0; c < numClasses; ++c) {
            if (r == c) maxDiag_    = std::max(maxDiag_,    matrix[r][c]);
            else        maxOffDiag_ = std::max(maxOffDiag_, matrix[r][c]);
        }

    update();
}

void ConfusionMatrixWidget::clear()
{
    matrix_.clear();
    numClasses_ = 0;
    update();
}

QSize ConfusionMatrixWidget::sizeHint() const { return {240, 240}; }

void ConfusionMatrixWidget::paintEvent(QPaintEvent*)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, false);

    if (numClasses_ == 0) {
        p.setPen(Qt::gray);
        p.drawText(rect(), Qt::AlignCenter, "No data");
        return;
    }

    const int labelW = 18;
    const int labelH = 18;

    int avail = std::min(width() - labelW, height() - labelH);
    int cell  = std::max(4, avail / numClasses_);
    int grid  = cell * numClasses_;
    int startX = labelW + (width()  - labelW - grid) / 2;
    int startY = labelH + (height() - labelH - grid) / 2;

    QFont small = p.font();
    small.setPointSize(cell >= 20 ? 7 : 6);
    p.setFont(small);

    for (int r = 0; r < numClasses_; ++r) {
        for (int c = 0; c < numClasses_; ++c) {
            int count = matrix_[r][c];
            QRect cr(startX + c * cell, startY + r * cell, cell, cell);

            QColor bg;
            if (r == c) {
                float t = std::sqrt((float)count / maxDiag_);
                // diagonal: dark blue-green
                bg = QColor::fromRgbF(
                    0.05f + 0.10f * (1 - t),
                    0.30f + 0.45f * t,
                    0.50f + 0.45f * t
                );
            } else if (count == 0) {
                bg = QColor(245, 245, 245);
            } else {
                float t = std::sqrt((float)count / maxOffDiag_);
                // off-diagonal: dark red
                bg = QColor::fromRgbF(
                    0.45f + 0.55f * t,
                    0.05f * (1 - t),
                    0.05f * (1 - t)
                );
            }
            p.fillRect(cr, bg);

            if (cell >= 16 && count > 0) {
                float lum = 0.299f * bg.redF() + 0.587f * bg.greenF() + 0.114f * bg.blueF();
                p.setPen(lum > 0.45f ? Qt::black : Qt::white);
                p.drawText(cr, Qt::AlignCenter, QString::number(count));
            }
        }
    }

    // grid lines
    p.setPen(QPen(QColor(100, 100, 100), 0.5));
    for (int i = 0; i <= numClasses_; ++i) {
        p.drawLine(startX + i * cell, startY,        startX + i * cell, startY + grid);
        p.drawLine(startX,            startY + i * cell, startX + grid, startY + i * cell);
    }

    // axis labels
    QFont lf = p.font();
    lf.setPointSize(7);
    p.setFont(lf);
    p.setPen(Qt::lightGray);
    for (int i = 0; i < numClasses_; ++i) {
        p.drawText(QRect(startX + i * cell, 0, cell, labelH),
                   Qt::AlignCenter, QString::number(i));
        p.drawText(QRect(0, startY + i * cell, labelW, cell),
                   Qt::AlignCenter, QString::number(i));
    }
}

} // namespace App
