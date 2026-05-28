#include "MisclassifiedWidget.h"
#include <QPainter>
#include <QImage>

namespace App {

MisclassifiedWidget::MisclassifiedWidget(QWidget* parent)
    : QWidget(parent)
{}

void MisclassifiedWidget::setData(const QVector<MisclassifiedSample>& samples)
{
    samples_ = samples;
    int rows = (samples_.size() + kCols - 1) / kCols;
    setFixedHeight(rows * (kImgSize + kLabelH + kPad) + kPad);
    update();
}

void MisclassifiedWidget::clear()
{
    samples_.clear();
    setFixedHeight(0);
    update();
}

void MisclassifiedWidget::paintEvent(QPaintEvent*)
{
    if (samples_.isEmpty()) return;

    QPainter p(this);
    p.setRenderHint(QPainter::SmoothPixmapTransform);

    QFont font = p.font();
    font.setPointSize(7);
    p.setFont(font);

    const int cellW = kImgSize + kPad;
    const int cellH = kImgSize + kLabelH + kPad;

    for (int i = 0; i < samples_.size(); ++i) {
        const auto& s  = samples_[i];
        int col = i % kCols;
        int row = i / kCols;
        int x   = col * cellW + kPad;
        int y   = row * cellH + kPad;

        QImage img(28, 28, QImage::Format_Grayscale8);
        for (int py = 0; py < 28; ++py) {
            uchar* line = img.scanLine(py);
            for (int px = 0; px < 28; ++px) {
                int idx = py * 28 + px;
                line[px] = idx < s.input.size()
                    ? static_cast<uchar>(s.input[idx] * 255.0)
                    : 0;
            }
        }

        p.drawImage(QRect(x, y, kImgSize, kImgSize), img);

        p.setPen(QPen(QColor(200, 60, 60), 1));
        p.drawRect(x, y, kImgSize - 1, kImgSize - 1);

        p.setPen(QColor(220, 110, 110));
        p.drawText(QRect(x, y + kImgSize + 1, cellW, kLabelH - 1),
                   Qt::AlignCenter,
                   QString("%1→%2").arg(s.actual).arg(s.predicted));
    }
}

} // namespace App
