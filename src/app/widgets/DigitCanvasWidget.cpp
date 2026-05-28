#include "DigitCanvasWidget.h"
#include "app/ThemeManager.h"
#include <QVBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <algorithm>
#include <cmath>
#include <QPainterPath>

namespace App {

    DigitCanvasWidget::DigitCanvasWidget(QWidget* parent)
        : QWidget(parent)
    {
        canvas = QImage(CANVAS_SIZE, CANVAS_SIZE, QImage::Format_Grayscale8);
        canvas.fill(Qt::black);

        setMinimumSize(DISPLAY_SIZE, DISPLAY_SIZE + 120);
        setMouseTracking(true);
        setCursor(Qt::CrossCursor);

        connect(&ThemeManager::instance(), &ThemeManager::themeChanged,
                this, [this]{ update(); });
    }

    void DigitCanvasWidget::setPredictor(
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> pred)
    {
        predictor = std::move(pred);
        predict();
        update();
    }

    void DigitCanvasWidget::clearCanvas()
    {
        canvas.fill(Qt::black);
        probabilities = Eigen::VectorXd();
        update();
    }

    void DigitCanvasWidget::drawAt(QPointF pos)
    {
        // переводим из координат виджета в координаты холста 28x28
        double scaleX = (double)CANVAS_SIZE / DISPLAY_SIZE;
        double scaleY = (double)CANVAS_SIZE / DISPLAY_SIZE;

        int cx = (int)(pos.x() * scaleX);
        int cy = (int)(pos.y() * scaleY);

        // рисуем с мягкими краями
        for (int dy = -BRUSH_RADIUS; dy <= BRUSH_RADIUS; ++dy) {
            for (int dx = -BRUSH_RADIUS; dx <= BRUSH_RADIUS; ++dx) {
                int nx = cx + dx;
                int ny = cy + dy;
                if (nx < 0 || nx >= CANVAS_SIZE || ny < 0 || ny >= CANVAS_SIZE)
                    continue;

                double dist = std::sqrt(dx * dx + dy * dy);
                double alpha = std::max(0.0, 1.0 - dist / BRUSH_RADIUS);
                int current = canvas.pixel(nx, ny) & 0xFF;
                int newVal = std::min(255, current + (int)(alpha * 200));
                canvas.setPixel(nx, ny, qRgb(newVal, newVal, newVal));
            }
        }
    }

    void DigitCanvasWidget::predict()
    {
        if (!predictor) return;

        QImage centered = centerImage(canvas);

        // конвертируем холст в вектор [0,1]
        Eigen::VectorXd input(CANVAS_SIZE * CANVAS_SIZE);
        for (int y = 0; y < CANVAS_SIZE; ++y)
            for (int x = 0; x < CANVAS_SIZE; ++x)
                input[y * CANVAS_SIZE + x] = (canvas.pixel(x, y) & 0xFF) / 255.0;

        probabilities = predictor(input);
    }

    void DigitCanvasWidget::paintEvent(QPaintEvent*)
    {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);

        const auto& pal = ThemeManager::instance().palette();

        // rounded clip for background + canvas
        QPainterPath clip;
        clip.addRoundedRect(rect(), 8, 8);
        p.setClipPath(clip);

        p.fillRect(rect(), pal.canvasBg);

        // холст — масштабируем 28x28 → DISPLAY_SIZE x DISPLAY_SIZE
        QRect canvasRect(0, 0, DISPLAY_SIZE, DISPLAY_SIZE);
        p.drawImage(canvasRect, canvas);

        p.setClipping(false);

        p.setPen(QPen(pal.canvasBorder, 1));
        p.drawRoundedRect(rect().adjusted(0, 0, -1, -1), 8, 8);

        p.setPen(pal.hint);
        p.setFont(QFont("Arial", 7));
        p.drawText(2, DISPLAY_SIZE + 12, "Draw a digit  |  RMB to clear");

        if (probabilities.size() == 10) {
            int bestDigit = 0;
            probabilities.maxCoeff(&bestDigit);
            p.setPen(pal.title);
            p.setFont(QFont("Arial", 11, QFont::Bold));
            p.drawText(0, DISPLAY_SIZE + 24, width(), 20,
                Qt::AlignCenter,
                QString("Prediction: %1 (%2%)")
                .arg(bestDigit)
                .arg((int)(probabilities[bestDigit] * 100)));

            // диаграмма ниже
            QRect barRect(0, DISPLAY_SIZE + 48, width(), height() - DISPLAY_SIZE - 48);
            drawProbabilities(p, barRect);
        }
    }

    void DigitCanvasWidget::drawProbabilities(QPainter& p, const QRect& r)
    {
        if (r.height() < 20) return;

        int barW = r.width() / 10;
        double maxProb = probabilities.maxCoeff();
        int bestDigit = 0;
        probabilities.maxCoeff(&bestDigit);

        for (int i = 0; i < 10; ++i) {
            double prob = probabilities[i];
            int barH = (int)(prob / std::max(maxProb, 1e-9) * (r.height() - 20));
            int x = r.left() + i * barW;
            int y = r.bottom() - barH - 14;

            // цвет — яркий для максимального, тусклый для остальных
            QColor barColor = (i == bestDigit)
                ? QColor(70, 180, 70)
                : QColor(60, 100, 160);

            p.fillRect(x + 2, y, barW - 4, barH, barColor);

            const auto& pal2 = ThemeManager::instance().palette();
            p.setPen(i == bestDigit ? pal2.title : pal2.gridLabel);
            p.setFont(QFont("Arial", i == bestDigit ? 9 : 7,
                i == bestDigit ? QFont::Bold : QFont::Normal));
            p.drawText(x, r.bottom() - 2, barW, 14,
                Qt::AlignCenter, QString::number(i));

            if (prob > 0.05) {
                p.setPen(ThemeManager::instance().palette().gridLabel);
                p.setFont(QFont("Arial", 6));
                p.drawText(x, y - 10, barW, 10,
                    Qt::AlignCenter,
                    QString("%1%").arg((int)(prob * 100)));
            }
        }
    }

    void DigitCanvasWidget::mousePressEvent(QMouseEvent* event)
    {
        if (event->button() == Qt::LeftButton) {
            drawing = true;
            lastPos = event->position();
            drawAt(lastPos);
            predict();
            update();
        }
        else if (event->button() == Qt::RightButton) {
            clearCanvas();
        }
    }

    void DigitCanvasWidget::mouseMoveEvent(QMouseEvent* event)
    {
        if (!drawing) return;

        QPointF pos = event->position();
        // рисуем линию между предыдущей и текущей точкой
        QLineF line(lastPos, pos);
        int steps = std::max(1, (int)line.length());
        for (int i = 0; i <= steps; ++i) {
            double t = (double)i / steps;
            drawAt(lastPos + t * (pos - lastPos));
        }
        lastPos = pos;
        predict();
        update();
    }

    void DigitCanvasWidget::mouseReleaseEvent(QMouseEvent* event)
    {
        if (event->button() == Qt::LeftButton)
            drawing = false;
    }

    QImage DigitCanvasWidget::centerImage(const QImage& src) {
        // найти границы нарисованного
        int minX = CANVAS_SIZE, maxX = 0;
        int minY = CANVAS_SIZE, maxY = 0;

        for (int y = 0; y < CANVAS_SIZE; ++y)
            for (int x = 0; x < CANVAS_SIZE; ++x)
                if ((src.pixel(x, y) & 0xFF) > 10) {
                    minX = std::min(minX, x);
                    maxX = std::max(maxX, x);
                    minY = std::min(minY, y);
                    maxY = std::max(maxY, y);
                }

        if (minX > maxX) return src; // пустой холст

        // центрируем
        int cx = (minX + maxX) / 2;
        int cy = (minY + maxY) / 2;
        int dx = CANVAS_SIZE / 2 - cx;
        int dy = CANVAS_SIZE / 2 - cy;

        QImage result(CANVAS_SIZE, CANVAS_SIZE, QImage::Format_Grayscale8);
        result.fill(Qt::black);

        for (int y = 0; y < CANVAS_SIZE; ++y)
            for (int x = 0; x < CANVAS_SIZE; ++x) {
                int nx = x - dx, ny = y - dy;
                if (nx >= 0 && nx < CANVAS_SIZE && ny >= 0 && ny < CANVAS_SIZE)
                    result.setPixel(x, y, src.pixel(nx, ny));
            }

        return result;
    }

} // namespace App