#pragma once

#include <QWidget>
#include <QImage>
#include <QMouseEvent>
#include <QPainter>
#include <QVector>
#include <functional>
#include <Eigen/Dense>

namespace App {

    class DigitCanvasWidget : public QWidget {
        Q_OBJECT

    public:
        explicit DigitCanvasWidget(QWidget* parent = nullptr);

        // установить функцию предсказани€ Ч вызываетс€ при каждом штрихе
        void setPredictor(std::function<Eigen::VectorXd(const Eigen::VectorXd&)> predictor);
        void clearCanvas();
        QImage centerImage(const QImage& src);

    protected:
        void paintEvent(QPaintEvent* event) override;
        void mousePressEvent(QMouseEvent* event) override;
        void mouseMoveEvent(QMouseEvent* event) override;
        void mouseReleaseEvent(QMouseEvent* event) override;

    private:
        void drawAt(QPointF pos);
        void predict();
        void drawProbabilities(QPainter& p, const QRect& r);

        QImage          canvas;       // 28x28 пикселей
        bool            drawing = false;
        QPointF         lastPos;

        Eigen::VectorXd probabilities;  // выход сети
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> predictor;

        static constexpr int CANVAS_SIZE = 28;
        static constexpr int DISPLAY_SIZE = 196;  // 28 * 7
        static constexpr int BRUSH_RADIUS = 2;    // в пиксел€х холста
    };

} // namespace App