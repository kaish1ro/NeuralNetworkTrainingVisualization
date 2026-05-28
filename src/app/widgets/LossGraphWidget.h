#pragma once

#include <QWidget>
#include <QVector>
#include <QColor>
#include <QDialog>
#include <QMouseEvent>
#include <QWheelEvent>

namespace App {

    struct GraphRun {
        QVector<double> values;
        QColor          color;
        int             runNumber = 0;
    };

    class LossGraphWidget : public QWidget {
        Q_OBJECT

    public:
        explicit LossGraphWidget(const QString& title, QWidget* parent = nullptr);

        void addValue(double value);   // добавить точку в текущий прогон
        void nextRun();                // начать новый прогон
        void clear();                  // очистить все прогоны

        // для отдельного окна — передаём копию данных
        const QVector<GraphRun>& getRuns() const { return runs; }
        const QString& getTitle() const { return title; }

    protected:
        void paintEvent(QPaintEvent* event) override;
        void mouseDoubleClickEvent(QMouseEvent* event) override;

    private:
        void drawGraph(QPainter& p, const QRect& drawRect,
            const QVector<GraphRun>& data, bool showLegend = false);


        QString         title;
        QVector<GraphRun> runs;
        GraphRun        currentRun;
        int             runCounter = 0;
    };

    // ── Полноэкранное окно графика ────────────────────────────────────

    class GraphDialog : public QDialog {
        Q_OBJECT

    public:
        explicit GraphDialog(const QString& title,
            const QVector<GraphRun>& runs,
            QWidget* parent = nullptr);

    protected:
        void paintEvent(QPaintEvent* event) override;
        void wheelEvent(QWheelEvent* event) override;
        void mousePressEvent(QMouseEvent* event) override;
        void mouseMoveEvent(QMouseEvent* event) override;
        void mouseReleaseEvent(QMouseEvent* event) override;

    private:
        void drawGraph(QPainter& p);

        QString           title;
        QVector<GraphRun> runs;

        double  scaleFactor = 1.0;
        QPointF offset = { 0, 0 };
        QPointF lastMousePos;
        bool    isDragging = false;

        static constexpr double MIN_SCALE = 0.3;
        static constexpr double MAX_SCALE = 5.0;
    };

} // namespace App