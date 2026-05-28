#include "LossGraphWidget.h"
#include "app/ThemeManager.h"
#include <QPainter>
#include <QPainterPath>
#include <QVBoxLayout>
#include <algorithm>
#include <cmath>

namespace App {

    // ── LossGraphWidget ───────────────────────────────────────────────

    LossGraphWidget::LossGraphWidget(const QString& title, QWidget* parent)
        : QWidget(parent), title(title)
    {
        setMinimumHeight(150);
        currentRun.runNumber = 0;
        runCounter = 0;
        currentRun.color = ThemeManager::instance().runColor(0, 1);

        connect(&ThemeManager::instance(), &ThemeManager::themeChanged, this, [this]() {
            int total = (int)runs.size() + 1;
            for (int i = 0; i < (int)runs.size(); ++i)
                runs[i].color = ThemeManager::instance().runColor(i, total);
            currentRun.color = ThemeManager::instance().runColor((int)runs.size(), total);
            update();
        });
    }

    void LossGraphWidget::addValue(double value)
    {
        currentRun.values.append(value);
        int total = (int)runs.size() + 1;
        for (int i = 0; i < (int)runs.size(); ++i)
            runs[i].color = ThemeManager::instance().runColor(i, total);
        currentRun.color = ThemeManager::instance().runColor((int)runs.size(), total);
        update();
    }

    void LossGraphWidget::nextRun()
    {
        if (!currentRun.values.isEmpty())
            runs.append(currentRun);

        currentRun = GraphRun();
        currentRun.runNumber = ++runCounter;

        int total = (int)runs.size() + 1;
        for (int i = 0; i < (int)runs.size(); ++i)
            runs[i].color = ThemeManager::instance().runColor(i, total);
        currentRun.color = ThemeManager::instance().runColor((int)runs.size(), total);
    }

    void LossGraphWidget::clear()
    {
        runs.clear();
        currentRun = GraphRun();
        runCounter = 0;
        currentRun.runNumber = ++runCounter;
        currentRun.color = ThemeManager::instance().runColor(0, 1);
        update();
    }

    void LossGraphWidget::paintEvent(QPaintEvent*)
    {
        const auto& pal = ThemeManager::instance().palette();

        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        p.fillRect(rect(), pal.widgetBg);

        p.setPen(pal.title);
        p.setFont(QFont("Arial", 9, QFont::Bold));
        p.drawText(8, 12, title);

        p.setPen(pal.hint);
        p.setFont(QFont("Arial", 7));
        p.drawText(rect().adjusted(0, 0, -4, -2),
            Qt::AlignBottom | Qt::AlignRight, "double-click to expand");

        QVector<GraphRun> allRuns = runs;
        if (!currentRun.values.isEmpty())
            allRuns.append(currentRun);

        drawGraph(p, rect().adjusted(30, 24, -4, -18), allRuns);
    }

    void LossGraphWidget::drawGraph(QPainter& p, const QRect& r,
        const QVector<GraphRun>& data,
        bool showLegend)
    {
        const auto& pal = ThemeManager::instance().palette();

        if (data.isEmpty() || r.width() <= 0 || r.height() <= 0) {
            p.setPen(pal.gridLabel);
            p.setFont(QFont("Arial", 8));
            p.drawText(r, Qt::AlignCenter, "Waiting for data...");
            return;
        }

        double minV = 1e18, maxV = -1e18;
        int maxLen = 0;
        for (const auto& run : data) {
            for (double v : run.values) {
                minV = std::min(minV, v);
                maxV = std::max(maxV, v);
            }
            maxLen = std::max(maxLen, (int)run.values.size());
        }
        if (maxLen < 2) return;
        if (std::abs(maxV - minV) < 1e-9) maxV = minV + 1.0;

        // grid
        p.setPen(QPen(pal.grid, 0.5));
        for (int i = 0; i <= 4; ++i) {
            int y = r.top() + r.height() * i / 4;
            p.drawLine(r.left(), y, r.right(), y);
            double val = maxV - (maxV - minV) * i / 4;
            p.setPen(pal.gridLabel);
            p.setFont(QFont("Arial", 7));
            p.drawText(0, y - 6, 28, 12, Qt::AlignRight | Qt::AlignVCenter,
                QString::number(val, 'f', 3));
            p.setPen(QPen(pal.grid, 0.5));
        }

        // runs
        for (const auto& run : data) {
            if (run.values.size() < 2) continue;

            int n = run.values.size();
            QPainterPath path;
            for (int i = 0; i < n; ++i) {
                double x = r.left() + (double)i / (maxLen - 1) * r.width();
                double y = r.top() + (1.0 - (run.values[i] - minV) / (maxV - minV)) * r.height();
                i == 0 ? path.moveTo(x, y) : path.lineTo(x, y);
            }

            if (&run == &data.last()) {
                QPainterPath fill = path;
                fill.lineTo(r.right(), r.bottom());
                fill.lineTo(r.left(), r.bottom());
                fill.closeSubpath();
                QLinearGradient grad(0, r.top(), 0, r.bottom());
                grad.setColorAt(0, QColor(run.color.red(), run.color.green(),
                    run.color.blue(), 80));
                grad.setColorAt(1, QColor(run.color.red(), run.color.green(),
                    run.color.blue(), 5));
                p.fillPath(fill, grad);
            }

            p.setPen(QPen(run.color, &run == &data.last() ? 2.0 : 1.0));
            p.drawPath(path);

            if (&run == &data.last()) {
                p.setBrush(run.color);
                p.setPen(Qt::NoPen);
                for (int i = 0; i < n; ++i) {
                    double x = r.left() + (double)i / (maxLen - 1) * r.width();
                    double y = r.top() + (1.0 - (run.values[i] - minV) / (maxV - minV)) * r.height();
                    p.drawEllipse(QPointF(x, y), 2.5, 2.5);
                }
            }
        }

        // legend
        if (showLegend && data.size() > 1) {
            int lx = r.right() - 100;
            int ly = r.top() + 4;
            for (int i = 0; i < (int)data.size(); ++i) {
                p.setPen(QPen(data[i].color, 2));
                p.drawLine(lx, ly + i * 14 + 6, lx + 16, ly + i * 14 + 6);
                p.setPen(data[i].color);
                p.setFont(QFont("Arial", 7));
                p.drawText(lx + 20, ly + i * 14, 80, 14,
                    Qt::AlignVCenter, QString("Run %1").arg(data[i].runNumber));
            }
        }
    }

    void LossGraphWidget::mouseDoubleClickEvent(QMouseEvent*)
    {
        QVector<GraphRun> allRuns = runs;
        if (!currentRun.values.isEmpty())
            allRuns.append(currentRun);

        if (allRuns.isEmpty()) return;

        auto* dlg = new GraphDialog(title, allRuns, this);
        dlg->setAttribute(Qt::WA_DeleteOnClose);
        dlg->resize(800, 500);
        dlg->exec();
    }

    // ── GraphDialog ───────────────────────────────────────────────────

    GraphDialog::GraphDialog(const QString& title,
        const QVector<GraphRun>& runs,
        QWidget* parent)
        : QDialog(parent), title(title), runs(runs)
    {
        setWindowTitle(title);
        setMinimumSize(600, 400);

        connect(&ThemeManager::instance(), &ThemeManager::themeChanged,
                this, [this]{ update(); });
    }

    void GraphDialog::paintEvent(QPaintEvent*)
    {
        const auto& pal = ThemeManager::instance().palette();

        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        p.fillRect(rect(), pal.dialogBg);
        drawGraph(p);

        p.resetTransform();
        p.setPen(pal.hint);
        p.setFont(QFont("Arial", 8));
        p.drawText(8, height() - 4,
            QString("Ctrl+scroll to zoom  |  drag to pan  |  %1%")
            .arg((int)(scaleFactor * 100)));
    }

    void GraphDialog::drawGraph(QPainter& p)
    {
        if (runs.isEmpty()) return;

        const auto& pal = ThemeManager::instance().palette();
        const int padL = 50, padT = 30, padR = 120, padB = 30;
        QRect r(padL, padT, width() - padL - padR, height() - padT - padB);

        p.translate(width() / 2.0 + offset.x(), height() / 2.0 + offset.y());
        p.scale(scaleFactor, scaleFactor);
        p.translate(-width() / 2.0, -height() / 2.0);

        p.setPen(pal.title);
        p.setFont(QFont("Arial", 11, QFont::Bold));
        p.drawText(padL, 20, title);

        double minV = 1e18, maxV = -1e18;
        int maxLen = 0;
        for (const auto& run : runs) {
            for (double v : run.values) {
                minV = std::min(minV, v);
                maxV = std::max(maxV, v);
            }
            maxLen = std::max(maxLen, (int)run.values.size());
        }
        if (maxLen < 2 || std::abs(maxV - minV) < 1e-9) return;

        // grid Y
        p.setPen(QPen(pal.grid, 0.5));
        for (int i = 0; i <= 6; ++i) {
            int y = r.top() + r.height() * i / 6;
            p.drawLine(r.left(), y, r.right(), y);
            double val = maxV - (maxV - minV) * i / 6;
            p.setPen(pal.gridLabel);
            p.setFont(QFont("Arial", 8));
            p.drawText(0, y - 8, padL - 4, 16,
                Qt::AlignRight | Qt::AlignVCenter,
                QString::number(val, 'f', 4));
            p.setPen(QPen(pal.grid, 0.5));
        }

        // grid X
        for (int i = 0; i <= std::min(maxLen - 1, 10); ++i) {
            int x = r.left() + r.width() * i / std::max(maxLen - 1, 1);
            p.setPen(QPen(pal.grid, 0.5));
            p.drawLine(x, r.top(), x, r.bottom());
            p.setPen(pal.gridLabel);
            p.setFont(QFont("Arial", 8));
            int epoch = 1 + i * (maxLen - 1) / std::max(10, 1);
            p.drawText(x - 15, r.bottom() + 2, 30, 16,
                Qt::AlignCenter, QString::number(epoch));
        }

        // runs
        for (int ri = 0; ri < (int)runs.size(); ++ri) {
            const auto& run = runs[ri];
            if (run.values.size() < 2) continue;

            int n = run.values.size();
            QPainterPath path;
            for (int i = 0; i < n; ++i) {
                double x = r.left() + (double)i / (maxLen - 1) * r.width();
                double y = r.top() + (1.0 - (run.values[i] - minV) / (maxV - minV)) * r.height();
                i == 0 ? path.moveTo(x, y) : path.lineTo(x, y);
            }

            bool isLast = (ri == (int)runs.size() - 1);

            if (isLast) {
                QPainterPath fill = path;
                fill.lineTo(r.right(), r.bottom());
                fill.lineTo(r.left(), r.bottom());
                fill.closeSubpath();
                QLinearGradient grad(0, r.top(), 0, r.bottom());
                grad.setColorAt(0, QColor(run.color.red(), run.color.green(),
                    run.color.blue(), 60));
                grad.setColorAt(1, Qt::transparent);
                p.fillPath(fill, grad);
            }

            p.setPen(QPen(run.color, isLast ? 2.5 : 1.2));
            p.drawPath(path);

            p.setBrush(run.color);
            p.setPen(Qt::NoPen);
            for (int i = 0; i < n; ++i) {
                double x = r.left() + (double)i / (maxLen - 1) * r.width();
                double y = r.top() + (1.0 - (run.values[i] - minV) / (maxV - minV)) * r.height();
                p.drawEllipse(QPointF(x, y), isLast ? 3.5 : 2.0, isLast ? 3.5 : 2.0);
            }
        }

        // legend
        int lx = r.right() + 8;
        int ly = r.top();
        p.setFont(QFont("Arial", 8));
        for (int i = 0; i < (int)runs.size(); ++i) {
            bool isLast = (i == (int)runs.size() - 1);
            p.setPen(QPen(runs[i].color, isLast ? 2.5 : 1.5));
            p.drawLine(lx, ly + i * 18 + 7, lx + 20, ly + i * 18 + 7);
            p.setPen(isLast ? pal.title : pal.gridLabel);
            QString label = QString("Run %1").arg(runs[i].runNumber);
            if (!runs[i].values.isEmpty())
                label += QString("  %1").arg(runs[i].values.last(), 0, 'f', 4);
            p.drawText(lx + 24, ly + i * 18, 90, 14, Qt::AlignVCenter, label);
        }
    }

    void GraphDialog::wheelEvent(QWheelEvent* event)
    {
        if (event->modifiers() & Qt::ControlModifier) {
            double delta = event->angleDelta().y() / 120.0;
            scaleFactor = std::clamp(scaleFactor + delta * 0.15, MIN_SCALE, MAX_SCALE);
            update();
            event->accept();
        }
        else {
            QDialog::wheelEvent(event);
        }
    }

    void GraphDialog::mousePressEvent(QMouseEvent* event)
    {
        if (event->button() == Qt::LeftButton) {
            isDragging = true;
            lastMousePos = event->position();
            setCursor(Qt::ClosedHandCursor);
        }
    }

    void GraphDialog::mouseMoveEvent(QMouseEvent* event)
    {
        if (isDragging) {
            offset += event->position() - lastMousePos;
            lastMousePos = event->position();
            update();
        }
    }

    void GraphDialog::mouseReleaseEvent(QMouseEvent* event)
    {
        if (event->button() == Qt::LeftButton) {
            isDragging = false;
            setCursor(Qt::ArrowCursor);
        }
    }

} // namespace App
