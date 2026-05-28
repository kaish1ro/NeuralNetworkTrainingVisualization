#include "NeuralNetworkView.h"
#include "app/ThemeManager.h"

#include <QPainterPath>
#include <QVBoxLayout>
#include <algorithm>
#include <cmath>
#include <QWheelEvent>
#include <QToolTip>
#include <QLineF>

namespace App {

    NeuralNetworkView::NeuralNetworkView(QWidget* parent)
        : QWidget(parent)
    {
        auto* layout = new QVBoxLayout(this);
        layout->setContentsMargins(4, 4, 4, 4);

        detailCheckBox = new QCheckBox("Extended view (weights & activations)", this);
        layout->addWidget(detailCheckBox);
        layout->addStretch();

        setMouseTracking(true);

        connect(detailCheckBox, &QCheckBox::toggled, this, [this] { update(); });
        connect(&ThemeManager::instance(), &ThemeManager::themeChanged, this, [this]{ update(); });
    }

    void NeuralNetworkView::setArchitecture(const QVector<int>& sizes)
    {
        layerSizes = sizes;
        update();
    }

    void NeuralNetworkView::setWeights(const QVector<Eigen::MatrixXd>& w)
    {
        weights = w;
        update();
    }

    void NeuralNetworkView::clear()
    {
        layerSizes.clear();
        weights.clear();
        update();
    }

    void NeuralNetworkView::wheelEvent(QWheelEvent* event)
    {
        if (event->modifiers() & Qt::ControlModifier) {
            double delta = event->angleDelta().y() / 120.0;
            scaleFactor = std::clamp(scaleFactor + delta * 0.1, MIN_SCALE, MAX_SCALE);
            update();
            event->accept();
        }
        else {
            QWidget::wheelEvent(event);
        }
    }

    void NeuralNetworkView::paintEvent(QPaintEvent*)
    {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);

        const auto& pal = ThemeManager::instance().palette();

        p.fillRect(rect(), pal.networkBg);

        if (layerSizes.isEmpty()) {
            p.setPen(pal.gridLabel);
            p.setFont(QFont("Arial", 10));
            p.drawText(rect(), Qt::AlignCenter, "No network loaded");
            return;
        }

        // применяем масштаб относительно центра виджета
        p.translate(width() / 2.0 + offset.x(), height() / 2.0 + offset.y());
        p.scale(scaleFactor, scaleFactor);
        p.translate(-width() / 2.0, -height() / 2.0);

        if (detailCheckBox->isChecked())
            drawDetailed(p);
        else
            drawSchematic(p);

        // подсказка и масштаб — рисуем без трансформации
        p.resetTransform();
        p.setPen(pal.hint);
        p.setFont(QFont("Arial", 8));
        p.drawText(8, height() - 4, QString("Ctrl+scroll to zoom  |  %1%")
            .arg((int)(scaleFactor * 100)));

        if (detailCheckBox->isChecked()) {
            p.setFont(QFont("Arial", 8));
            p.setPen(QColor(200, 80, 80));
            p.drawText(10, height() - 38, "— negative weight");
            p.setPen(QColor(80, 80, 200));
            p.drawText(10, height() - 24, "— positive weight");

            // легенда активаций
            if (!activations.isEmpty()) {
                int gx = width() - 200, gy = 30;
                int gw = 120, gh = 14;

                p.setCompositionMode(QPainter::CompositionMode_SourceOver);
                p.setOpacity(1.0);
                p.setPen(Qt::NoPen);
                p.setBrush(Qt::NoBrush);

                for (int i = 0; i < gw; ++i) {
                    double t = (double)i / gw;
                    QColor c;
                    if (t < 0.5) {
                        double s = t * 2.0;
                        c = QColor((int)(20 + 0 * s), (int)(20 + 160 * s), (int)(200 - 180 * s));
                    }
                    else {
                        double s = (t - 0.5) * 2.0;
                        c = QColor((int)(20 + 200 * s), (int)(180 - 160 * s), (int)(20));
                    }
                    p.fillRect(gx + i, gy, 1, gh, c);
                }

                p.setPen(QColor(100, 100, 100));
                p.drawRect(gx, gy, gw, gh);
                p.setPen(QColor(150, 150, 150));
                p.setFont(QFont("Arial", 7));
                p.drawText(gx - 2, gy - 2, "0");
                p.drawText(gx + gw - 4, gy - 2, "1");
                p.drawText(gx + gw + 4, gy, 60, gh, Qt::AlignVCenter, "activation");
            }
        }
    }

    // вычисляем позиции нейронов по вертикали
    static QVector<int> neuronPositions(int displayCount, int totalHeight,
        int radius, int maxSpacing = 50)
    {
        QVector<int> pos;
        int spacing = (totalHeight - 2 * radius) / std::max(displayCount - 1, 1);
        spacing = std::min(spacing, maxSpacing);
        int totalH = (displayCount - 1) * spacing;
        int startY = (totalHeight - totalH) / 2;
        for (int i = 0; i < displayCount; ++i)
            pos.append(startY + i * spacing);
        return pos;
    }

    void NeuralNetworkView::drawSchematic(QPainter& p)
    {
        int totalHeight = height() - 40;
        int totalWidth = width() - 40;
        int layerCount = layerSizes.size();

        int spacing = totalWidth / std::max(layerCount - 1, 1);
        spacing = std::min(spacing, (int)(LAYER_SPACING * scaleFactor));
        int startX = (totalWidth - (layerCount - 1) * spacing) / 2 + 20;

        // собираем позиции всех нейронов
        QVector<QVector<QPointF>> positions(layerCount);
        for (int l = 0; l < layerCount; ++l) {
            int count = layerSizes[l];
            int display = std::min(count, MAX_NEURONS);
            int x = startX + l * spacing;
            auto yPos = neuronPositions(display, totalHeight, NEURON_RADIUS,
                (int)(50 * scaleFactor));
            for (int n = 0; n < display; ++n)
                positions[l].append(QPointF(x, yPos[n] + 20));
        }

        this->neuronPositionsVec = positions;

        const auto& pal = ThemeManager::instance().palette();

        // рисуем связи
        p.setPen(QPen(pal.connection, 0.5));
        for (int l = 0; l < layerCount - 1; ++l) {
            for (auto& from : positions[l])
                for (auto& to : positions[l + 1])
                    p.drawLine(from, to);
        }

        // рисуем нейроны
        for (int l = 0; l < layerCount; ++l) {
            int count = layerSizes[l];
            int display = std::min(count, MAX_NEURONS);
            bool hasMore = count > MAX_NEURONS;

            for (int n = 0; n < display; ++n) {
                QPointF pos = positions[l][n];

                // последний видимый нейрон — рисуем "..."
                if (hasMore && n == display - 1) {
                    p.setPen(QColor(150, 150, 150));
                    p.setFont(QFont("Arial", 9));
                    p.drawText(QRectF(pos.x() - 20, pos.y() - 10, 40, 20),
                        Qt::AlignCenter, "...");
                    continue;
                }

                p.setBrush(pal.neuronDefault);
                p.setPen(QPen(pal.neuronDefault.lighter(160), 1.5));
                p.drawEllipse(pos, NEURON_RADIUS, NEURON_RADIUS);
            }

            p.setPen(pal.title);
            p.setFont(QFont("Arial", 8));
            int x = (int)positions[l][0].x();
            int labelY = (int)positions[l].back().y() + NEURON_RADIUS + 8;
            p.drawText(x - 30, labelY, 60, 16, Qt::AlignCenter, QString::number(count));
        }
    }

    void NeuralNetworkView::drawDetailed(QPainter& p)
    {
        if (weights.isEmpty()) {
            drawSchematic(p);
            return;
        }

        int totalHeight = height() - 40;
        int totalWidth = width() - 40;
        int layerCount = layerSizes.size();

        int spacing = totalWidth / std::max(layerCount - 1, 1);
        spacing = std::min(spacing, (int)(LAYER_SPACING * scaleFactor));
        int startX = (totalWidth - (layerCount - 1) * spacing) / 2 + 20;

        // позиции нейронов
        QVector<QVector<QPointF>> positions(layerCount);
        for (int l = 0; l < layerCount; ++l) {
            int count = layerSizes[l];
            int display = std::min(count, MAX_NEURONS);
            int x = startX + l * spacing;
            auto yPos = neuronPositions(display, totalHeight, NEURON_RADIUS,
                (int)(50 * scaleFactor));
            for (int n = 0; n < display; ++n)
                positions[l].append(QPointF(x, yPos[n] + 20));
        }

        this->neuronPositionsVec = positions;

        // рисуем связи с цветом по весу
        for (int l = 0; l < (int)weights.size() && l < layerCount - 1; ++l) {
            const auto& W = weights[l];
            double maxW = W.cwiseAbs().maxCoeff();
            if (maxW < 1e-9) maxW = 1.0;

            int fromCount = std::min((int)positions[l].size(), (int)W.cols());
            int toCount = std::min((int)positions[l + 1].size(), (int)W.rows());

            for (int to = 0; to < toCount; ++to) {
                for (int from = 0; from < fromCount; ++from) {
                    double w = W(to, from);
                    QColor c = weightColor(w, maxW);
                    double thickness = 0.3 + 1.5 * std::abs(w) / maxW;
                    p.setPen(QPen(c, thickness));
                    p.drawLine(positions[l][from], positions[l + 1][to]);
                }
            }
        }

        // рисуем нейроны
        for (int l = 0; l < layerCount; ++l) {
            int count = layerSizes[l];
            int display = std::min(count, MAX_NEURONS);
            bool hasMore = count > MAX_NEURONS;

            for (int n = 0; n < display; ++n) {
                QPointF pos = positions[l][n];

                if (hasMore && n == display - 1) {
                    p.setPen(QColor(150, 150, 150));
                    p.setFont(QFont("Arial", 9));
                    p.drawText(QRectF(pos.x() - 20, pos.y() - 10, 40, 20),
                        Qt::AlignCenter, "...");
                    continue;
                }

                QColor neuronColor = QColor(60, 120, 200);
                if (!activations.isEmpty() && l > 0 && l - 1 < activations.size()) {
                    int actIdx = l - 1;
                    int neuronIdx = (n < (int)activations[actIdx].size()) ? n : (int)activations[actIdx].size() - 1;
                    neuronColor = activationColor(activations[actIdx][neuronIdx]);
                }
                p.setBrush(neuronColor);
                p.setPen(QPen(neuronColor.lighter(150), 1.5));
                p.drawEllipse(pos, NEURON_RADIUS, NEURON_RADIUS);
            }

            p.setPen(ThemeManager::instance().palette().title);
            p.setFont(QFont("Arial", 8));
            int x = (int)positions[l][0].x();
            int labelY = (int)positions[l].back().y() + NEURON_RADIUS + 8;
            p.drawText(x - 30, labelY, 60, 16, Qt::AlignCenter, QString::number(count));
        }
    }

    QColor NeuralNetworkView::weightColor(double w, double maxW) const
    {
        double t = std::clamp(w / maxW, -1.0, 1.0);
        if (t > 0)
            return QColor(
                (int)(80 * (1 - t)),
                (int)(80 * (1 - t)),
                (int)(200 * t + 80 * (1 - t)),
                180
            );
        else
            return QColor(
                (int)(200 * (-t) + 80 * (1 + t)),
                (int)(80 * (1 + t)),
                (int)(80 * (1 + t)),
                180
            );
    }

    QColor NeuralNetworkView::activationColor(double a) const
    {
        double t = std::clamp(a, 0.0, 1.0);
        if (t < 0.5) {
            double s = t * 2.0;
            return QColor(
                (int)(20 * (1 - s)),
                (int)(180 * s),
                (int)(200 * (1 - s))
            );
        }
        else {
            double s = (t - 0.5) * 2.0;
            return QColor(
                (int)(220 * s),
                (int)(180 * (1 - s)),
                (int)(20 * (1 - s))
            );
        }
    }

    void NeuralNetworkView::mousePressEvent(QMouseEvent* event)
    {
        if (event->button() == Qt::LeftButton) {
            isDragging = true;
            lastMousePos = event->position();
            setCursor(Qt::ClosedHandCursor);
        }
    }

    void NeuralNetworkView::mouseMoveEvent(QMouseEvent* event)
    {
        if (isDragging) {
            QPointF delta = event->position() - lastMousePos;
            offset += delta;
            lastMousePos = event->position();

            // keep network within visible area
            double limX = width()  * 0.65;
            double limY = height() * 0.65;
            offset.rx() = std::clamp(offset.x(), -limX, limX);
            offset.ry() = std::clamp(offset.y(), -limY, limY);

            update();
        }

        QPointF mousePos = event->position();
        mousePos -= QPointF(width() / 2.0 + offset.x(), height() / 2.0 + offset.y());
        mousePos /= scaleFactor;
        mousePos += QPointF(width() / 2.0, height() / 2.0);

        bool found = false;
        for (int l = 0; l < neuronPositionsVec.size(); ++l) {
            for (int n = 0; n < neuronPositionsVec[l].size(); ++n) {
                if (QLineF(mousePos, neuronPositionsVec[l][n]).length() <= NEURON_RADIUS) {
                    QString tip = QString("Layer %1, Neuron %2").arg(l).arg(n);

                    // activations[0] соответствует первому скрытому слою (l=1)
                    // входной слой (l=0) активаций не имеет
                    int activationLayer = l - 1;
                    if (!activations.isEmpty()
                        && activationLayer >= 0
                        && activationLayer < activations.size()
                        && n < (int)activations[activationLayer].size())
                    {
                        tip += QString("\nActivation: %1")
                            .arg(activations[activationLayer][n], 0, 'f', 4);
                    }

                    setToolTip(tip);
                    return;
                }
            }
        }
        setToolTip("");
        QToolTip::hideText();
    }

    void NeuralNetworkView::mouseReleaseEvent(QMouseEvent* event)
    {
        if (event->button() == Qt::LeftButton) {
            isDragging = false;
            setCursor(Qt::ArrowCursor);
        }
    }

    void NeuralNetworkView::mouseDoubleClickEvent(QMouseEvent* event)
    {
        if (event->button() == Qt::LeftButton) {
            offset = { 0, 0 };
            scaleFactor = 1.0;
            update();
        }
    }

    void NeuralNetworkView::setActivations(const QVector<Eigen::VectorXd>& a)
    {
        activations = a;
        update();
    }
} // namespace App