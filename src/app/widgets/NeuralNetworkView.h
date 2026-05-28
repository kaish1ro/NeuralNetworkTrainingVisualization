#pragma once

#include <QWidget>
#include <QCheckBox>
#include <QVBoxLayout>
#include <QPainter>
#include <QVector>
#include <Eigen/Dense>

namespace App {

    class NeuralNetworkView : public QWidget {
        Q_OBJECT

    public:
        explicit NeuralNetworkView(QWidget* parent = nullptr);

        // �������� ����������� ���� (���������� �������� � ������ ����)
        void setArchitecture(const QVector<int>& layerSizes);

        // �������� ���� ��� ������������ ������
        void setWeights(const QVector<Eigen::MatrixXd>& weights);

        void clear();

        void setActivations(const QVector<Eigen::VectorXd>& activations);

    protected:
        void paintEvent(QPaintEvent* event) override;
        void wheelEvent(QWheelEvent* event) override;
        void mousePressEvent(QMouseEvent* event) override;
        void mouseMoveEvent(QMouseEvent* event) override;
        void mouseReleaseEvent(QMouseEvent* event) override;
        void mouseDoubleClickEvent(QMouseEvent* event) override;
    private:
        void drawSchematic(QPainter& p);
        void drawDetailed(QPainter& p);

        void drawLayer(QPainter& p, int layerIndex,
            int x, int totalHeight,
            int neuronCount, int displayCount);

        QColor weightColor(double w, double maxW) const;
        QColor activationColor(double a) const;

        QCheckBox* detailCheckBox = nullptr;
        QWidget* canvas = nullptr;

        QVector<int>           layerSizes;
        QVector<Eigen::MatrixXd> weights;
        QVector<Eigen::VectorXd> activations;

        double scaleFactor = 1.0;
        static constexpr double MIN_SCALE = 0.3;
        static constexpr double MAX_SCALE = 3.0;

        static constexpr int MAX_NEURONS = 16;
        static constexpr int NEURON_RADIUS = 14;
        static constexpr int LAYER_SPACING = 200;

        QPointF offset = { 0, 0 };
        QPointF lastMousePos;
        bool isDragging = false;
        QVector<QVector<QPointF>> neuronPositionsVec;
    };

} // namespace App