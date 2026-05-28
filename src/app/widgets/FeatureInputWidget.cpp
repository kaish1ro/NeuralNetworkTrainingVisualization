#include "FeatureInputWidget.h"

#include <QVBoxLayout>
#include <QFormLayout>
#include <QDoubleSpinBox>
#include <QPushButton>
#include <QScrollArea>
#include <QLabel>
#include <QPainter>
#include <algorithm>
#include <cmath>

namespace App {

// ── Probability bars (horizontal) ────────────────────────────────────────────

class FeatureInputWidget::BarsWidget : public QWidget {
public:
    explicit BarsWidget(QWidget* parent = nullptr) : QWidget(parent) {
        setMinimumHeight(60);
    }

    void setData(const QVector<double>& probs, const QVector<QString>& names) {
        probs_ = probs;
        names_ = names;
        int h = std::max(60, (int)probs.size() * 26 + 16);
        setMinimumHeight(h);
        update();
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing, false);

        if (probs_.isEmpty()) {
            p.setPen(Qt::gray);
            p.drawText(rect(), Qt::AlignCenter, "—");
            return;
        }

        const int labelW = 110;
        const int pctW   = 36;
        const int barH   = 18;
        const int rowH   = 24;
        const int pad    = 8;

        int bestIdx = 0;
        for (int i = 1; i < probs_.size(); ++i)
            if (probs_[i] > probs_[bestIdx]) bestIdx = i;

        QFont font = p.font();
        font.setPointSize(8);
        p.setFont(font);

        for (int i = 0; i < probs_.size(); ++i) {
            int y = pad + i * rowH;
            double prob = (i < probs_.size()) ? probs_[i] : 0.0;

            // class name
            p.setPen(i == bestIdx ? Qt::white : QColor(180, 180, 180));
            if (i == bestIdx) {
                QFont bf = font;
                bf.setBold(true);
                p.setFont(bf);
            } else {
                p.setFont(font);
            }
            QString name = (i < names_.size()) ? names_[i] : QString::number(i);
            p.drawText(QRect(0, y, labelW, barH), Qt::AlignVCenter | Qt::AlignLeft, name);

            // bar
            int maxBarW = width() - labelW - pctW - 4;
            int bw = (int)(prob * maxBarW);
            QColor barColor = (i == bestIdx) ? QColor(70, 180, 100) : QColor(60, 100, 160);
            p.fillRect(labelW, y + 2, bw, barH - 4, barColor);
            p.setPen(QPen(QColor(80, 80, 80), 0.5));
            p.drawRect(labelW, y + 2, maxBarW, barH - 4);

            // percentage
            p.setPen(i == bestIdx ? Qt::white : QColor(160, 160, 160));
            p.setFont(font);
            p.drawText(QRect(labelW + maxBarW + 2, y, pctW, barH),
                       Qt::AlignVCenter | Qt::AlignRight,
                       QString("%1%").arg((int)(prob * 100)));
        }
    }

private:
    QVector<double>  probs_;
    QVector<QString> names_;
};

// ── Form with spinboxes ───────────────────────────────────────────────────────

class FeatureInputWidget::FormWidget : public QWidget {
public:
    explicit FormWidget(QWidget* parent = nullptr) : QWidget(parent) {}

    void rebuild(const DatasetMeta& meta) {
        meta_ = meta;
        spinBoxes_.clear();

        // remove old layout
        if (layout()) {
            QLayoutItem* item;
            while ((item = layout()->takeAt(0)) != nullptr) {
                delete item->widget();
                delete item;
            }
            delete layout();
        }

        auto* form = new QFormLayout(this);
        form->setSpacing(4);
        form->setContentsMargins(4, 4, 4, 4);

        for (int i = 0; i < meta.featureNames.size(); ++i) {
            auto* box = new QDoubleSpinBox(this);
            box->setDecimals(3);

            double lo = (i < meta.minVals.size()) ? meta.minVals[i] : 0.0;
            double hi = (i < meta.maxVals.size()) ? meta.maxVals[i] : 1.0;
            double mid = (lo + hi) / 2.0;
            double step = std::max(0.001, (hi - lo) / 100.0);

            box->setRange(lo, hi);
            box->setSingleStep(step);
            box->setValue(mid);

            form->addRow(meta.featureNames[i], box);
            spinBoxes_.push_back(box);
        }
    }

    Eigen::VectorXd readNormalized() const {
        Eigen::VectorXd v(spinBoxes_.size());
        for (int i = 0; i < spinBoxes_.size(); ++i) {
            double raw   = spinBoxes_[i]->value();
            double lo    = (i < meta_.minVals.size()) ? meta_.minVals[i] : 0.0;
            double hi    = (i < meta_.maxVals.size()) ? meta_.maxVals[i] : 1.0;
            double range = hi - lo;
            v[i] = (range < 1e-9) ? 0.0 : (raw - lo) / range;
        }
        return v;
    }

    bool hasInputs() const { return !spinBoxes_.isEmpty(); }

private:
    DatasetMeta              meta_;
    QVector<QDoubleSpinBox*> spinBoxes_;
};

// ── FeatureInputWidget ────────────────────────────────────────────────────────

FeatureInputWidget::FeatureInputWidget(QWidget* parent)
    : QWidget(parent)
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(4);

    auto* scroll = new QScrollArea(this);
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    form_ = new FormWidget(scroll);
    scroll->setWidget(form_);
    scroll->setMaximumHeight(200);
    layout->addWidget(scroll);

    auto* predictBtn = new QPushButton("Predict", this);
    connect(predictBtn, &QPushButton::clicked, this, &FeatureInputWidget::onPredict);
    layout->addWidget(predictBtn);

    bars_ = new BarsWidget(this);
    layout->addWidget(bars_);
}

void FeatureInputWidget::configure(const DatasetMeta& meta)
{
    meta_ = meta;
    probs_.clear();
    form_->rebuild(meta);
    bars_->setData({}, meta.classNames);
}

void FeatureInputWidget::setPredictor(
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> pred)
{
    predictor_ = std::move(pred);
}

void FeatureInputWidget::reset()
{
    probs_.clear();
    bars_->setData({}, meta_.classNames);
}

void FeatureInputWidget::onPredict()
{
    if (!predictor_ || !form_->hasInputs()) return;

    auto input = form_->readNormalized();
    auto result = predictor_(input);

    probs_.resize(result.size());
    for (int i = 0; i < result.size(); ++i)
        probs_[i] = result[i];

    bars_->setData(probs_, meta_.classNames);
}

} // namespace App
