#pragma once

#include <QObject>
#include <QColor>

namespace App
{

    struct GraphPalette {
        QColor widgetBg;
        QColor dialogBg;
        QColor networkBg;
        QColor canvasBg;
        QColor canvasBorder;
        QColor grid;
        QColor gridLabel;
        QColor title;
        QColor hint;
        QColor currentLine;
        QColor neuronDefault;
        QColor connection;
    };

    class ThemeManager : public QObject {
        Q_OBJECT
    public:
        enum class Theme { Dark, Light, Blue };

        static ThemeManager& instance();

        void apply(Theme theme);
        Theme current() const { return current_; }
        const GraphPalette& palette() const { return palette_; }
        QColor runColor(int index, int total) const;

    signals:
        void themeChanged();

    private:
        explicit ThemeManager(QObject* parent = nullptr);
        void buildPalette();

        Theme        current_ = Theme::Dark;
        GraphPalette palette_;
    };

} // namespace App