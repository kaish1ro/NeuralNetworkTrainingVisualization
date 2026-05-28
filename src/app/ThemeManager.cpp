#include "ThemeManager.h"

#include <QApplication>
#include <QFile>

namespace App
{

    ThemeManager& ThemeManager::instance()
    {
        static ThemeManager inst;
        return inst;
    }

    ThemeManager::ThemeManager(QObject* parent)
        : QObject(parent)
    {
        buildPalette();
    }

    void ThemeManager::apply(Theme theme)
    {
        current_ = theme;
        buildPalette();

        QString path;
        switch (theme) {
        case Theme::Dark:  path = "resources/styles/dark.qss";  break;
        case Theme::Light: path = "resources/styles/light.qss"; break;
        case Theme::Blue:  path = "resources/styles/blue.qss";  break;
        }

        QFile f(path);
        if (f.open(QFile::ReadOnly | QFile::Text))
            qApp->setStyleSheet(QString::fromUtf8(f.readAll()));

        emit themeChanged();
    }

    QColor ThemeManager::runColor(int index, int total) const
    {
        if (current_ == Theme::Light) {
            if (index == total - 1)
                return QColor(21, 101, 192);
            int t = (total > 1) ? index * 80 / (total - 1) : 0;
            return QColor(80 + t, 130 + t, 210, 200);
        }
        else {
            if (index == total - 1)
                return QColor(220, 228, 255);
            int t = (total > 1) ? index * 110 / (total - 1) : 0;
            return QColor(55 + t, 65 + t, 85 + t, 210);
        }
    }

    void ThemeManager::buildPalette()
    {
        switch (current_) {
        case Theme::Dark:
            palette_.widgetBg = QColor(26, 26, 26);
            palette_.dialogBg = QColor(18, 18, 18);
            palette_.networkBg = QColor(22, 22, 22);
            palette_.canvasBg = QColor(26, 26, 26);
            palette_.canvasBorder = QColor(60, 60, 60);
            palette_.grid = QColor(48, 48, 48);
            palette_.gridLabel = QColor(105, 105, 105);
            palette_.title = QColor(210, 210, 210);
            palette_.hint = QColor(72, 72, 72);
            palette_.currentLine = QColor(220, 228, 255);
            palette_.neuronDefault = QColor(80, 40, 255);
            //palette_.neuronDefault = QColor(155, 80, 255);
            palette_.connection = QColor(65, 65, 80);
            break;

        case Theme::Light:
            palette_.widgetBg = QColor(252, 252, 252);
            palette_.dialogBg = QColor(255, 255, 255);
            palette_.networkBg = QColor(245, 245, 245);
            palette_.canvasBg = QColor(255, 255, 255);
            palette_.canvasBorder = QColor(200, 200, 200);
            palette_.grid = QColor(220, 220, 220);
            palette_.gridLabel = QColor(150, 150, 150);
            palette_.title = QColor(40, 40, 40);
            palette_.hint = QColor(185, 185, 185);
            palette_.currentLine = QColor(21, 101, 192);
            palette_.neuronDefault = QColor(62, 180, 137);
            palette_.connection = QColor(180, 180, 190);
            break;

        case Theme::Blue:
            palette_.widgetBg = QColor(26, 26, 46);
            palette_.dialogBg = QColor(22, 33, 62);
            palette_.networkBg = QColor(18, 24, 58);
            palette_.canvasBg = QColor(22, 33, 62);
            palette_.canvasBorder = QColor(42, 42, 74);
            palette_.grid = QColor(42, 60, 90);
            palette_.gridLabel = QColor(100, 130, 165);
            palette_.title = QColor(200, 215, 235);
            palette_.hint = QColor(60, 80, 110);
            palette_.currentLine = QColor(200, 220, 255);
            palette_.neuronDefault = QColor(70, 140, 220);
            palette_.connection = QColor(40, 60, 100);
            break;
        }
    }

} // namespace App