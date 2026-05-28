#include "app/MainWindow.h"
#include "app/ThemeManager.h"
#include "app/LanguageManager.h"
#include "threading/AnalysisResult.h"

#include <QDir>
#include <QSettings>
#include <QApplication>

int main(int argc, char* argv[]) {
    QApplication::setStyle("Fusion");

    QApplication app(argc, argv);

    app.setWindowIcon(QIcon("resources/icons/app.ico"));

    QApplication::setOrganizationName("NNVisualizer");
    QApplication::setApplicationName("NeuralNetworkVisualizer");

    qRegisterMetaType<App::AnalysisResult>();

    QSettings s;
    auto theme = static_cast<App::ThemeManager::Theme>(
        s.value("ui/theme", static_cast<int>(App::ThemeManager::Theme::Dark)).toInt());
    auto lang = static_cast<App::LanguageManager::Language>(
        s.value("ui/language", static_cast<int>(App::LanguageManager::Language::English)).toInt());

    App::ThemeManager::instance().apply(theme);
    App::LanguageManager::instance().apply(lang);

    App::MainWindow window;
    window.show();

    return app.exec();
}
