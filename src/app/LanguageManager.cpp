#include "LanguageManager.h"

#include <QApplication>
#include <QLocale>

namespace App
{

LanguageManager& LanguageManager::instance()
{
    static LanguageManager inst;
    return inst;
}

LanguageManager::LanguageManager(QObject* parent)
    : QObject(parent)
{}

void LanguageManager::apply(Language lang)
{
    current_ = lang;

    qApp->removeTranslator(&translator_);

    if (lang == Language::Russian) {
        if (translator_.load(":/i18n/nn_ru.qm"))
            qApp->installTranslator(&translator_);
    }

    emit languageChanged();
}

} // namespace App
