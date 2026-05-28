#pragma once

#include <QObject>
#include <QString>
#include <QTranslator>

namespace App
{

class LanguageManager : public QObject {
    Q_OBJECT
public:
    enum class Language { English, Russian };

    static LanguageManager& instance();

    void apply(Language lang);
    Language current() const { return current_; }

signals:
    void languageChanged();

private:
    explicit LanguageManager(QObject* parent = nullptr);

    QTranslator translator_;
    Language    current_ = Language::English;
};

} // namespace App
