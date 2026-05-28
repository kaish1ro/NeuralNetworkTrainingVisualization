#pragma once

#include <QDialog>
#include <QComboBox>

namespace App
{

class SettingsDialog : public QDialog {
    Q_OBJECT
public:
    explicit SettingsDialog(QWidget* parent = nullptr);

private slots:
    void onResetPreferences();
    void onOpenDocumentation();

private:
    void setupUi();

    QComboBox* languageBox = nullptr;
    QComboBox* themeBox    = nullptr;
};

} // namespace App
