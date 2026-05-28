#include "SettingsDialog.h"
#include "ThemeManager.h"
#include "LanguageManager.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QLabel>
#include <QPushButton>
#include <QFrame>
#include <QSettings>
#include <QMessageBox>
#include <QTextBrowser>
#include <QFile>

namespace App
{

SettingsDialog::SettingsDialog(QWidget* parent)
    : QDialog(parent)
{
    setWindowTitle(tr("Settings"));
    setMinimumWidth(340);
    setModal(true);
    setupUi();
}

void SettingsDialog::setupUi()
{
    auto* root = new QVBoxLayout(this);
    root->setSpacing(16);
    root->setContentsMargins(20, 20, 20, 20);

    auto* form = new QFormLayout();
    form->setHorizontalSpacing(16);
    form->setVerticalSpacing(10);

    languageBox = new QComboBox(this);
    languageBox->addItem(tr("English"), static_cast<int>(LanguageManager::Language::English));
    languageBox->addItem(tr("Русский"), static_cast<int>(LanguageManager::Language::Russian));
    if (LanguageManager::instance().current() == LanguageManager::Language::Russian)
        languageBox->setCurrentIndex(1);
    form->addRow(tr("Language:"), languageBox);

    themeBox = new QComboBox(this);
    themeBox->addItem(tr("Dark"),  static_cast<int>(ThemeManager::Theme::Dark));
    themeBox->addItem(tr("Light"), static_cast<int>(ThemeManager::Theme::Light));
    themeBox->addItem(tr("Blue"),  static_cast<int>(ThemeManager::Theme::Blue));
    switch (ThemeManager::instance().current()) {
        case ThemeManager::Theme::Dark:  themeBox->setCurrentIndex(0); break;
        case ThemeManager::Theme::Light: themeBox->setCurrentIndex(1); break;
        case ThemeManager::Theme::Blue:  themeBox->setCurrentIndex(2); break;
    }
    form->addRow(tr("Theme:"), themeBox);

    root->addLayout(form);

    auto* sep1 = new QFrame(this);
    sep1->setFrameShape(QFrame::HLine);
    sep1->setFrameShadow(QFrame::Plain);
    root->addWidget(sep1);

    auto* resetBtn = new QPushButton(tr("Reset preferences"), this);
    resetBtn->setToolTip(tr("Clears \"don't ask again\" flags, resets language and theme to defaults"));
    connect(resetBtn, &QPushButton::clicked, this, &SettingsDialog::onResetPreferences);
    root->addWidget(resetBtn);

    auto* docsBtn = new QPushButton(tr("Open documentation"), this);
    connect(docsBtn, &QPushButton::clicked, this, &SettingsDialog::onOpenDocumentation);
    root->addWidget(docsBtn);

    auto* sep2 = new QFrame(this);
    sep2->setFrameShape(QFrame::HLine);
    sep2->setFrameShadow(QFrame::Plain);
    root->addWidget(sep2);

    // за авторством меня
    auto* creditsLabel = new QLabel(this);
    creditsLabel->setText(
        "<b>Neural Network Visualizer</b><br>"
        "<span style='color: gray; font-size: 11px;'>"
        "Built with Qt 6 and Eigen.<br>"
        "Developed by kaish1ro</span>"
    );
    creditsLabel->setTextFormat(Qt::RichText);
    creditsLabel->setAlignment(Qt::AlignCenter);
    root->addWidget(creditsLabel);

    auto* sep3 = new QFrame(this);
    sep3->setFrameShape(QFrame::HLine);
    sep3->setFrameShadow(QFrame::Plain);
    root->addWidget(sep3);

    auto* closeBtn = new QPushButton(tr("Close"), this);
    connect(closeBtn, &QPushButton::clicked, this, &QDialog::accept);
    root->addWidget(closeBtn);

    connect(languageBox, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int) {
        auto lang = static_cast<LanguageManager::Language>(languageBox->currentData().toInt());
        LanguageManager::instance().apply(lang);
        QSettings().setValue("ui/language", languageBox->currentData().toInt());
    });

    connect(themeBox, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int) {
        auto theme = static_cast<ThemeManager::Theme>(themeBox->currentData().toInt());
        ThemeManager::instance().apply(theme);
        QSettings().setValue("ui/theme", themeBox->currentData().toInt());
    });
}

void SettingsDialog::onResetPreferences()
{
    QSettings s;
    s.remove("warnings/skipAll");
    s.remove("warnings/skipForDataset");

    LanguageManager::instance().apply(LanguageManager::Language::English);
    ThemeManager::instance().apply(ThemeManager::Theme::Dark);

    languageBox->setCurrentIndex(0);
    themeBox->setCurrentIndex(0);

    QMessageBox::information(this, tr("Reset"), tr("Preferences have been reset."));
}

void SettingsDialog::onOpenDocumentation()
{
    auto* dlg = new QDialog(this);
    dlg->setWindowTitle(tr("Documentation"));
    dlg->resize(820, 640);
    dlg->setAttribute(Qt::WA_DeleteOnClose);

    auto* browser = new QTextBrowser(dlg);
    browser->setOpenExternalLinks(true);

    QFile f(":/docs/README.md");
    if (f.open(QFile::ReadOnly | QFile::Text))
        browser->setMarkdown(QString::fromUtf8(f.readAll()));
    else
        browser->setPlainText(tr("Documentation file not found."));

    auto* layout = new QVBoxLayout(dlg);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(browser);

    auto* closeBtn = new QPushButton(tr("Close"), dlg);
    closeBtn->setFixedHeight(32);
    connect(closeBtn, &QPushButton::clicked, dlg, &QDialog::accept);
    auto* btnLayout = new QHBoxLayout();
    btnLayout->setContentsMargins(12, 8, 12, 12);
    btnLayout->addStretch();
    btnLayout->addWidget(closeBtn);
    layout->addLayout(btnLayout);

    dlg->exec();
}

} // namespace App
