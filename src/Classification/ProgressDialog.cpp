#include "ProgressDialog.h"

//Qt
#include <QtCore/QCoreApplication>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QProgressBar>

ProgressDialog::ProgressDialog(bool showCancelButton,
                               QWidget *parent /*=0*/)
    : QProgressDialog(parent), m_currentValue(0), m_lastRefreshValue(-1)
{
    setAutoClose(true);
    setWindowModality(Qt::ApplicationModal);

    setRange(0, 100);
    setMinimumDuration(0);

    QPushButton *cancelButton = 0;
    if (showCancelButton)
    {
        cancelButton = new QPushButton("Cancel");
        cancelButton->setDefault(false);
        cancelButton->setFocusPolicy(Qt::NoFocus);
    }
    setCancelButton(cancelButton);

    connect(this, SIGNAL(scheduleRefresh()), this, SLOT(refresh()), Qt::QueuedConnection); //can't use DirectConnection here!
}

void ProgressDialog::refresh()
{
    int value = m_currentValue;
    if (m_lastRefreshValue != value)
    {
        m_lastRefreshValue = value;
        setValue(value); //See Qt doc: if the progress dialog is modal, setValue() calls QApplication::processEvents()
    }
}

void ProgressDialog::update(float percent)
{
    //thread-safe
    int value = static_cast<int>(percent);
    if (value != m_currentValue)
    {
        m_currentValue = value;
        emit scheduleRefresh();
        QCoreApplication::processEvents(); //we let the main thread breath (so that the call to 'refresh' can be performed)
    }
}

void ProgressDialog::setMethodTitle(QString methodTitle)
{
    setWindowTitle(methodTitle);
}

void ProgressDialog::setInfo(QString infoStr)
{
    setLabelText(infoStr);
    if (isVisible())
    {
        QProgressDialog::update();
        QCoreApplication::processEvents();
    }
}

void ProgressDialog::start()
{
    m_lastRefreshValue = -1;
    show();
    QCoreApplication::processEvents();
}

void ProgressDialog::stop()
{
    hide();
    QCoreApplication::processEvents();
}
