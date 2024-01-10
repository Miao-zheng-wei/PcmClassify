#ifndef _H_PROGRESSDIALOG_H_
#define _H_PROGRESSDIALOG_H_

//CCLib
#include "GenericProgressCallback.h"

//Qt
#include <QtWidgets/QProgressDialog>
#include <QtCore/QAtomicInt>
#include <QtCore/QTimer>

//! Graphical progress indicator (thread-safe)
/** Implements the GenericProgressCallback interface, in order
to be passed to the CCLib algorithms (check the
CCLib documentation for more information about the
inherited methods).
**/
class ProgressDialog : public QProgressDialog, public GenericProgressCallback
{
    Q_OBJECT

public:
    //! Default constructor
    /** By default, a cancel button is always displayed on the
	progress interface. It is only possible to activate or
	deactivate this button. Sadly, the fact that this button is
	activated doesn't mean it will be possible to stop the ongoing
	process: it depends only on the client algorithm implementation.
	\param cancelButton activates or deactivates the cancel button
	\param parent parent widget
	**/
    ProgressDialog(bool cancelButton = false,
                   QWidget *parent = 0);

    //! Destructor (virtual)
    virtual ~ProgressDialog() {}

    //inherited method
    virtual void update(float percent) override;
    inline virtual void setMAXRange(int maxvalue) { setRange(0, maxvalue); }
    inline virtual void setMethodTitle(const char *methodTitle) override { setMethodTitle(QString::fromLocal8Bit(methodTitle)); }
    inline virtual void setInfo(const char *infoStr) override { setInfo(QString::fromLocal8Bit(infoStr)); }
    inline virtual bool isCancelRequested() override { return wasCanceled(); }
    virtual void start() override;
    virtual void stop() override;

    //! setMethodTitle with a QString as argument
    virtual void setMethodTitle(QString methodTitle);
    //! setInfo with a QString as argument
    virtual void setInfo(QString infoStr);

protected slots:

    //! Refreshes the progress
    /** Should only be called in the main Qt thread!
	This slot is automatically called by 'update' (in Qt::QueuedConnection mode).
	**/
    void refresh();

signals:

    //! Schedules a call to refresh
    void scheduleRefresh();

protected:
    //! Current progress value (percent)
    QAtomicInt m_currentValue;

    //! Last displayed progress value (percent)
    QAtomicInt m_lastRefreshValue;
};

#endif //CC_PROGRESS_DIALOG_HEADER
