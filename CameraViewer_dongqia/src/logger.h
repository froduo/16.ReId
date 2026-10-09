#ifndef LOGGER_H
#define LOGGER_H

#include <QString>
#include <QFile>
#include <QTextStream>
#include <QDateTime>
#include <QMutex>
#include <QDir>
#include <QDebug>

class Logger {
public:
    enum LogLevel { INFO, WARN, ERR };

    static Logger* getInstance() {
        static Logger instance;
        return &instance;
    }

    void log(const QString &message, LogLevel level = INFO) {
        QMutexLocker locker(&mutex);

        QString levelStr;
        switch (level) {
        case INFO: levelStr = "[INFO]"; break;
        case WARN: levelStr = "[WARN]"; break;
        case ERR:  levelStr = "[ERROR]"; break;
        }

        QString timeStr = QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss.zzz");
        QString logLine = QString("%1 %2 %3").arg(timeStr, levelStr, message);

        if (level == ERR) qCritical() << logLine;
        else qDebug() << logLine;

        if (logFile.isOpen()) {
            QTextStream out(&logFile);
            out << logLine << "\n";
            out.flush();
        }
    }

    void cleanOldLogs() {
        QDir dir("logs");
        if (!dir.exists()) return;

        dir.setFilter(QDir::Files | QDir::NoSymLinks);
        dir.setNameFilters(QStringList() << "run_log_*.txt");
        QFileInfoList list = dir.entryInfoList();
        QDateTime limit = QDateTime::currentDateTime().addDays(-14);
        int removedCount = 0;

        for (int i = 0; i < list.size(); ++i) {
            if (list.at(i).lastModified() < limit) {
                if (QFile::remove(list.at(i).absoluteFilePath())) {
                    removedCount++;
                }
            }
        }

        if (removedCount > 0) {
            qDebug() << "[Logger] Cleaned up" << removedCount << "old log files";
        }
    }

private:
    Logger() {
        QDir logDir;
        if (!logDir.exists("logs")) {
            logDir.mkdir("logs");
        }
        cleanOldLogs();

        QString fileName = QString("logs/run_log_%1.txt")
                               .arg(QDateTime::currentDateTime().toString("yyyyMMdd"));
        logFile.setFileName(fileName);
        logFile.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text);
    }

    ~Logger() {
        if (logFile.isOpen()) logFile.close();
    }

    QFile logFile;
    QMutex mutex;
};

#define LOG_INFO(msg) Logger::getInstance()->log(msg, Logger::INFO)
#define LOG_WARN(msg) Logger::getInstance()->log(msg, Logger::WARN)
#define LOG_ERR(msg)  Logger::getInstance()->log(msg, Logger::ERR)

#endif
