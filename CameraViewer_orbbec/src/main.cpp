#include "mainwindow.h"

#include <QApplication>
#include <QMetaType>
#include <QMessageBox>
#include <opencv2/opencv.hpp>
#include "persontracker.h"

#include <csignal>
#include <cstdio>
#include <cstring>
#include <execinfo.h>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

// ============================================================================
// 崩溃捕获：发生 SIGSEGV/SIGABRT 等时，把调用栈写入可执行文件目录下的
// crash_backtrace.log，便于定位崩溃原因。
// ============================================================================
static void writeCrashBacktrace(int sig)
{
    int fd = -1;

    // 目标文件：<可执行文件目录>/crash_backtrace.log
    char exePath[4096];
    ssize_t n = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
    if (n > 0) {
        exePath[n] = '\0';
        char *slash = strrchr(exePath, '/');
        if (slash) {
            const size_t len = static_cast<size_t>(slash - exePath);
            char path[4200];
            if (len + 32 < sizeof(path)) {
                memcpy(path, exePath, len);
                strcpy(path + len, "/crash_backtrace.log");
                fd = open(path, O_CREAT | O_WRONLY | O_APPEND, 0644);
            }
        }
    }
    if (fd < 0) fd = STDERR_FILENO;

    char header[256];
    int hn = snprintf(header, sizeof(header),
                      "\n===== CRASH signal=%d (%s) =====\n",
                      sig, strsignal(sig));
    if (hn > 0) {
        ssize_t w = write(fd, header, static_cast<size_t>(hn));
        (void)w;
    }

    void *frames[64];
    const int count = backtrace(frames, 64);
    backtrace_symbols_fd(frames, count, fd);

    if (fd != STDERR_FILENO) close(fd);

    // 恢复默认处理并重新抛出以生成 core dump
    signal(sig, SIG_DFL);
    raise(sig);
}

static void installCrashHandler()
{
    signal(SIGSEGV, writeCrashBacktrace);
    signal(SIGABRT, writeCrashBacktrace);
    signal(SIGBUS,  writeCrashBacktrace);
    signal(SIGILL,  writeCrashBacktrace);
    signal(SIGFPE,  writeCrashBacktrace);
}

// ============================================================================
// 单实例保护：同时运行多个同类实例会抢占同一个相机 / 日志与记录文件，
// 极易导致崩溃。这里用 flock 文件锁，进程结束后（含崩溃）锁自动释放。
//
// 锁文件按“可执行文件名”区分：不同变体（如 CameraViewer 与
// CameraViewer_orbbec）使用各自独立的锁，可同时运行互不干扰；
// 而同一个变体的重复启动仍会被互斥拦截。
// ============================================================================
static int g_instanceLockFd = -1;

static bool acquireSingleInstance()
{
    // 用可执行文件名作为锁文件名，避免不同变体互相误判为“已在运行”。
    const char *exeName = "CameraViewer_orbbec";
    char exePath[4096];
    ssize_t n = ::readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
    if (n > 0) {
        exePath[n] = '\0';
        const char *slash = strrchr(exePath, '/');
        if (slash && slash[1] != '\0') {
            exeName = slash + 1;
        }
    }

    // 前缀 cv-singleton- 以避免与历史遗留的 /tmp/.CameraViewer.lock 冲突
    // （旧版本二进制在运行时会占用该名字，导致新版本误判“已在运行”）。
    char lockPath[4200];
    snprintf(lockPath, sizeof(lockPath), "/tmp/cv-singleton-%s.lock", exeName);

    g_instanceLockFd = ::open(lockPath, O_CREAT | O_RDWR, 0644);
    if (g_instanceLockFd < 0) {
        // 无法创建锁文件时不阻止启动
        return true;
    }
    if (::flock(g_instanceLockFd, LOCK_EX | LOCK_NB) != 0) {
        ::close(g_instanceLockFd);
        g_instanceLockFd = -1;
        return false;
    }
    return true;
}

int main(int argc, char *argv[])
{
    installCrashHandler();

    QApplication a(argc, argv);

    // 单实例互锁：仅拦截同一可执行文件（变体）的重复启动，
    // 不同变体使用各自的锁文件，可同时运行。
    if (!acquireSingleInstance()) {
        QMessageBox::warning(
            nullptr, "CameraViewer_orbbec 已在运行",
            "检测到本程序已经在运行中。\n\n"
            "请勿重复启动：同时运行多个同类实例会抢占同一个相机与记录文件，"
            "并可能导致程序崩溃。");
        return 1;
    }

    // 注册自定义类型以支持跨线程信号槽
    qRegisterMetaType<cv::Mat>("cv::Mat");
    qRegisterMetaType<cv::Rect>("cv::Rect");
    qRegisterMetaType<QVector<TrackedPerson>>("QVector<TrackedPerson>");
    qRegisterMetaType<QVector<QPointF>>("QVector<QPointF>");

    MainWindow w;
    w.show();
    return a.exec();
}
