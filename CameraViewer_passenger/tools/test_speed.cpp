// 临时验证工具：验证 CameraController 解码 + 倍速 sleep 节奏是否按目标倍速推进。
#include <QCoreApplication>
#include <QElapsedTimer>
#include <QThread>
#include <cstdio>
#include <cstdlib>
#include "cameracontroller.h"

int main(int argc, char **argv)
{
    QCoreApplication app(argc, argv);
    if (argc < 2) {
        std::printf("usage: %s <video> [speed]\n", argv[0]);
        return 1;
    }
    const double speed = (argc >= 3) ? std::atof(argv[2]) : 6.0;

    CameraController cam;
    if (!cam.openVideoFile(argv[1])) {
        std::printf("open video failed\n");
        return 1;
    }
    const double fps = cam.fps();
    const qint64 targetMs = qMax<qint64>(1, static_cast<qint64>(1000.0 / (fps * speed)));
    std::printf("video fps=%.2f speed=%.1f target_interval=%lldms\n", fps, speed, (long long)targetMs);

    cv::Mat frame;
    int n = 0;
    QElapsedTimer total;
    total.start();
    QElapsedTimer pace;
    pace.start();

    while (n < 600 && cam.getFrame(frame)) {
        n++;
        const qint64 elapsed = pace.elapsed();
        if (elapsed < targetMs)
            QThread::msleep(static_cast<unsigned long>(targetMs - elapsed));
        pace.restart();
    }

    const qint64 elapsedTotal = total.elapsed();
    std::printf("frames=%d elapsed=%lldms actual_fps=%.2f expected_fps=%.2f\n",
                n, (long long)elapsedTotal,
                n * 1000.0 / qMax<qint64>(1, elapsedTotal), fps * speed);
    return 0;
}
