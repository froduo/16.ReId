// 仅用于内存安全验证（ASan）的驱动测试：模拟一个人跨过虚拟门线。
// 覆盖 ZoneMonitor 采样/跨越/轨迹丢失结算，以及 PersonTracker + ReIDExtractor 融合路径。
#include "regionmanager.h"
#include "zonemonitor.h"
#include "persontracker.h"
#include "reidextractor.h"

#include <opencv2/opencv.hpp>
#include <QCoreApplication>
#include <cstdio>

int main(int argc, char **argv)
{
    QCoreApplication app(argc, argv);

    const int W = 1920, H = 1080;
    cv::Mat frame(H, W, CV_8UC3, cv::Scalar(35, 35, 35));

    RegionManager rm;
    DetectionRegion r;
    r.id = 1;
    r.name = "test";
    r.polygon = { QPointF(0.30, 0.20), QPointF(0.70, 0.20),
                  QPointF(0.70, 0.80), QPointF(0.30, 0.80) };
    rm.addRegion(r);

    ZoneMonitor zm;
    zm.setChannelZones(rm.regions());
    int board = 0, alight = 0;
    QObject::connect(&zm, &ZoneMonitor::boardingEvent, [&board](const ZoneEvent &) { ++board; });
    QObject::connect(&zm, &ZoneMonitor::alightingEvent, [&alight](const ZoneEvent &) { ++alight; });

    // 自顶向下跨越门线（y: 0.30 -> 0.75）
    const int trackId = 7;
    cv::Point2f prev(0.5f * W, 0.30f * H);
    for (int i = 0; i <= 45; ++i) {
        const float ny = 0.30f + 0.45f * (i / 45.0f);
        TrackedPerson p;
        p.trackId = trackId;
        p.detected = true;
        p.center = cv::Point2f(0.5f * W, ny * H);
        p.prevCenter = prev;
        p.bbox = cv::Rect(static_cast<int>(0.5f * W - 90), static_cast<int>(ny * H - 140), 180, 280);
        prev = p.center;
        QVector<TrackedPerson> persons; persons.append(p);
        zm.processFrame(persons, frame, &rm, W, H);
    }

    // 轨迹丢失（人从画面消失）→ 触发 finalizeLost
    QVector<TrackedPerson> none;
    zm.processFrame(none, frame, &rm, W, H);

    // 反向跨越（y: 0.75 -> 0.30）验证下车
    prev = cv::Point2f(0.5f * W, 0.75f * H);
    for (int i = 0; i <= 45; ++i) {
        const float ny = 0.75f - 0.45f * (i / 45.0f);
        TrackedPerson p;
        p.trackId = 8;
        p.detected = true;
        p.center = cv::Point2f(0.5f * W, ny * H);
        p.prevCenter = prev;
        p.bbox = cv::Rect(static_cast<int>(0.5f * W - 90), static_cast<int>(ny * H - 140), 180, 280);
        prev = p.center;
        QVector<TrackedPerson> persons; persons.append(p);
        zm.processFrame(persons, frame, &rm, W, H);
    }
    zm.processFrame(none, frame, &rm, W, H);

    // 追踪器 + ReID 融合路径
    PersonTracker tracker(0.3f, 30);
    ReIDExtractor reid;
    for (int i = 0; i < 40; ++i) {
        cv::Mat f = frame.clone();
        const int x = 700 + i * 10, y = 380;
        cv::rectangle(f, cv::Rect(x, y, 120, 260), cv::Scalar(200, 180, 160), -1);
        QVector<cv::Rect> dets; dets.append(cv::Rect(x, y, 120, 260));
        QVector<TrackedPerson> persons = tracker.update(dets, f, &reid);
        (void)persons;
    }

    printf("RESULT board=%d alight=%d (expect board>=1, alight>=1)\n", board, alight);
    return 0;
}
