#include "mainwindow.h"
#include "flowaggregator.h"
#include "httpuplink.h"

#include <QApplication>
#include <QMetaType>
#include <QDateTime>
#include <QJsonDocument>
#include <QJsonObject>
#include <QCoreApplication>
#include <QTimer>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include "persontracker.h"
#include "persondetector.h"
#include "reidextractor.h"
#include "visitorlog.h"

#include <QFile>

/**
 * @brief 检测+跟踪+ReID 自检：用一张真实图片跑一遍管线，用于复现/定位崩溃
 * 运行： ./CameraViewer --selftest-detect /path/to/frame.jpg
 */
static int runDetectSelfTest(const QString &imagePath)
{
    cv::Mat frame = cv::imread(imagePath.toStdString());
    if (frame.empty()) {
        fprintf(stderr, "cannot read image: %s\n", imagePath.toUtf8().constData());
        return 3;
    }
    printf("FRAME=%dx%d\n", frame.cols, frame.rows);

    PersonDetector det;
    const QString model = QCoreApplication::applicationDirPath()
                          + "/model/yolo26n_trt10_x86.engine";
    const bool ok = det.loadModel(model, 0.45f, 0.5f);
    printf("MODEL_LOADED=%d TRT=%d\n", ok ? 1 : 0, det.isTrtBackend() ? 1 : 0);

    PersonTracker tracker(0.3f, 30);
    tracker.setKalmanEnabled(true);
    tracker.setReidMatchingEnabled(true);
    tracker.setReidMatchThreshold(0.65f);
    tracker.setGalleryMaxAge(10000);
    ReIDExtractor reid;

    for (int i = 0; i < 5; ++i) {
        QVector<cv::Rect> rects = det.detect(frame);
        printf("ITER=%d DETS=%d\n", i, int(rects.size()));
        fflush(stdout);
        QVector<TrackedPerson> persons = tracker.update(rects, frame, &reid);
        printf("  TRACKS=%d", int(persons.size()));
        for (const auto &p : persons) {
            printf(" [id=%d feat=%dx%d]", p.trackId,
                   p.reidFeature.cols, p.reidFeature.rows);
        }
        printf("\n");
        fflush(stdout);
    }
    printf("DETECT_SELFTEST=PASS\n");
    return 0;
}

/**
 * @brief 客流上行自检：用真实代码路径验证协议报文与「精准识别某人的上车/下车」
 *
 * 运行： ./CameraViewer --selftest-uplink
 * 该模式不启动 GUI/相机/网络，注入合成事件后打印并自校验协议不变量。
 */
static int runUplinkSelfTest()
{
    using namespace DongqiaFlow;

    FlowAggregator agg;
    agg.setMasterSn("CAM-SELFTEST");
    agg.setDeviceSns(QStringList{"CAM-SELFTEST"});
    agg.setStatBasis("enter");
    agg.setTimezone("Asia/Shanghai", 28800, false);

    QVector<IntervalAggregate> intervals;
    QVector<PersonFlowEvent> persons;
    QObject::connect(&agg, &FlowAggregator::intervalReady,
                     [&](const IntervalAggregate &x) { intervals.append(x); });
    QObject::connect(&agg, &FlowAggregator::personEventReady,
                     [&](const PersonFlowEvent &x) { persons.append(x); });

    // 时间基准放在过去，确保时段为已完成分钟
    const QDateTime base = QDateTime::currentDateTime().addSecs(-600);
    const QDateTime t0 = base;                  // V1 进入通道
    const QDateTime t1 = base.addSecs(8);       // V1 离开通道（B→A = 上车）
    const QDateTime t2 = base.addSecs(130);     // V1 再次进入
    const QDateTime t3 = base.addSecs(155);     // V1 再次离开（A→B = 下车）

    // 目标人物 V1：上车 + 下车
    agg.onBoarding(1, t0, t1, 0.93f, cv::Mat(), cv::Mat(), 1);
    agg.onAlighting(1, t2, t3, 0.91f, cv::Mat(), cv::Mat(), 1);
    // 另一顾客 V2：普通进店 + 离店
    agg.onEnter(2, t1, PersonKind::Customer, 0.88f);
    agg.onLeave(2, t2, PersonKind::Customer);
    // 门口其它方向
    agg.onPassBy(t1);
    agg.onTurnBack(t2);

    // ===== 协议自校验（每个统计块）=====
    int fails = 0;
    for (const auto &a : intervals) {
        if (a.endTsS != a.startTsS + 60) {
            printf("FAIL[%lld] end_ts != start+60\n", (long long)a.startTsS);
            fails++;
        }
        // 严格恒等式：raw = final + repeat + non_customer
        if (a.rawCount != a.finalCount + a.finalRepeat + a.nonCustomerCount) {
            printf("FAIL[%lld] identity raw=%d != final=%d + repeat=%d + nonC=%d\n",
                   (long long)a.startTsS, a.rawCount, a.finalCount,
                   a.finalRepeat, a.nonCustomerCount);
            fails++;
        }
        int hs = 0; for (const auto &r : a.rawHeight) hs += r.count;
        int ags = 0; for (const auto &r : a.rawAgeGender) ags += r.count;
        int fhs = 0; for (const auto &r : a.finalHeight) fhs += r.count;
        int fags = 0; for (const auto &r : a.finalAgeGender) fags += r.count;
        int nh = 0; for (const auto &r : a.nonCustomerPersonType) nh += r.count;
        int dh = 0; for (const auto &r : a.dwellRange) dh += r.count;
        if (hs != a.rawCount)   { printf("FAIL raw.height sum %d != %d\n", hs, a.rawCount); fails++; }
        if (ags != a.rawCount)  { printf("FAIL raw.agegender sum %d != %d\n", ags, a.rawCount); fails++; }
        if (fhs != a.finalCount){ printf("FAIL final.height sum %d != %d\n", fhs, a.finalCount); fails++; }
        if (fags != a.finalCount){ printf("FAIL final.agegender sum %d != %d\n", fags, a.finalCount); fails++; }
        if (nh != a.nonCustomerCount){ printf("FAIL noncust sum %d != %d\n", nh, a.nonCustomerCount); fails++; }
        if (dh != a.dwellCount) { printf("FAIL dwell sum %d != %d\n", dh, a.dwellCount); fails++; }
    }

    printf("INTERVALS=%d PERSONS=%d\n", int(intervals.size()), int(persons.size()));
    printf("SUMMARY_V1=%s\n", agg.summarizePerson(1).toUtf8().constData());
    printf("SUMMARY_V2=%s\n", agg.summarizePerson(2).toUtf8().constData());

    for (const auto &e : persons) {
        printf("PE|visitor=%d|type=%s|dir=%s|enter=%lld|exit=%lld|event=%lld|dwell=%d|sim=%.2f\n",
               e.visitorId, e.eventType.toUtf8().constData(),
               e.direction.toUtf8().constData(),
               (long long)e.enterTsS, (long long)e.exitTsS, (long long)e.eventTsS,
               e.dwellSeconds, double(e.reidSimilarity));
    }

    // 用真实 JSON 构建器输出一件报文，供抓包核对
    HttpUplink up;
    HttpUplink::Config uc;
    uc.masterSn = "CAM-SELFTEST";
    up.setConfig(uc);
    const IntervalAggregate sample = intervals.isEmpty() ? agg.snapshotCurrentInterval()
                                                         : intervals.last();
    const QJsonObject obj = up.buildIntervalJson(sample);
    printf("INTERVAL_JSON=%s\n",
           QJsonDocument(obj).toJson(QJsonDocument::Compact).constData());

    printf("SELFTEST_RESULT=%s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 2;
}

/**
 * @brief 去重自检：验证「同一个人只保留一个 ID」
 *
 * 运行： ./CameraViewer --selftest-dedup
 * 场景：
 *   阶段1 - 同一人被检测出两个高度重叠的框（检测分裂/抖动）
 *           → 轨迹级去重应合并为 1 条轨迹（mergeCount >= 1）
 *   阶段2 - 该人短暂消失（超过 maxMissed）后重新出现
 *           → 应恢复原 trackId（身份连续，不会产生新 ID）
 */
static int runDedupSelfTest()
{
    // 合成一帧：纯色背景 + 一个“人”（供 ReID 颜色直方图提取）
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(30, 30, 30));
    cv::rectangle(frame, cv::Rect(100, 100, 120, 260), cv::Scalar(200, 180, 90), cv::FILLED);

    PersonTracker tracker(0.3f, 30);
    tracker.setKalmanEnabled(true);
    tracker.setReidMatchingEnabled(true);
    tracker.setReidMatchThreshold(0.65f);
    tracker.setTrackMergeSimilarity(0.85f);
    ReIDExtractor reid;

    // ===== 阶段1：同一人两个重叠框 =====
    int maxTracks = 0;
    int firstTrackId = -1;
    for (int i = 0; i < 40; ++i) {
        const int dx = i * 4;
        QVector<cv::Rect> dets;
        dets.append(cv::Rect(100 + dx, 100, 120, 260));
        dets.append(cv::Rect(104 + dx, 102, 118, 258));   // 高度重叠的重复检测
        const QVector<TrackedPerson> persons = tracker.update(dets, frame, &reid);
        maxTracks = std::max(maxTracks, int(persons.size()));
        if (persons.size() != 1) {
            printf("  frame %2d: tracks=%d (期望 1)\n", i, int(persons.size()));
        }
        if (!persons.isEmpty()) firstTrackId = persons.first().trackId;
    }
    printf("PHASE1_MAX_TRACKS=%d MERGES=%d TRACK_ID=%d\n",
           maxTracks, tracker.mergeCount(), firstTrackId);

    // ===== 阶段2：丢失后重新出现（位置接近、外观相同）=====
    for (int i = 0; i < 40; ++i)
        tracker.update(QVector<cv::Rect>(), frame, &reid);
    printf("PHASE2_GALLERY=%d ACTIVE=%d\n",
           tracker.gallerySize(), tracker.activeTrackCount());

    QVector<cv::Rect> back;
    back.append(cv::Rect(100 + 39 * 4 + 8, 100, 120, 260));
    const QVector<TrackedPerson> p2 = tracker.update(back, frame, &reid);
    const int recoveredId = p2.isEmpty() ? -1 : p2.first().trackId;
    printf("PHASE2_RECOVERED_ID=%d\n", recoveredId);

    // ===== 阶段3：人员进出记录落盘（同一 ID：进入时间/方向 + 离开时间/方向 + 停留）=====
    const QString tmpDir = QStringLiteral("/tmp/reid_person_records_test");
    VisitorLog vlog;
    vlog.setSaveDir(tmpDir);

    VisitorLog::PersonRecordEntry rec;
    rec.visitorId = firstTrackId;
    rec.enterTime = QDateTime::currentDateTime().toString(Qt::ISODate);
    rec.enterDirection = QStringLiteral("boarding");
    rec.entrySide = 1;                       // B 端进入 = 上车
    rec.exitTime = QDateTime::currentDateTime().addSecs(12).toString(Qt::ISODate);
    rec.exitDirection = QStringLiteral("boarding");
    rec.exitSide = 0;                        // A 端离开
    rec.dwellSeconds = 12;
    rec.regionId = 1;
    vlog.logPersonRecord(rec);

    QFile csv(tmpDir + "/person_records.csv");
    QString csvContent;
    if (csv.open(QIODevice::ReadOnly | QIODevice::Text))
        csvContent = QString::fromUtf8(csv.readAll());
    printf("PERSON_RECORDS_CSV_BEGIN\n%s\nPERSON_RECORDS_CSV_END\n",
           csvContent.toUtf8().constData());

    const bool csvOk = csvContent.contains(QStringLiteral("visitor_id,enter_time"))
                       && csvContent.contains(QStringLiteral("boarding"))
                       && csvContent.contains(QStringLiteral(",B,"))
                       && csvContent.contains(QStringLiteral(",A,"))
                       && csvContent.contains(QStringLiteral(",12,"));

    const bool pass = (maxTracks <= 1) && (tracker.mergeCount() >= 1)
                      && (recoveredId == firstTrackId) && (firstTrackId > 0)
                      && csvOk;
    printf("DEDUP_SELFTEST=%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 2;
}

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    // 注册自定义类型以支持跨线程信号槽
    qRegisterMetaType<cv::Mat>("cv::Mat");
    qRegisterMetaType<cv::Rect>("cv::Rect");
    qRegisterMetaType<QVector<TrackedPerson>>("QVector<TrackedPerson>");
    qRegisterMetaType<QVector<QPointF>>("QVector<QPointF>");

    // 无头删除历史上下车记录：
    //   --clear-history          仅删除记录（CSV/JSON + 内存状态）
    //   --clear-history-images   同时删除抓拍图片
    if (a.arguments().contains(QStringLiteral("--clear-history")) ||
        a.arguments().contains(QStringLiteral("--clear-history-images"))) {
        MainWindow w;
        const QString summary = w.clearHistoryRecords(
            a.arguments().contains(QStringLiteral("--clear-history-images")));
        printf("CLEAR_HISTORY_RESULT=%s\n", summary.toUtf8().constData());
        return 0;
    }

    // 客流上行协议 / 单人上下车识别 自检模式
    if (a.arguments().contains(QStringLiteral("--selftest-uplink"))) {
        return runUplinkSelfTest();
    }

    // 去重自检（同一人只保留一个 ID）
    if (a.arguments().contains(QStringLiteral("--selftest-dedup"))) {
        return runDedupSelfTest();
    }

    // 检测+跟踪+ReID 自检（复现/定位崩溃）
    const int detIdx = a.arguments().indexOf(QStringLiteral("--selftest-detect"));
    if (detIdx >= 0 && detIdx + 1 < a.arguments().size()) {
        return runDetectSelfTest(a.arguments().at(detIdx + 1));
    }

    MainWindow w;
    w.show();

    // 调试：自动点击「启动采集」（需图形平台；offscreen 亦可用于无头复现）
    if (a.arguments().contains(QStringLiteral("--autostart"))) {
        QTimer::singleShot(500, [&w]() {
            QMetaObject::invokeMethod(&w, "onToggleCapture", Qt::QueuedConnection);
        });
    }

    return a.exec();
}
