// 命令行离线复核 + 轨迹特征导出工具
//
// 用途：
//   1) 对「人物片段」视频复现工程内判定逻辑（检测 → 跟踪 → ZoneMonitor 穿越判定），
//      给出上/下车结论（严格完整穿越优先，否则按轨迹方向做宽松判定）；
//   2) 导出每条轨迹的特征 CSV（--csv），用于训练「上下车分类模型」。
//
// 用法：
//   ./offline_clip_check <视频文件或目录> [step=2] [模型路径] [regions.json] [--csv out.csv]
// 默认：
//   模型 = build-CameraViewer-main-Desktop-Debug/model/yolo26n_trt10_x86.engine
//   区域 = build-CameraViewer-main-Desktop-Debug/regions.json

#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QMap>
#include <QPair>
#include <QStringList>
#include <QTextStream>
#include <QSettings>
#include <cstdio>
#include <memory>

#include "persondetector.h"
#include "persontracker.h"
#include "reidextractor.h"
#include "regionmanager.h"
#include "zonemonitor.h"
#include "logger.h"

int main(int argc, char **argv)
{
    QCoreApplication app(argc, argv);
    const QStringList args = app.arguments();
    if (args.size() < 2) {
        printf("用法: %s <视频文件或目录> [step=2] [模型路径] [regions.json] [--csv out.csv]\n",
               qPrintable(args[0]));
        return 2;
    }

    // 可选参数：--csv <path>
    QString csvPath;
    QStringList pos;
    for (int i = 1; i < args.size(); ++i) {
        if (args[i] == QStringLiteral("--csv") && i + 1 < args.size()) {
            csvPath = args[++i];
        } else {
            pos << args[i];
        }
    }
    if (pos.isEmpty()) { printf("缺少视频参数\n"); return 2; }

    const QString target = pos[0];
    const int step = (pos.size() > 1) ? qMax(1, pos[1].toInt()) : 2;
    const QString model = (pos.size() > 2)
            ? pos[2]
            : QStringLiteral("build-CameraViewer-main-Desktop-Debug/model/yolo26n_trt10_x86.engine");
    const QString regionsPath = (pos.size() > 3)
            ? pos[3]
            : QStringLiteral("build-CameraViewer-main-Desktop-Debug/regions.json");
    // 可选第 5 参数：过滤目标类型 person|head（缺省从 config.ini 或模型文件名推断）
    const QString cfgKind = (pos.size() > 4) ? pos[4].trimmed() : QString();
    // 可选第 6 参数：letterbox|squash（TRT 预处理模式；缺省按 head→squash 推断）
    const QString cfgBox = (pos.size() > 5) ? pos[5].trimmed() : QString();
    bool cfgLetterbox = true;
    {
        QSettings cfg(QCoreApplication::applicationDirPath() + "/config.ini", QSettings::IniFormat);
        if (!cfgBox.isEmpty())
            cfgLetterbox = (cfgBox.compare("letterbox", Qt::CaseInsensitive) == 0);
        else if (cfg.contains("Inference/Letterbox"))
            cfgLetterbox = cfg.value("Inference/Letterbox").toBool();
        else
            cfgLetterbox = !(cfgKind.compare("head", Qt::CaseInsensitive) == 0);
    }

    RegionManager rm;
    rm.loadRegions(regionsPath);
    if (rm.regions().isEmpty()) {
        printf("!! 未加载到检测区域: %s （无法判定）\n", qPrintable(regionsPath));
        return 3;
    }
    float zoneTop = 2.0f, zoneBot = -1.0f, zoneLeft = 2.0f, zoneRight = -1.0f;
    {
        // 注意：regions() 按值返回，直接对返回值取引用会绑定到「临时对象」，
        // 语句结束后临时对象析构 → r0 悬垂（heap-use-after-free）。此处按值拷贝。
        const DetectionRegion r0 = rm.regions().first();
        for (const QPointF &pt : r0.polygon) {
            zoneTop = qMin(zoneTop, float(pt.y()));  zoneBot = qMax(zoneBot, float(pt.y()));
            zoneLeft = qMin(zoneLeft, float(pt.x())); zoneRight = qMax(zoneRight, float(pt.x()));
        }
        printf("区域: regionId=%d  x[%.3f~%.3f] y[%.3f~%.3f]\n", r0.id, zoneLeft, zoneRight, zoneTop, zoneBot);
    }

    PersonDetector det;
    // TRT 预处理模式：head 模型建议 squash（Letterbox=false）
    det.setLetterbox(cfgLetterbox);
    // 置信度：head 模式用 HeadConfThreshold（更高，抑制误检），否则用 ConfThreshold
    float confTh = 0.45f;
    {
        QSettings cfg(QCoreApplication::applicationDirPath() + "/config.ini", QSettings::IniFormat);
        if (cfgKind.compare(QStringLiteral("head"), Qt::CaseInsensitive) == 0)
            confTh = cfg.value("Inference/HeadConfThreshold", 0.55).toFloat();
        else
            confTh = cfg.value("Inference/ConfThreshold", 0.45).toFloat();
        printf("检测置信度阈值: %.2f\n", confTh);
    }
    if (!det.loadModel(model, confTh, 0.5f)) {
        printf("!! 模型加载失败: %s\n", qPrintable(model));
        return 4;
    }
    det.setMinSizeFilterEnabled(true);
    // 依据 config.ini 的 Inference/ModelKind 选择过滤参数：
    // head 头部检测框小且近似方形，沿用人体参数（minBoxHeightPx=90 等）会被全部误杀。
    {
        QString kindName = cfgKind;   // 命令行 5th 参数优先
        QSettings cfg(QCoreApplication::applicationDirPath() + "/config.ini",
                      QSettings::IniFormat);
        if (kindName.isEmpty())
            kindName = cfg.value("Inference/ModelKind").toString();
        if (kindName.isEmpty())
            kindName = QFileInfo(model).fileName().contains("head", Qt::CaseInsensitive)
                           ? QStringLiteral("head") : QStringLiteral("person");

        if (kindName.compare(QStringLiteral("head"), Qt::CaseInsensitive) == 0) {
            det.setMinSizeFilter(cfg.value("Inference/HeadMinPersonHeightRatio", 0.0100).toFloat(),
                                 cfg.value("Inference/HeadMinPersonAreaRatio", 0.00020).toFloat());
            det.setMinBoxSizePx(cfg.value("Inference/HeadMinBoxWidthPx", 10).toInt(),
                                cfg.value("Inference/HeadMinBoxHeightPx", 10).toInt());
            det.setBoxAspectRange(cfg.value("Inference/HeadMinBoxAspect", 0.35).toFloat(),
                                  cfg.value("Inference/HeadMaxBoxAspect", 3.00).toFloat());
            printf("过滤参数: head 模式（头部检测）\n");
        } else {
            det.setMinSizeFilter(0.08f, 0.005f);
            det.setMinBoxSizePx(30, 90);
            det.setBoxAspectRange(0.12f, 1.00f);
            printf("过滤参数: person 模式（人体检测）\n");
        }
    }
    // TRT 预处理模式提示（head 模型建议 squash：letterbox 会使头部分数低于阈值）
    printf("TRT 预处理: %s\n", cfgLetterbox ? "letterbox(保持长宽比)" : "squash(等比拉伸)");
    printf("模型: %s (backend=%s)\n", qPrintable(model),
           det.isTrtBackend() ? "TensorRT" : (det.isDnnCudaActive() ? "OpenCV-DNN-CUDA" : "OpenCV-DNN-CPU"));

    ReIDExtractor reid;

    QStringList files;
    QFileInfo fi(target);
    if (fi.isDir()) {
        const auto list = QDir(target).entryInfoList(QStringList() << "*.mp4" << "*.avi",
                                                     QDir::Files, QDir::Name);
        for (const QFileInfo &f : list) files << f.absoluteFilePath();
    } else {
        files << target;
    }
    if (files.isEmpty()) { printf("!! 未找到视频文件\n"); return 5; }

    int totalBoarding = 0, totalAlighting = 0, totalNone = 0;

    for (const QString &path : files) {
        const QString vname = QFileInfo(path).fileName();
        printf("\n================ %s ================\n", qPrintable(vname));

        cv::VideoCapture cap(path.toStdString());
        if (!cap.isOpened()) { printf("!! 无法打开视频\n"); continue; }
        double srcFps = cap.get(cv::CAP_PROP_FPS);
        if (!(srcFps > 1.0)) srcFps = 25.0;

        std::unique_ptr<PersonTracker> tracker(new PersonTracker(0.3f, 75));
        tracker->setIouThreshold(0.15f);
        tracker->setReidMatchThreshold(0.62f);

        ZoneMonitor zm;
        zm.setReidExtractor(&reid);
        zm.setConfirmFrames(1);
        zm.setStartConfirmFrames(1);
        zm.setReverseCooldownMs(0);
        zm.setMinOverlapArea(0.0002f);
        zm.setChannelZones(rm.regions());

        QVector<QString> events;
        QObject::connect(&zm, &ZoneMonitor::boardingEvent, [&events](const ZoneEvent &e) {
            const QString s = QString("BOARDING  track=%1 startY=%2 endY=%3 dy=%4 frames=%5")
                                  .arg(e.trackId).arg(e.startNormY, 0, 'f', 3)
                                  .arg(e.endNormY, 0, 'f', 3).arg(e.moveDy, 0, 'f', 3)
                                  .arg(e.trackFrames);
            events << s;
            printf("  ★ %s\n", qPrintable(s));
        });
        QObject::connect(&zm, &ZoneMonitor::alightingEvent, [&events](const ZoneEvent &e) {
            const QString s = QString("ALIGHTING track=%1 startY=%2 endY=%3 dy=%4 frames=%5")
                                  .arg(e.trackId).arg(e.startNormY, 0, 'f', 3)
                                  .arg(e.endNormY, 0, 'f', 3).arg(e.moveDy, 0, 'f', 3)
                                  .arg(e.trackFrames);
            events << s;
            printf("  ★ %s\n", qPrintable(s));
        });

        // 轨迹统计（仅真实检测框）
        QMap<int, QPair<float, float>> yRange;      // trackId -> (minCy, maxCy)
        QMap<int, QPair<float, float>> firstLast;   // trackId -> (firstCy, lastCy)
        QMap<int, int> trackFrames;                 // trackId -> 检出帧数
        QMap<int, int> trackFirstProc;              // trackId -> 首次出现的处理帧号
        QMap<int, int> trackLastProc;               // trackId -> 末次出现的处理帧号
        QMap<int, double> cxSum;                    // trackId -> sum(cx)
        QMap<int, int> cxCnt;

        int idx = 0, proc = 0;
        qint64 sumDet = 0, sumTrk = 0;
        cv::Mat frame;
        while (cap.read(frame)) {
            ++idx;
            if (idx % step) continue;
            ++proc;

            const qint64 t0 = QDateTime::currentMSecsSinceEpoch();
            QVector<cv::Rect> dets = det.detect(frame);
            const qint64 tDet = QDateTime::currentMSecsSinceEpoch() - t0;

            const qint64 t1 = QDateTime::currentMSecsSinceEpoch();
            QVector<TrackedPerson> persons = tracker->update(dets, frame, &reid);
            const qint64 tTrk = QDateTime::currentMSecsSinceEpoch() - t1;
            sumDet += tDet; sumTrk += tTrk;

            for (const TrackedPerson &p : persons) {
                if (frame.rows <= 0 || frame.cols <= 0) break;
                if (!p.detected) continue;          // 只用真实检测框
                const float cy = p.center.y / float(frame.rows);
                const float cx = p.center.x / float(frame.cols);

                auto it = yRange.find(p.trackId);
                if (it == yRange.end()) {
                    yRange.insert(p.trackId, qMakePair(cy, cy));
                    firstLast.insert(p.trackId, qMakePair(cy, cy));
                    trackFirstProc.insert(p.trackId, proc);
                } else {
                    it->first = qMin(it->first, cy);
                    it->second = qMax(it->second, cy);
                    firstLast[p.trackId].second = cy;
                }
                trackLastProc[p.trackId] = proc;
                trackFrames[p.trackId] = trackFrames.value(p.trackId, 0) + 1;
                cxSum[p.trackId] = cxSum.value(p.trackId, 0.0) + cx;
                cxCnt[p.trackId] = cxCnt.value(p.trackId, 0) + 1;
            }

            zm.processFrame(persons, frame, &rm, frame.cols, frame.rows);
        }
        cap.release();

        printf("  处理帧: %d（原始 %d, step=%d）| 平均 detect=%.0fms, track=%.0fms\n",
               proc, idx, step,
               proc ? double(sumDet) / proc : 0.0,
               proc ? double(sumTrk) / proc : 0.0);

        int nb = 0, na = 0;
        for (const QString &e : events) {
            if (e.startsWith("BOARDING")) nb++;
            if (e.startsWith("ALIGHTING")) na++;
        }

        printf("  轨迹 %d 条:\n", yRange.size());
        for (auto it = yRange.constBegin(); it != yRange.constEnd(); ++it) {
            printf("    ID%d: y[%.3f~%.3f]\n", it.key(), it->first, it->second);
        }

        // ===== 结论：严格完整穿越优先，否则方向判定 =====
        QString verdict, suggest;
        if (nb > 0 && na == 0)      { verdict = "上车（从上往下完整穿越）"; suggest = "boarding"; totalBoarding++; }
        else if (na > 0 && nb == 0) { verdict = "下车（从下往上完整穿越）"; suggest = "alighting"; totalAlighting++; }
        else if (nb > 0 && na > 0)  { verdict = QString("多次穿越(上%1/下%2)").arg(nb).arg(na); suggest = "multi"; totalNone++; }
        else {
            const float zoneH = qMax(0.05f, zoneBot - zoneTop);
            const float needDy = qMax(0.08f, zoneH * 0.45f);
            int bestTid = -1; float bestDy = 0.0f;
            for (auto it = firstLast.constBegin(); it != firstLast.constEnd(); ++it) {
                const float dy = it.value().second - it.value().first;
                if (qAbs(dy) > qAbs(bestDy)) { bestDy = dy; bestTid = it.key(); }
            }
            if (bestTid > 0 && qAbs(bestDy) >= needDy) {
                verdict = (bestDy < 0)
                        ? QString("下车（方向判定：由下往上, ID%1 dy=%2）").arg(bestTid).arg(bestDy, 0, 'f', 3)
                        : QString("上车（方向判定：由上往下, ID%1 dy=%2）").arg(bestTid).arg(bestDy, 0, 'f', 3);
                suggest = (bestDy < 0) ? "alighting" : "boarding";
                if (bestDy < 0) totalAlighting++; else totalBoarding++;
            } else {
                verdict = "未检测到完整穿越（方向位移亦不足）";
                suggest = "none";
                totalNone++;
            }
        }
        printf("  ==> 结论: %s\n", qPrintable(verdict));

        // ===== 导出轨迹特征 CSV（供训练上下车分类模型）=====
        if (!csvPath.isEmpty() && !yRange.isEmpty()) {
            QFile cf(csvPath);
            const bool needHeader = !cf.exists();
            if (cf.open(QIODevice::Append | QIODevice::Text)) {
                QTextStream ts(&cf);
                if (needHeader) {
                    ts << "video,track,frames,firstCy,lastCy,minCy,maxCy,dy,spanCy,meanCx,"
                          "durSec,zoneTop,zoneBot,overlapY,detRatio,suggest\n";
                }
                const double framePeriod = double(step) / srcFps;
                for (auto it = yRange.constBegin(); it != yRange.constEnd(); ++it) {
                    const int tid = it.key();
                    const float minY = it->first, maxY = it->second;
                    const float fY = firstLast.value(tid).first;
                    const float lY = firstLast.value(tid).second;
                    const int nf = trackFrames.value(tid, 0);
                    const double dur = double(trackLastProc.value(tid, 0) - trackFirstProc.value(tid, 0)) * framePeriod;
                    const float overlapY = qMax(0.0f, qMin(maxY, zoneBot) - qMax(minY, zoneTop));
                    const double meanCx = (cxCnt.value(tid, 0) > 0)
                            ? cxSum.value(tid, 0.0) / cxCnt.value(tid, 0) : 0.0;
                    const double detRatio = (proc > 0) ? double(nf) / proc : 0.0;

                    ts << vname << ',' << tid << ',' << nf << ','
                       << QString::number(fY, 'f', 4) << ',' << QString::number(lY, 'f', 4) << ','
                       << QString::number(minY, 'f', 4) << ',' << QString::number(maxY, 'f', 4) << ','
                       << QString::number(lY - fY, 'f', 4) << ','
                       << QString::number(maxY - minY, 'f', 4) << ','
                       << QString::number(meanCx, 'f', 4) << ','
                       << QString::number(dur, 'f', 3) << ','
                       << QString::number(zoneTop, 'f', 4) << ',' << QString::number(zoneBot, 'f', 4) << ','
                       << QString::number(overlapY, 'f', 4) << ','
                       << QString::number(detRatio, 'f', 4) << ','
                       << suggest << '\n';
                }
                cf.close();
            }
        }
    }

    printf("\n======== 汇总: 共 %d 段 | 判定上车 %d | 判定下车 %d | 未判定 %d ========\n",
           files.size(), totalBoarding, totalAlighting, totalNone);
    if (!csvPath.isEmpty()) {
        printf("轨迹特征已写入: %s\n", qPrintable(csvPath));
    }
    return 0;
}
