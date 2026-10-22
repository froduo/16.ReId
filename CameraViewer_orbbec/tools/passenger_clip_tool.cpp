// ============================================================================
// passenger_clip_tool.cpp —— 乘客上下车录像「离线批处理」工具
//
// 功能：
//   1) 自动解压输入目录下的录像压缩包（.rar 含分卷，支持 unrar/unar/7z/bsdtar）；
//   2) 对目录内所有视频逐帧做「检测 + 跟踪 + ReID」；
//   3) 用 ReID（优先深度 OSNet）把不同轨迹聚合为「同一个乘客」（visitor）；
//   4) 按每位乘客的「出现时段」裁剪原始视频，输出为独立片段；
//   5) 不同乘客保存到不同文件夹，并生成 manifest.csv 清单。
//
// 典型用法：
//   ./passenger_clip_tool --input "/path/25.乘客视频" --output ./passenger_clips \
//       --model ./model/yolo26n_trt10_x86.engine --deep ./model/osnet_x1_0_head_finetuned.onnx
//
// 输出结构：
//   <output>/passenger_1/<视频名>_clip0_<start>-<end>.mp4
//   <output>/passenger_2/...
//   <output>/manifest.csv
// ============================================================================

#include <QCoreApplication>
#include <QCommandLineParser>
#include <QCommandLineOption>
#include <QDir>
#include <QDirIterator>
#include <QFileInfo>
#include <QFile>
#include <QTextStream>
#include <QProcess>
#include <QStandardPaths>
#include <QElapsedTimer>
#include <QVector>
#include <QMap>
#include <QPair>
#include <QStringList>

#include <algorithm>
#include <cmath>

#include <opencv2/opencv.hpp>

#include "persondetector.h"
#include "persontracker.h"
#include "reidextractor.h"
#include "reiddeep.h"

// ============================================================================
// 工具函数
// ============================================================================
static bool isVideoExt(const QString &suffix)
{
    static const QStringList exts = {
        "mp4", "avi", "mkv", "mov", "ts", "flv", "m4v", "mpg", "mpeg", "wmv", "webm"
    };
    return exts.contains(suffix.toLower());
}

static QStringList listVideos(const QString &dir)
{
    QStringList out;
    if (dir.isEmpty() || !QDir(dir).exists())
        return out;
    QDirIterator it(dir, QDir::Files, QDirIterator::Subdirectories);
    while (it.hasNext()) {
        const QString p = it.next();
        if (isVideoExt(QFileInfo(p).suffix()))
            out << p;
    }
    out.sort();
    return out;
}

/// 解压目录下的录像压缩包（含 .part 分卷）。找到 unrar/unar/7z/bsdtar 之一即可。
static void extractArchives(const QString &dir, const QString &dest)
{
    QStringList rars;
    if (QDir(dir).exists()) {
        QDirIterator it(dir, QStringList() << "*.rar", QDir::Files, QDirIterator::Subdirectories);
        while (it.hasNext())
            rars << it.next();
    }
    if (rars.isEmpty()) {
        qInfo() << "[解压] 未发现 .rar 压缩包，跳过解压";
        return;
    }
    rars.sort();

    // 选择可用的解压工具
    QString tool;
    for (const QString &t : QStringList{"unrar", "unar", "7z", "7za", "bsdtar"}) {
        if (!QStandardPaths::findExecutable(t).isEmpty()) { tool = t; break; }
    }
    if (tool.isEmpty()) {
        qWarning() << "[解压] 未找到 unrar/unar/7z/7za/bsdtar，无法解压。"
                      "请安装（如：sudo apt install unrar）后重试，或用 --no-extract 指向已解压目录";
        return;
    }
    QDir().mkpath(dest);
    qInfo() << "[解压] 使用工具:" << tool << "| 发现" << rars.size() << "个 .rar 文件";

    QStringList doneGroups;
    for (const QString &r : rars) {
        const QString fileName = QFileInfo(r).fileName();
        const QString absDir   = QFileInfo(r).absolutePath();
        // 以 “xxx.partNN.rar” 的 xxx 作为分卷组名；非分卷则用文件名本身
        const QString groupKey = fileName.contains(".part")
                ? fileName.section(".part", 0, 0)
                : fileName;
        if (doneGroups.contains(groupKey))
            continue;
        doneGroups << groupKey;

        // 分卷必须从 part01 开始解压；若缺失则提示（常见于分卷未同步完整）
        QString firstVol = r;
        if (fileName.contains(".part")) {
            const QString p01 = absDir + "/" + groupKey + ".part01.rar";
            if (QFileInfo::exists(p01)) {
                firstVol = p01;
            } else {
                qWarning() << "[解压] 未找到" << groupKey + ".part01.rar"
                           << "（当前目录仅有部分分卷，解压可能失败或不完整）";
            }
        }

        QStringList args;
        if (tool == "unar")
            args << "-o" << dest << "-f" << firstVol;
        else if (tool == "bsdtar")
            args << "-xf" << firstVol << "-C" << dest;
        else if (tool.startsWith("7z"))
            args << "x" << "-y" << ("-o" + dest) << firstVol;
        else // unrar
            args << "x" << "-o+" << firstVol << (dest + "/");

        qInfo() << "[解压]" << tool << args.join(' ');
        QProcess proc;
        proc.start(tool, args);
        proc.waitForFinished(-1);   // 大文件解压可能耗时较长
        if (proc.exitCode() == 0) {
            qInfo() << "[解压] 完成:" << fileName;
        } else {
            qWarning() << "[解压] 失败(exit" << proc.exitCode() << "):" << fileName
                       << QString::fromLocal8Bit(proc.readAllStandardError()).trimmed();
        }
    }
}

// ============================================================================
// 单视频处理：检测/跟踪/ReID → 按乘客聚合 → 裁剪导出
// ============================================================================
struct PassengerAgg {
    int    id{-1};
    // 多锚点特征：保留该乘客少量代表性特征，匹配时取「最大相似度」。
    // 相比“单一质心 + EMA”，可避免质心随新样本漂移而逐渐“泛化”，
    // 从而防止不同乘客被错误并入同一身份（身份坍缩）。
    QVector<cv::Mat> anchors;               // 最多 ANCHOR_MAX 个
    QVector<QPair<int,int>> ranges;         // 出现帧区间 [start,end]（按 gap 自动分段）
    float  firstY{0.f};
    float  lastY{0.f};
    bool   hasY{false};
    static constexpr int ANCHOR_MAX = 8;
};

static void processVideo(const QString &videoPath,
                         const QString &outRoot,
                         PersonDetector &detector,
                         ReIDExtractor  &reid,
                         QVector<PassengerAgg> &passengers,   // 跨视频共享（同一乘客跨分段保持同一 ID）
                         int &nextPid,
                         float sameTh, float padSec, float gapSec, int step,
                         QStringList &manifestRows)
{
    cv::VideoCapture cap(videoPath.toStdString());
    if (!cap.isOpened()) {
        qWarning() << "[视频] 无法打开:" << videoPath;
        return;
    }

    const double fps = (cap.get(cv::CAP_PROP_FPS) > 1.0) ? cap.get(cv::CAP_PROP_FPS) : 25.0;
    const int    total = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    const cv::Size sz(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
                      static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
    const QString baseName = QFileInfo(videoPath).completeBaseName();

    qInfo() << "[视频]" << QFileInfo(videoPath).fileName()
            << QString("%1x%2").arg(sz.width).arg(sz.height)
            << QString("@%1fps").arg(fps, 0, 'f', 2)
            << "总帧数=" << total;

    // 每个视频使用独立跟踪器（与离线判定一致：IoU 0.15, maxMissed 75）
    PersonTracker tracker(0.15f, 75);
    tracker.setKalmanEnabled(true);
    tracker.setReidMatchingEnabled(true);
    tracker.setReidMatchThreshold(0.62f);
    tracker.setGalleryMaxAge(10000);

    // passengers / nextPid 由 main 传入并在所有视频间共享，保证“同一乘客”跨分段一致，
    // 且输出文件夹 passenger_<全局ID> 不会在多个视频间重名冲突。
    // trackId 每个视频都会从 1 重新分配，故映射表必须是本视频局部的。
    QMap<int,int> trackToVisitor;            // trackId → passengerId
    const int gapFrames = std::max(1, static_cast<int>(gapSec * fps));

    // 关键：身份(anchors)在所有视频间共享，但“出现时段 ranges”属于单个视频，
    // 必须在每个视频开始时清空，否则上一视频的时段会被重复导出到本视频
    // （表现为不同视频输出完全相同的片段 / 裁剪出错误内容）。
    for (auto &v : passengers) { v.ranges.clear(); v.hasY = false; }

    cv::Mat frame;
    int f = 0;
    QElapsedTimer timer; timer.start();

    while (cap.read(frame)) {
        if (!frame.empty() && (f % step == 0)) {
            const QVector<cv::Rect> dets = detector.detect(frame);
            const QVector<TrackedPerson> persons = tracker.update(dets, frame, &reid);

            for (const auto &p : persons) {
                if (!p.detected) continue;       // 只统计被真实检测到的轨迹

                // ---- 轨迹 → 乘客 身份映射（新轨迹才做一次特征比对）----
                int pid = -1;
                auto it = trackToVisitor.constFind(p.trackId);
                if (it != trackToVisitor.constEnd()) {
                    pid = it.value();
                } else {
                    cv::Mat feat = p.reidFeature;
                    if (feat.empty())
                        feat = reid.extract(frame, p.bbox);

                    float best = -1.f;
                    int   bestId = -1;
                    if (!feat.empty()) {
                        // 与该乘客的「所有锚点」比较取最大值：任一锚点像即判为同一人，
                        // 避免质心漂移导致的身份坍缩。
                        for (const auto &v : passengers) {
                            for (const auto &a : v.anchors) {
                                if (a.empty() || a.size() != feat.size()) continue;
                                const float s = reid.compare(a, feat);
                                if (s > best) { best = s; bestId = v.id; }
                            }
                        }
                    }
                    if (bestId >= 0 && best >= sameTh) {
                        pid = bestId;
                        for (auto &v : passengers) {
                            if (v.id != pid) continue;
                            if (v.anchors.size() < PassengerAgg::ANCHOR_MAX)
                                v.anchors.append(feat.clone());
                            break;
                        }
                    }
                    if (pid < 0) {               // 新乘客
                        PassengerAgg nv;
                        nv.id = nextPid++;
                        if (!feat.empty())
                            nv.anchors.append(feat.clone());
                        passengers.append(nv);
                        pid = nv.id;
                    }
                    trackToVisitor.insert(p.trackId, pid);

                    qInfo() << QString("  轨迹 #%1 → 乘客 P%2 (bestSim=%3)")
                                   .arg(p.trackId).arg(pid).arg(best, 0, 'f', 3);
                }

                // ---- 记录该乘客的出现时段 ----
                for (auto &v : passengers) {
                    if (v.id != pid) continue;
                    if (v.ranges.isEmpty() || (f - v.ranges.last().second) > gapFrames)
                        v.ranges.append(qMakePair(f, f));
                    else
                        v.ranges.last().second = f;

                    if (!v.hasY) { v.firstY = p.center.y; v.hasY = true; }
                    v.lastY = p.center.y;
                    break;
                }
            }
        }

        ++f;
        if (total > 0 && f % std::max(1, static_cast<int>(fps * 10)) == 0) {
            qInfo() << QString("  进度 %1/%2 (%3%)")
                           .arg(f).arg(total).arg(f * 100 / total);
        }
    }
    cap.release();

    qInfo() << QString("[视频] 处理完成：%1 帧，耗时 %2s，识别乘客 %3 位")
                   .arg(f).arg(timer.elapsed() / 1000.0, 0, 'f', 1).arg(passengers.size());

    // ===== 裁剪导出：按乘客分文件夹 =====
    for (const auto &v : passengers) {
        int clipIdx = 0;
        for (const auto &rg : v.ranges) {
            const int padFrames = static_cast<int>(padSec * fps);
            int s = std::max(0, rg.first - padFrames);
            int e = rg.second + padFrames;
            if (total > 0) e = std::min(e, total - 1);
            if (e - s < static_cast<int>(fps * 0.3))      // 过短片段忽略
                continue;

            const QString dir = outRoot + QString("/passenger_%1").arg(v.id);
            QDir().mkpath(dir);
            const QString outp = QString("%1/%2_clip%3_%4-%5.mp4")
                                     .arg(dir).arg(baseName).arg(clipIdx).arg(s).arg(e);

            cv::VideoCapture cap2(videoPath.toStdString());
            if (!cap2.isOpened()) {
                qWarning() << "[导出] 重新打开视频失败:" << videoPath;
                break;
            }
            cap2.set(cv::CAP_PROP_POS_FRAMES, s);
            cv::VideoWriter writer(outp.toStdString(),
                                   cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                                   fps, sz);
            if (!writer.isOpened()) {
                qWarning() << "[导出] 无法写入(编码不可用):" << outp;
                cap2.release();
                continue;
            }

            cv::Mat fr;
            int cur = s;
            while (cur <= e && cap2.read(fr)) {
                if (!fr.empty()) writer.write(fr);
                ++cur;
            }
            writer.release();
            cap2.release();

            const QString dirLabel = (v.lastY > v.firstY) ? "boarding_down" : "alighting_up";
            manifestRows << QString("%1,%2,%3,%4,%5,%6,%7,%8")
                                .arg(QFileInfo(videoPath).fileName())
                                .arg(v.id).arg(dirLabel).arg(outp)
                                .arg(s).arg(e)
                                .arg(s / fps, 0, 'f', 2).arg(e / fps, 0, 'f', 2);

            qInfo() << QString("  导出 P%1 [%2] 片段%3: %4 [帧%5-%6]")
                           .arg(v.id).arg(dirLabel).arg(clipIdx).arg(outp).arg(s).arg(e);
            ++clipIdx;
        }
    }
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);
    QCoreApplication::setApplicationName("passenger_clip_tool");
    QCoreApplication::setApplicationVersion("1.0");

    QCommandLineParser parser;
    parser.setApplicationDescription(
        "离线批处理：自动解压乘客上下车录像 → 检测/跟踪/ReID → 按乘客裁剪片段并分文件夹保存");
    parser.addHelpOption();
    parser.addVersionOption();

    QCommandLineOption optIn({"i", "input"}, "输入目录（含 .rar 分卷或已解压视频）", "dir");
    QCommandLineOption optOut({"o", "output"}, "输出目录", "dir", "./passenger_clips");
    QCommandLineOption optModel("model", "检测模型路径（.engine 或 .onnx）", "path",
                                "./model/yolo26n_trt10_x86.engine");
    QCommandLineOption optConf("conf", "检测置信度阈值", "f", "0.45");
    QCommandLineOption optStep("step", "检测抽帧步长（1=每帧）", "n", "1");
    QCommandLineOption optSame("same", "同一乘客判定阈值 [0,1]（越大越严格）", "f", "0.80");
    QCommandLineOption optDeep("deep", "深度 ReID 模型(.onnx)；留空则仅用深度=关闭", "path", "");
    QCommandLineOption optPad("pad", "每个片段前后留白（秒）", "f", "1.0");
    QCommandLineOption optGap("gap", "同一乘客两次出现间隔超过该秒数则分段", "f", "2.0");
    QCommandLineOption optNoExtract("no-extract", "跳过解压（输入目录已是解压后的视频）");

    parser.addOptions({optIn, optOut, optModel, optConf, optStep, optSame,
                       optDeep, optPad, optGap, optNoExtract});
    parser.process(app);

    const QString inDir  = parser.value(optIn);
    const QString outDir = parser.value(optOut);
    if (inDir.isEmpty()) {
        qCritical() << "缺少 --input 参数。使用 --help 查看用法。";
        return 2;
    }
    if (!QDir(inDir).exists()) {
        qCritical() << "输入目录不存在:" << inDir;
        return 2;
    }
    QDir().mkpath(outDir);

    // 1) 解压（可选）
    if (!parser.isSet(optNoExtract)) {
        extractArchives(inDir, inDir + "/_extracted");
    }

    // 2) 收集视频
    QStringList videos = listVideos(inDir);
    qInfo() << "[扫描] 发现视频文件" << videos.size() << "个";
    if (videos.isEmpty()) {
        qWarning() << "[扫描] 未发现可处理的视频。"
                      "若为分卷压缩，请确认 part01 等分卷完整；或用 --no-extract 指向已解压目录。";
    }

    // 3) 初始化检测 / ReID 后端
    PersonDetector detector;
    detector.setDnnUseCuda(true);
    detector.setDnnInputSize(640);
    detector.setLetterbox(true);
    const float conf = parser.value(optConf).toFloat();
    if (!detector.loadModel(parser.value(optModel), conf, 0.5f)) {
        qCritical() << "[检测] 模型加载失败:" << parser.value(optModel);
        return 3;
    }
    qInfo() << "[检测] 模型已加载:"
            << (detector.isTrtBackend() ? "TensorRT(engine)" : "OpenCV DNN(onnx)")
            << parser.value(optModel);

    DeepReID deep;
    bool useDeep = false;
    if (!parser.value(optDeep).isEmpty()) {
        deep.setUseCuda(true);
        if (deep.load(parser.value(optDeep))) {
            useDeep = true;
            qInfo() << "[ReID] 深度模型已加载，backend ="
                    << (deep.cudaActive() ? "CUDA(GPU)" : "CPU");
        } else {
            qWarning() << "[ReID] 深度模型加载失败，回退手工多模态特征";
        }
    }
    ReIDExtractor reid;
    reid.setComponentWeights(0.30f, 0.35f, 0.35f);
    reid.setDeepReid(useDeep ? &deep : nullptr, useDeep);

    const float sameTh  = parser.value(optSame).toFloat();
    const float padSec  = std::max(0.0f, parser.value(optPad).toFloat());
    const float gapSec  = std::max(0.1f, parser.value(optGap).toFloat());
    const int   step    = std::max(1, parser.value(optStep).toInt());

    qInfo() << QString("[参数] same=%1 pad=%2s gap=%3s step=%4")
                   .arg(sameTh, 0, 'f', 2).arg(padSec, 0, 'f', 1)
                   .arg(gapSec, 0, 'f', 1).arg(step);

    // 4) manifest 清单：先在内存累积所有行，最后一次性写盘。
    //    （此前用 QTextStream 流式写入，长时段运行后会出现后续视频丢行的问题。）
    QStringList manifestRows;
    manifestRows << "video,passenger_id,direction,clip_file,start_frame,end_frame,start_sec,end_sec";

    // 5) 逐视频处理
    QElapsedTimer all; all.start();
    QVector<PassengerAgg> passengers;   // 跨视频共享的乘客身份（同一乘客跨分段保持同一 ID）
    int nextPid = 1;
    for (const QString &v : videos) {
        processVideo(v, outDir, detector, reid, passengers, nextPid,
                     sameTh, padSec, gapSec, step, manifestRows);
    }

    QFile mf(outDir + "/manifest.csv");
    if (mf.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream ts(&mf);
        for (const QString &line : manifestRows) ts << line << "\n";
        mf.close();
        qInfo() << "[清单] 已写出" << (manifestRows.size() - 1) << "条片段记录 →"
                << (outDir + "/manifest.csv");
    } else {
        qWarning() << "[清单] 无法写入 manifest.csv";
    }

    qInfo() << QString("[完成] 共处理 %1 个视频，总耗时 %2s，输出目录: %3")
                   .arg(videos.size()).arg(all.elapsed() / 1000.0, 0, 'f', 1).arg(outDir);
    return 0;
}
