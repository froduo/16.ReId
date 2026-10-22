#ifndef APP_CONFIG_H
#define APP_CONFIG_H

#include <QString>
#include <QSettings>
#include <QCoreApplication>

struct AppConfig {
    // ===== 相机设置 =====
    QString cameraIP = "192.168.99.10";
    QString cameraUser = "admin";
    QString cameraPass = "admin123";
    int cameraPort = 8554;
    // 完整 RTSP URL（默认由 IP/端口/用户名/密码 自动派生，见 rebuildRtspUrl）
    QString rtspUrl = "rtsp://192.168.99.10:8554/live/camera";
    // RTSP URL 是否由用户「手动指定」：
    //   false（默认）—— URL 始终由 IP/端口/用户名/密码 派生，保证「改 IP 即生效」；
    //   true          —— 使用上面的自定义 rtspUrl，忽略 IP 等参数（高级用法）。
    bool cameraRtspManual = false;

    // 主界面标题
    QString windowTitle = "HX-CCD23 人员检测 ReID 系统";

    // ===== 下车方向持久化（下次启动默认加载上次选择）=====
    // "TopToBottom"：从上往下 ↓（下车）；"BottomToTop"：从下往上 ↑（下车）
    QString entryDirection = "TopToBottom";

    // ===== Web 服务（手机端查看上下车状态）=====
    bool webServerEnabled = true;
    int  webServerPort = 8080;

    // ===== 图像保存参数 =====
    QString savePath = "./save_images";
    QString saveFormat = "jpg";
    int jpegQuality = 95;

    // ===== 推理参数 =====
    // 模型路径（优先 TensorRT 引擎，回退到 ONNX）
    QString modelPath = "./model/yolo26n_trt10_x86.engine";
    // 是否启用推理
    bool enableInference = true;
    // 检测置信度阈值。
    // 【俯视场景召回】工厂 yolo26n 在本俯视场景对真实乘客的分数偏低
    // （实测同一乘客 0.35~0.78），旧阈值 0.45 会漏检 → 下调至 0.30 提升召回；
    // 误检由「尺寸 / 宽高比 / 静态抑制」三道过滤兜底；部署专用乘客模型后可再上调。
    float detectorConfThreshold = 0.30f;
    // NMS 阈值
    float detectorNmsThreshold = 0.5f;
    // ===== 小目标过滤（以整幅图像尺寸为参考，默认针对 1920x1080）=====
    // 是否启用按尺寸过滤误识别的小物体
    bool detectorSizeFilterEnabled = true;
    // 框高 / 图像高 的最小比例（0.08 ≈ 1080p 下 86px）
    float detectorMinPersonHeightRatio = 0.08f;
    // 框面积 / 图像面积 的最小比例（0.005 ≈ 1080p 下 10368px²）
    float detectorMinPersonAreaRatio = 0.005f;
    // 绝对像素最小框（叠加在比例过滤之上；专治区域内小框误检）
    int   detectorMinBoxWidthPx = 30;
    int   detectorMinBoxHeightPx = 90;
    // 宽高比(宽/高)合理区间：过细(竖条)/过扁(横条)均视为误检。
    // 【俯视场景】相机位于车门正上方，人形被压缩、检测框近方形
    // （实测 person_43=305×307、本机 333×325，w/h≈0.99~1.03）；旧上限 1.00
    // 会把 w/h 略大于 1 的真实乘客误杀 → 放宽为 [0.35, 3.00]（与 orbbec 工程一致）。
    float detectorMinBoxAspect = 0.35f;
    float detectorMaxBoxAspect = 3.00f;

    // ===== 最大尺寸上限（比例；1.0=不限制）：抑制超大静态误检框 =====
    // 空场景中固定物体被误检时，框通常远大于真实目标；head 头部框很小，
    // 超过上限可直接判为非目标（比单纯抬高置信度更可靠，实测误检置信度可达 0.81）。
    float detectorHeadMaxBoxWidthRatio  = 0.40f;   // 头框宽/图像宽 上限
    float detectorHeadMaxBoxHeightRatio = 0.45f;   // 头框高/图像高 上限
    float detectorHeadMaxBoxAreaRatio   = 0.12f;   // 头框面积/图像面积 上限（近景大头留足余量）
    float detectorMaxBoxWidthRatio      = 1.00f;   // person 模式默认不限制
    float detectorMaxBoxHeightRatio     = 1.00f;
    float detectorMaxBoxAreaRatio       = 1.00f;

    // ===== 静态误检抑制（固定摄像头：同位置、近零位移的持续检测判为背景误检）=====
    bool  detectorStaticSuppressEnabled = true;
    int   detectorStaticConfirmFrames   = 150;    // 连续命中多少帧后拉黑（约 3s@50fps）
    float detectorStaticMaxMoveRatio    = 0.008f; // 中心相对起点位移上限（相对图像长边）
    float detectorStaticIouThreshold    = 0.70f;  // 与已有静态区域的 IoU 匹配阈值
    int   detectorStaticMissTolerance   = 30;     // 连续未命中多少帧后移除该区域

    // ===== 检测目标类型（适配新训练的 head 头部检测模型）=====
    // "person"：人体检测（旧 yolo26n）；"head"：头部检测（YOLO11n head，单类 1×5×8400）。
    // head 框显著小于人体框且更接近方形，复用人体过滤参数会被全部误杀，
    // 因此 head 模式使用独立的一套尺寸/长宽比过滤参数。
    QString detectorModelKind = "person";
    // TensorRT 预处理：true=letterbox（保持长宽比+灰边，旧 yolo26n 用）；
    // false=等比拉伸(squash，head 模型推荐，实测置信度 0.72 vs 0.23)。
    bool detectorLetterbox = true;
    // head 模式专用置信度：比人体模式更高，用于抑制“空场景误检成乘客”。
    float detectorHeadConfThreshold = 0.55f;
    // head 过滤参数（较人体更严：头部框小，过松会把背景斑块当成人头）
    float detectorHeadMinHeightRatio = 0.025f;  // 头框高/图像高（1080p ≈27px）
    float detectorHeadMinAreaRatio   = 0.0012f; // 头框面积/图像面积
    int   detectorHeadMinBoxWidthPx  = 22;
    int   detectorHeadMinBoxHeightPx = 22;
    float detectorHeadMinBoxAspect   = 0.50f;   // 头框近似方形
    float detectorHeadMaxBoxAspect   = 2.00f;

    // ===== 检测模型选择（优先轻量 ONNX 小模型，提升检测帧率）=====
    // 注意：实测通用 COCO 模型(如 yolov8n)在本俯视鱼眼场景下 person 置信度极低
    //      （max≈0.02~0.06），召回远不如场景专用模型，故默认关闭；
    //      仅在用本场景数据微调过 ONNX 小模型后再开启。
    bool detectorPreferOnnx = false;                       // 优先使用 ONNX 小模型（默认关闭）
    QString detectorOnnxPath = "./model/yolov8n.onnx";     // ONNX 模型路径
    bool detectorDnnUseCuda = true;                        // ONNX 推理优先 CUDA(GPU)，不可用自动回退 CPU
    int  detectorDnnInputSize = 640;                       // ONNX 输入尺寸（须与导出模型一致）
    QString detectorModelUrl = "";                         // 可选：ONNX 直链（配置后自动下载到 detectorOnnxPath）

    // ===== 跟踪参数 =====
    // IoU 关联阈值：0.15 与离线判定保持一致（offlinedetectform 的视频判定同样用 0.15）。
    // 旧值 0.3 偏严：同一人短暂丢失后的新检测不易复联原轨迹，会分裂出新轨迹，
    // 实时画面因此常出现“同一人两个框”。下调后显著减少轨迹分裂与 ID 切换。
    float trackerIouThreshold = 0.15f;
    // 丢失多少帧后丢弃：25fps 下 75 帧≈3s（原 30 帧仅 1.2s，短暂遮挡即丢 ID）
    int trackerMaxMissed = 75;
    // 卡曼滤波 + ReID 去重参数
    bool trackerKalmanEnabled = true;       // 启用卡尔曼预测匹配
    bool trackerReidEnabled = true;         // 启用ReID辅助匹配
    // 跟踪器内部ReID匹配阈值：与离线判定对齐（offlinedetectform 视频判定用 0.62）。
    // 旧值 0.72 偏严，同一人短暂丢失后难以复联原轨迹，易造成轨迹分裂与 ID 切换。
    float trackerReidThreshold = 0.62f;     // 跟踪器内部ReID匹配阈值
    int trackerGalleryMaxAgeMs = 10000;     // 丢失轨迹画廊最大保留时间(ms)

    // ===== 显示过滤（消除“同一人两个框”的卡尔曼幽灵框）=====
    // 背景：跟踪器每帧会把「本帧未匹配到真实检测」的轨迹用卡尔曼预测外推，
    //       并在 trackerMaxMissed 帧内持续保留（约 3s）。旧绘制逻辑无差别地把这些
    //       预测框也画出来，导致实时画面比同一段视频的离线判定多出一个框
    //       （实时日志可见 dets=1, persons=2）。
    // 策略：默认只绘制「本帧真实检测到」的轨迹，彻底消除幽灵框；如需保留短暂
    //       丢失期间的框，可将 displayOnlyDetectedTracks 设为 false，并用
    //       displayMaxMissedFrames 控制最多容忍几帧丢失。
    bool displayOnlyDetectedTracks = true;   // true=仅绘制真实检测轨迹（推荐）
    int  displayMinConfirmFrames   = 3;      // 轨迹累计命中达到该帧数才绘制（抑制偶发闪现）
    int  displayMaxMissedFrames    = 8;      // displayOnlyDetectedTracks=false 时，丢失超过该帧数停止绘制

    // ===== ReID 参数 =====
    // 说明：颜色直方图分箱已由 30x32x32 降为 15x8x8（见 reidextractor.h），
    // 描述子稳定性提升后，同一人上/下车实测相似度约 0.67~0.85，
    // 故将匹配阈值由 0.70 下调为 0.65（离线实测校准，见 offline_reid_*)。
    float reidMatchThreshold = 0.65f;
    // 多特征融合权重（颜色 / 纹理 / 身形），内部自动归一化
    // 颜色在鱼眼俯视跨朝向场景下最不稳定，降低其权重。
    float reidWeightColor = 0.30f;
    float reidWeightTexture = 0.40f;
    float reidWeightShape = 0.30f;
    // 是否启用背景扣除（固定摄像头：剔除背景像素，提高 ReID 准确率）
    bool reidUseBackground = true;
    // 是否自动保存「上/下车整段视频」（mp4，进入检测区域→离开检测区域）
    bool reidSaveVideo = true;

    // ===== 人物片段录制（画面出现人即录制，无人超时后分段保存）=====
    bool personClipEnabled = true;                    // 是否启用
    QString personClipPath = "./save_images/clips";   // 保存目录
    int personClipGapMs = 3000;                       // 连续无人超过该时长即结束本段
    int personClipFps = 25;                           // 起始帧率（运行中按实际写入速率自校准）

    // 通道方向判定：有效穿越所需的最小中心点纵向位移（归一化，0~1）
    float zoneMinDisplacement = 0.04f;

    // ===== 深度 ReID（OSNet/ONNX，基于 opencv_dnn）=====
    // 深度外观嵌入对相似着装/跨朝向的区分力显著优于手工颜色纹理特征
    // 默认关闭：预训练 OSNet 在本俯视鱼眼场景下区分力尚未验证，
    // 需现场采集数据微调并标定阈值后再开启，避免误判回归。
    bool reidUseDeepReid = false;                                    // 是否启用深度 ReID
    // 深度 ReID 推理后端：true=优先 CUDA(GPU)（把识别推理从 CPU 卸载到 GPU，
    // 降低 CPU 占用、提升实时性；不可用时自动回退 CPU）；false=强制 CPU
    bool reidDeepUseCuda = true;
    QString reidDeepModelPath = "./model/osnet_x1_0_msmt17.onnx";    // ONNX 模型路径
    float reidDeepMatchThreshold = 0.50f;                            // 深度特征余弦判定阈值
    // ===== 上/下车文件夹全图比对（准确率优先，后台线程执行）=====
    // 下车目录与该目录时间之前、且未标记的上车目录做全图深度 ReID 比对。
    float reidFolderMatchThreshold = 0.60f;      // 目录级相似度判定阈值
    // 目录级匹配的最小置信度差（最高分 - 次高分）。低于此值说明候选间区分度不足，
    // 不判定匹配（避免相似乘客误匹配；宁可漏配、留待后续/人工复核）。
    float reidFolderMatchMinMargin = 0.04f;
    int   personSaveEveryN = 3;                  // 乘客出现期间每 N 帧保存 1 张裁剪原图
    int   personMinTrackFrames = 3;              // 新乘客建档所需的最少真实检测帧数（预热）
    int   personFolderMaxImages = 200;           // 单目录参与比对的最大图片数
    int   personFolderQualityKeep = 15;          // 质量筛选：按清晰度保留的最多帧数（过滤模糊帧）
    // head 检测模型下：头框 → 人体框 的 ReID 裁剪扩展系数（1.0 表示不扩展）。
    // 深度 ReID 训练数据来自人体裁剪（visitor_V*），适度扩展可让头框裁剪更接近训练分布。
    float reidHeadBoxExpand = 1.35f;
    // ReID 图像保存路径
    QString reidSavePath = "./save_images/reid";
    // 区域保存路径
    QString regionSavePath = "./regions.json";
    // 访客记录保存路径
    QString visitorLogPath = "./visitor_log";

    // ===== 乘车数据存储（每站上下车人数 + 车厢内人数）=====
    // 后端："sqlite"（默认，零依赖、随程序落盘）或 "postgresql"（大规模车队集中存储）
    QString tripDbBackend = "sqlite";
    // SQLite 数据库文件路径（相对路径按程序目录解析）
    QString tripDbPath = "./trip_data/bus_trips.db";
    // PostgreSQL 连接参数（tripDbBackend == "postgresql" 时生效）
    QString tripDbHost = "127.0.0.1";
    int     tripDbPort = 5432;
    QString tripDbName = "bus";
    QString tripDbUser = "postgres";
    QString tripDbPassword = "";
    // 当前车辆标识 / 当前站点（站点可在运行期切换）
    QString vehicleId = "BUS-01";
    QString currentStationId = "STATION-01";
    QString currentStationName = "1号站";

    // ===== 程序更新（检查更新）=====
    // 更新清单(manifest)地址：http(s)://host/path/version.json
    // 留空时仅支持程序目录下的本地 update.json（离线/U盘升级）；
    // 两者都不可用时，「检查更新」会明确提示当前无法在线更新。
    QString updateUrl = "";

    void save() {
        QString path = QCoreApplication::applicationDirPath() + "/config.ini";
        QSettings s(path, QSettings::IniFormat);

        // 相机设置
        s.beginGroup("Camera");
        s.setValue("IP", cameraIP);
        s.setValue("User", cameraUser);
        s.setValue("Pass", cameraPass);
        s.setValue("Port", cameraPort);
        s.setValue("RTSP_URL", rtspUrl);
        s.setValue("RTSP_Manual", cameraRtspManual);
        s.setValue("EntryDirection", entryDirection);
        s.endGroup();

        // 显示设置
        s.beginGroup("Display");
        s.setValue("WindowTitle", windowTitle);
        s.setValue("OnlyDetectedTracks", displayOnlyDetectedTracks);
        s.setValue("MinConfirmFrames", displayMinConfirmFrames);
        s.setValue("MaxMissedFrames", displayMaxMissedFrames);
        s.endGroup();

        // 图像保存
        s.beginGroup("ImageSave");
        s.setValue("SavePath", savePath);
        s.setValue("SaveFormat", saveFormat);
        s.setValue("JpegQuality", jpegQuality);
        s.endGroup();

        // Web 服务
        s.beginGroup("Web");
        s.setValue("Enabled", webServerEnabled);
        s.setValue("Port", webServerPort);
        s.endGroup();

        // 推理参数
        s.beginGroup("Inference");
        s.setValue("EnableInference", enableInference);
        s.setValue("ModelPath", modelPath);
        s.setValue("ConfThreshold", static_cast<double>(detectorConfThreshold));
        s.setValue("NmsThreshold", static_cast<double>(detectorNmsThreshold));
        s.setValue("SizeFilterEnabled", detectorSizeFilterEnabled);
        s.setValue("MinPersonHeightRatio", static_cast<double>(detectorMinPersonHeightRatio));
        s.setValue("MinPersonAreaRatio", static_cast<double>(detectorMinPersonAreaRatio));
        s.setValue("MinBoxWidthPx", detectorMinBoxWidthPx);
        s.setValue("MinBoxHeightPx", detectorMinBoxHeightPx);
        s.setValue("MinBoxAspect", static_cast<double>(detectorMinBoxAspect));
        s.setValue("MaxBoxAspect", static_cast<double>(detectorMaxBoxAspect));
        s.setValue("HeadMaxBoxWidthRatio", static_cast<double>(detectorHeadMaxBoxWidthRatio));
        s.setValue("HeadMaxBoxHeightRatio", static_cast<double>(detectorHeadMaxBoxHeightRatio));
        s.setValue("HeadMaxBoxAreaRatio", static_cast<double>(detectorHeadMaxBoxAreaRatio));
        s.setValue("MaxBoxWidthRatio", static_cast<double>(detectorMaxBoxWidthRatio));
        s.setValue("MaxBoxHeightRatio", static_cast<double>(detectorMaxBoxHeightRatio));
        s.setValue("MaxBoxAreaRatio", static_cast<double>(detectorMaxBoxAreaRatio));
        s.setValue("StaticSuppressEnabled", detectorStaticSuppressEnabled);
        s.setValue("StaticConfirmFrames", detectorStaticConfirmFrames);
        s.setValue("StaticMaxMoveRatio", static_cast<double>(detectorStaticMaxMoveRatio));
        s.setValue("StaticIouThreshold", static_cast<double>(detectorStaticIouThreshold));
        s.setValue("StaticMissTolerance", detectorStaticMissTolerance);
        s.setValue("ModelKind", detectorModelKind);
        s.setValue("Letterbox", detectorLetterbox);
        s.setValue("HeadConfThreshold", static_cast<double>(detectorHeadConfThreshold));
        s.setValue("HeadMinPersonHeightRatio", static_cast<double>(detectorHeadMinHeightRatio));
        s.setValue("HeadMinPersonAreaRatio", static_cast<double>(detectorHeadMinAreaRatio));
        s.setValue("HeadMinBoxWidthPx", detectorHeadMinBoxWidthPx);
        s.setValue("HeadMinBoxHeightPx", detectorHeadMinBoxHeightPx);
        s.setValue("HeadMinBoxAspect", static_cast<double>(detectorHeadMinBoxAspect));
        s.setValue("HeadMaxBoxAspect", static_cast<double>(detectorHeadMaxBoxAspect));
        s.setValue("PreferOnnx", detectorPreferOnnx);
        s.setValue("OnnxPath", detectorOnnxPath);
        s.setValue("DnnUseCuda", detectorDnnUseCuda);
        s.setValue("DnnInputSize", detectorDnnInputSize);
        s.setValue("ModelUrl", detectorModelUrl);
        s.endGroup();

        // 跟踪参数
        s.beginGroup("Tracker");
        s.setValue("IouThreshold", static_cast<double>(trackerIouThreshold));
        s.setValue("MaxMissed", trackerMaxMissed);
        s.setValue("KalmanEnabled", trackerKalmanEnabled);
        s.setValue("ReidEnabled", trackerReidEnabled);
        s.setValue("ReidThreshold", static_cast<double>(trackerReidThreshold));
        s.setValue("GalleryMaxAgeMs", trackerGalleryMaxAgeMs);
        s.endGroup();

        // ReID 参数
        s.beginGroup("ReID");
        s.setValue("MatchThreshold", static_cast<double>(reidMatchThreshold));
        s.setValue("WeightColor", static_cast<double>(reidWeightColor));
        s.setValue("WeightTexture", static_cast<double>(reidWeightTexture));
        s.setValue("WeightShape", static_cast<double>(reidWeightShape));
        s.setValue("UseBackground", reidUseBackground);
        s.setValue("SaveVideo", reidSaveVideo);
        s.setValue("ZoneMinDisplacement", static_cast<double>(zoneMinDisplacement));
        s.setValue("UseDeepReid", reidUseDeepReid);
        s.setValue("DeepUseCuda", reidDeepUseCuda);
        s.setValue("DeepModelPath", reidDeepModelPath);
        s.setValue("DeepMatchThreshold", static_cast<double>(reidDeepMatchThreshold));
        s.setValue("FolderMatchThreshold", static_cast<double>(reidFolderMatchThreshold));
        s.setValue("PersonSaveEveryN", personSaveEveryN);
        s.setValue("PersonMinTrackFrames", personMinTrackFrames);
        s.setValue("PersonFolderMaxImages", personFolderMaxImages);
        s.setValue("PersonFolderQualityKeep", personFolderQualityKeep);
        s.setValue("HeadBoxExpand", static_cast<double>(reidHeadBoxExpand));
        s.setValue("SavePath", reidSavePath);
        s.setValue("RegionSavePath", regionSavePath);
        s.setValue("VisitorLogPath", visitorLogPath);
        s.endGroup();

        // 人物片段录制
        s.beginGroup("PersonClip");
        s.setValue("Enabled", personClipEnabled);
        s.setValue("Path", personClipPath);
        s.setValue("GapMs", personClipGapMs);
        s.setValue("Fps", personClipFps);
        s.endGroup();

        // 乘车数据存储
        s.beginGroup("TripDb");
        s.setValue("Backend", tripDbBackend);
        s.setValue("Path", tripDbPath);
        s.setValue("Host", tripDbHost);
        s.setValue("Port", tripDbPort);
        s.setValue("Name", tripDbName);
        s.setValue("User", tripDbUser);
        s.setValue("Password", tripDbPassword);
        s.setValue("VehicleId", vehicleId);
        s.setValue("StationId", currentStationId);
        s.setValue("StationName", currentStationName);
        s.endGroup();

        // 更新参数
        s.beginGroup("Update");
        s.setValue("UpdateUrl", updateUrl);
        s.endGroup();

        s.sync();
    }

    void load() {
        QSettings s(QCoreApplication::applicationDirPath() + "/config.ini", QSettings::IniFormat);

        // 相机设置
        cameraIP = s.value("Camera/IP", "192.168.99.10").toString();
        cameraUser = s.value("Camera/User", "admin").toString();
        cameraPass = s.value("Camera/Pass", "admin123").toString();
        cameraPort = s.value("Camera/Port", 8554).toInt();
        // 是否手动指定 RTSP URL（缺省 false → 以相机 IP/端口为唯一事实来源）
        cameraRtspManual = s.value("Camera/RTSP_Manual", false).toBool();
        rtspUrl = s.value("Camera/RTSP_URL", QString()).toString();
        // 关键修复：默认始终按「相机 IP + 端口 + 用户名 + 密码」重建 RTSP URL，
        // 避免历史 config.ini 中残留的旧 RTSP_URL 覆盖用户新配置的相机 IP，
        // 导致「改了 IP 但启动相机仍连旧地址」。仅当用户显式开启手动指定时才沿用自定义 URL。
        if (!cameraRtspManual || rtspUrl.isEmpty()) {
            rebuildRtspUrl();
            cameraRtspManual = false;
        }
        entryDirection = s.value("Camera/EntryDirection", "TopToBottom").toString();
        if (entryDirection != QLatin1String("BottomToTop"))
            entryDirection = QLatin1String("TopToBottom");

        // 显示设置
        windowTitle = s.value("Display/WindowTitle", "HX-CCD23 人员检测 ReID 系统").toString();
        // 显示过滤：默认只绘制真实检测轨迹，消除卡尔曼外推的“幽灵框”
        displayOnlyDetectedTracks = s.value("Display/OnlyDetectedTracks", true).toBool();
        displayMinConfirmFrames = s.value("Display/MinConfirmFrames", 3).toInt();
        displayMaxMissedFrames = s.value("Display/MaxMissedFrames", 8).toInt();

        // 图像保存
        savePath = s.value("ImageSave/SavePath", "./save_images").toString();
        saveFormat = s.value("ImageSave/SaveFormat", "jpg").toString();
        jpegQuality = s.value("ImageSave/JpegQuality", 95).toInt();

        // 推理参数
        enableInference = s.value("Inference/EnableInference", true).toBool();
        modelPath = s.value("Inference/ModelPath", "./model/passenger_yolov8m_trt10_x86_20261022_093436.engine").toString();
        detectorConfThreshold = static_cast<float>(s.value("Inference/ConfThreshold", 0.30).toDouble());
        detectorNmsThreshold = static_cast<float>(s.value("Inference/NmsThreshold", 0.5).toDouble());
        detectorSizeFilterEnabled = s.value("Inference/SizeFilterEnabled", true).toBool();
        detectorMinPersonHeightRatio = static_cast<float>(s.value("Inference/MinPersonHeightRatio", 0.08).toDouble());
        detectorMinPersonAreaRatio = static_cast<float>(s.value("Inference/MinPersonAreaRatio", 0.005).toDouble());
        detectorMinBoxWidthPx = s.value("Inference/MinBoxWidthPx", 30).toInt();
        detectorMinBoxHeightPx = s.value("Inference/MinBoxHeightPx", 90).toInt();
        detectorMinBoxAspect = static_cast<float>(s.value("Inference/MinBoxAspect", 0.35).toDouble());
        detectorMaxBoxAspect = static_cast<float>(s.value("Inference/MaxBoxAspect", 3.00).toDouble());
        detectorHeadMaxBoxWidthRatio = static_cast<float>(s.value("Inference/HeadMaxBoxWidthRatio", 0.40).toDouble());
        detectorHeadMaxBoxHeightRatio = static_cast<float>(s.value("Inference/HeadMaxBoxHeightRatio", 0.45).toDouble());
        detectorHeadMaxBoxAreaRatio = static_cast<float>(s.value("Inference/HeadMaxBoxAreaRatio", 0.12).toDouble());
        detectorMaxBoxWidthRatio = static_cast<float>(s.value("Inference/MaxBoxWidthRatio", 1.00).toDouble());
        detectorMaxBoxHeightRatio = static_cast<float>(s.value("Inference/MaxBoxHeightRatio", 1.00).toDouble());
        detectorMaxBoxAreaRatio = static_cast<float>(s.value("Inference/MaxBoxAreaRatio", 1.00).toDouble());
        detectorStaticSuppressEnabled = s.value("Inference/StaticSuppressEnabled", true).toBool();
        detectorStaticConfirmFrames = s.value("Inference/StaticConfirmFrames", 150).toInt();
        detectorStaticMaxMoveRatio = static_cast<float>(s.value("Inference/StaticMaxMoveRatio", 0.008).toDouble());
        detectorStaticIouThreshold = static_cast<float>(s.value("Inference/StaticIouThreshold", 0.70).toDouble());
        detectorStaticMissTolerance = s.value("Inference/StaticMissTolerance", 30).toInt();
        detectorModelKind = s.value("Inference/ModelKind", "person").toString();
        // 缺省随模型类型联动：head 模型默认 squash（letterbox 会使头部分数低于阈值）
        const bool lbDefault = (detectorModelKind.compare("head", Qt::CaseInsensitive) != 0);
        detectorLetterbox = s.value("Inference/Letterbox", lbDefault).toBool();
        detectorHeadConfThreshold = static_cast<float>(
            s.value("Inference/HeadConfThreshold", 0.55).toDouble());
        detectorHeadMinHeightRatio = static_cast<float>(s.value("Inference/HeadMinPersonHeightRatio", 0.025).toDouble());
        detectorHeadMinAreaRatio = static_cast<float>(s.value("Inference/HeadMinPersonAreaRatio", 0.0012).toDouble());
        detectorHeadMinBoxWidthPx = s.value("Inference/HeadMinBoxWidthPx", 22).toInt();
        detectorHeadMinBoxHeightPx = s.value("Inference/HeadMinBoxHeightPx", 22).toInt();
        detectorHeadMinBoxAspect = static_cast<float>(s.value("Inference/HeadMinBoxAspect", 0.50).toDouble());
        detectorHeadMaxBoxAspect = static_cast<float>(s.value("Inference/HeadMaxBoxAspect", 2.00).toDouble());
        detectorPreferOnnx = s.value("Inference/PreferOnnx", false).toBool();
        detectorOnnxPath = s.value("Inference/OnnxPath", "./model/passenger_yolov8m_trt10_x86_20261022_093436.onnx").toString();
        detectorDnnUseCuda = s.value("Inference/DnnUseCuda", true).toBool();
        detectorDnnInputSize = s.value("Inference/DnnInputSize", 640).toInt();
        detectorModelUrl = s.value("Inference/ModelUrl", "").toString();

        // 跟踪参数
        trackerIouThreshold = static_cast<float>(s.value("Tracker/IouThreshold", 0.15).toDouble());
        trackerMaxMissed = s.value("Tracker/MaxMissed", 75).toInt();
        trackerKalmanEnabled = s.value("Tracker/KalmanEnabled", true).toBool();
        trackerReidEnabled = s.value("Tracker/ReidEnabled", true).toBool();
        trackerReidThreshold = static_cast<float>(s.value("Tracker/ReidThreshold", 0.62).toDouble());
        trackerGalleryMaxAgeMs = s.value("Tracker/GalleryMaxAgeMs", 10000).toInt();

        // ReID 参数
        reidMatchThreshold = static_cast<float>(s.value("ReID/MatchThreshold", 0.65).toDouble());
        reidWeightColor = static_cast<float>(s.value("ReID/WeightColor", 0.30).toDouble());
        reidWeightTexture = static_cast<float>(s.value("ReID/WeightTexture", 0.40).toDouble());
        reidWeightShape = static_cast<float>(s.value("ReID/WeightShape", 0.30).toDouble());
        reidUseBackground = s.value("ReID/UseBackground", true).toBool();
        reidSaveVideo = s.value("ReID/SaveVideo", true).toBool();
        zoneMinDisplacement = static_cast<float>(s.value("ReID/ZoneMinDisplacement", 0.04).toDouble());
        reidUseDeepReid = s.value("ReID/UseDeepReid", false).toBool();
        reidDeepUseCuda = s.value("ReID/DeepUseCuda", true).toBool();
        reidDeepModelPath = s.value("ReID/DeepModelPath", "./model/osnet_x1_0_msmt17.onnx").toString();
        reidDeepMatchThreshold = static_cast<float>(s.value("ReID/DeepMatchThreshold", 0.50).toDouble());
        reidFolderMatchThreshold = static_cast<float>(s.value("ReID/FolderMatchThreshold", 0.60).toDouble());
        personSaveEveryN = s.value("ReID/PersonSaveEveryN", 3).toInt();
        personMinTrackFrames = s.value("ReID/PersonMinTrackFrames", 3).toInt();
        personFolderMaxImages = s.value("ReID/PersonFolderMaxImages", 200).toInt();
        personFolderQualityKeep = s.value("ReID/PersonFolderQualityKeep", 15).toInt();
        reidHeadBoxExpand = static_cast<float>(s.value("ReID/HeadBoxExpand", 1.35).toDouble());
        reidSavePath = s.value("ReID/SavePath", "./save_images/reid").toString();
        regionSavePath = s.value("ReID/RegionSavePath", "./regions.json").toString();
        visitorLogPath = s.value("ReID/VisitorLogPath", "./visitor_log").toString();

        // 人物片段录制
        personClipEnabled = s.value("PersonClip/Enabled", true).toBool();
        personClipPath = s.value("PersonClip/Path", "./save_images/clips").toString();
        personClipGapMs = s.value("PersonClip/GapMs", 3000).toInt();
        personClipFps = s.value("PersonClip/Fps", 25).toInt();

        // 乘车数据存储
        tripDbBackend = s.value("TripDb/Backend", "sqlite").toString();
        tripDbPath = s.value("TripDb/Path", "./trip_data/bus_trips.db").toString();
        tripDbHost = s.value("TripDb/Host", "127.0.0.1").toString();
        tripDbPort = s.value("TripDb/Port", 5432).toInt();
        tripDbName = s.value("TripDb/Name", "bus").toString();
        tripDbUser = s.value("TripDb/User", "postgres").toString();
        tripDbPassword = s.value("TripDb/Password", "").toString();
        vehicleId = s.value("TripDb/VehicleId", "BUS-01").toString();
        currentStationId = s.value("TripDb/StationId", "STATION-01").toString();
        currentStationName = s.value("TripDb/StationName", "1号站").toString();

        // Web 服务
        webServerEnabled = s.value("Web/Enabled", true).toBool();
        webServerPort = s.value("Web/Port", 8080).toInt();

        // 更新参数
        updateUrl = s.value("Update/UpdateUrl").toString();
    }

    /// 根据 IP/User/Pass/Port 生成 RTSP URL（不修改成员变量）
    QString derivedRtspUrl() const {
        return QString("rtsp://%1:%2@%3:%4/live/camera")
            .arg(cameraUser, cameraPass, cameraIP).arg(cameraPort);
    }

    /// 根据 IP/User/Pass/Port 重建 RTSP URL
    void rebuildRtspUrl() { rtspUrl = derivedRtspUrl(); }
};

#endif // APP_CONFIG_H
