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
    // 完整 RTSP URL（可手动覆盖）
    QString rtspUrl = "rtsp://192.168.99.10:8554/live/camera";

    // 主界面标题
    QString windowTitle = "HX-CCD23 人员检测 ReID 系统";

    // ===== 图像保存参数 =====
    QString savePath = "./save_images";
    QString saveFormat = "jpg";
    int jpegQuality = 95;

    // ===== 推理参数 =====
    // 模型路径（优先 TensorRT 引擎，回退到 ONNX）
    QString modelPath = "./model/yolo26n_trt10_x86.engine";
    // 是否启用推理
    bool enableInference = true;
    // 检测置信度阈值
    float detectorConfThreshold = 0.45f;
    // NMS 阈值
    float detectorNmsThreshold = 0.5f;

    // ===== 跟踪参数 =====
    float trackerIouThreshold = 0.3f;
    int trackerMaxMissed = 30;
    // 卡曼滤波 + ReID 去重参数
    bool trackerKalmanEnabled = true;       // 启用卡尔曼预测匹配
    bool trackerReidEnabled = true;         // 启用ReID辅助匹配
    float trackerReidThreshold = 0.65f;     // 跟踪器内部ReID匹配阈值
    int trackerGalleryMaxAgeMs = 10000;     // 丢失轨迹画廊最大保留时间(ms)
    // 重复轨迹合并（同一人只保留一个 ID）
    float trackerMergeSimilarity = 0.85f;   // 判定「同一人」的外观相似度阈值
    int trackerMergeMaxYoungAge = 20;       // 年轻轨迹允许合并的最大存活帧数

    // ===== ReID 参数 =====
    float reidMatchThreshold = 0.70f;
    // ReID 图像保存路径
    QString reidSavePath = "./save_images/reid";
    // 区域保存路径
    QString regionSavePath = "./regions.json";
    // 访客记录保存路径
    QString visitorLogPath = "./visitor_log";

    // ===== 动恰客流时段聚合上行（footfall.dongqia.com）=====
    bool uplinkEnabled = true;
    QString uplinkBaseUrl = "https://footfall.dongqia.com";
    QString uplinkIntervalPath = "/api/passenger-flow/interval-aggregate";
    QString uplinkPersonPath = "/api/passenger-flow/person-event";
    QString uplinkUser = "dongqia01";
    QString uplinkPass = "123456";
    QString uplinkMasterSn = "CAM-A01";
    QString uplinkDeviceSns;              // 逗号分隔；空则取 masterSn
    QString uplinkSpoolDir = "./uplink_spool";
    int uplinkRetryIntervalMs = 5000;     // 失败重试固定间隔
    int uplinkDayCutHour = 0;             // 日切时间（营业日划分）
    QString uplinkStatBasis = "enter";    // 统计基准方向：enter / leave
    QString uplinkTzName = "Asia/Shanghai";
    int uplinkUtcOffsetSeconds = 28800;

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
        s.endGroup();

        // 显示设置
        s.beginGroup("Display");
        s.setValue("WindowTitle", windowTitle);
        s.endGroup();

        // 图像保存
        s.beginGroup("ImageSave");
        s.setValue("SavePath", savePath);
        s.setValue("SaveFormat", saveFormat);
        s.setValue("JpegQuality", jpegQuality);
        s.endGroup();

        // 推理参数
        s.beginGroup("Inference");
        s.setValue("EnableInference", enableInference);
        s.setValue("ModelPath", modelPath);
        s.setValue("ConfThreshold", static_cast<double>(detectorConfThreshold));
        s.setValue("NmsThreshold", static_cast<double>(detectorNmsThreshold));
        s.endGroup();

        // 跟踪参数
        s.beginGroup("Tracker");
        s.setValue("IouThreshold", static_cast<double>(trackerIouThreshold));
        s.setValue("MaxMissed", trackerMaxMissed);
        s.setValue("KalmanEnabled", trackerKalmanEnabled);
        s.setValue("ReidEnabled", trackerReidEnabled);
        s.setValue("ReidThreshold", static_cast<double>(trackerReidThreshold));
        s.setValue("GalleryMaxAgeMs", trackerGalleryMaxAgeMs);
        s.setValue("MergeSimilarity", static_cast<double>(trackerMergeSimilarity));
        s.setValue("MergeMaxYoungAge", trackerMergeMaxYoungAge);
        s.endGroup();

        // ReID 参数
        s.beginGroup("ReID");
        s.setValue("MatchThreshold", static_cast<double>(reidMatchThreshold));
        s.setValue("SavePath", reidSavePath);
        s.setValue("RegionSavePath", regionSavePath);
        s.setValue("VisitorLogPath", visitorLogPath);
        s.endGroup();

        // 动恰客流上行
        s.beginGroup("Uplink");
        s.setValue("Enabled", uplinkEnabled);
        s.setValue("BaseUrl", uplinkBaseUrl);
        s.setValue("IntervalPath", uplinkIntervalPath);
        s.setValue("PersonPath", uplinkPersonPath);
        s.setValue("User", uplinkUser);
        s.setValue("Pass", uplinkPass);
        s.setValue("MasterSn", uplinkMasterSn);
        s.setValue("DeviceSns", uplinkDeviceSns);
        s.setValue("SpoolDir", uplinkSpoolDir);
        s.setValue("RetryIntervalMs", uplinkRetryIntervalMs);
        s.setValue("DayCutHour", uplinkDayCutHour);
        s.setValue("StatBasis", uplinkStatBasis);
        s.setValue("TzName", uplinkTzName);
        s.setValue("UtcOffsetSeconds", uplinkUtcOffsetSeconds);
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
        rtspUrl = s.value("Camera/RTSP_URL",
            "rtsp://192.168.99.10:8554/live/camera").toString();

        // 显示设置
        windowTitle = s.value("Display/WindowTitle", "HX-CCD23 人员检测 ReID 系统").toString();

        // 图像保存
        savePath = s.value("ImageSave/SavePath", "./save_images").toString();
        saveFormat = s.value("ImageSave/SaveFormat", "jpg").toString();
        jpegQuality = s.value("ImageSave/JpegQuality", 95).toInt();

        // 推理参数
        enableInference = s.value("Inference/EnableInference", true).toBool();
        modelPath = s.value("Inference/ModelPath", "./model/yolo26n_trt10_x86.engine").toString();
        detectorConfThreshold = static_cast<float>(s.value("Inference/ConfThreshold", 0.45).toDouble());
        detectorNmsThreshold = static_cast<float>(s.value("Inference/NmsThreshold", 0.5).toDouble());

        // 跟踪参数
        trackerIouThreshold = static_cast<float>(s.value("Tracker/IouThreshold", 0.3).toDouble());
        trackerMaxMissed = s.value("Tracker/MaxMissed", 30).toInt();
        trackerKalmanEnabled = s.value("Tracker/KalmanEnabled", true).toBool();
        trackerReidEnabled = s.value("Tracker/ReidEnabled", true).toBool();
        trackerReidThreshold = static_cast<float>(s.value("Tracker/ReidThreshold", 0.65).toDouble());
        trackerGalleryMaxAgeMs = s.value("Tracker/GalleryMaxAgeMs", 10000).toInt();
        trackerMergeSimilarity = static_cast<float>(s.value("Tracker/MergeSimilarity", 0.85).toDouble());
        trackerMergeMaxYoungAge = s.value("Tracker/MergeMaxYoungAge", 20).toInt();

        // ReID 参数
        reidMatchThreshold = static_cast<float>(s.value("ReID/MatchThreshold", 0.70).toDouble());
        reidSavePath = s.value("ReID/SavePath", "./save_images/reid").toString();
        regionSavePath = s.value("ReID/RegionSavePath", "./regions.json").toString();
        visitorLogPath = s.value("ReID/VisitorLogPath", "./visitor_log").toString();

        // 动恰客流上行
        uplinkEnabled = s.value("Uplink/Enabled", true).toBool();
        uplinkBaseUrl = s.value("Uplink/BaseUrl", "https://footfall.dongqia.com").toString();
        uplinkIntervalPath = s.value("Uplink/IntervalPath",
            "/api/passenger-flow/interval-aggregate").toString();
        uplinkPersonPath = s.value("Uplink/PersonPath",
            "/api/passenger-flow/person-event").toString();
        uplinkUser = s.value("Uplink/User", "dongqia01").toString();
        uplinkPass = s.value("Uplink/Pass", "123456").toString();
        uplinkMasterSn = s.value("Uplink/MasterSn", "CAM-A01").toString();
        uplinkDeviceSns = s.value("Uplink/DeviceSns", "").toString();
        uplinkSpoolDir = s.value("Uplink/SpoolDir", "./uplink_spool").toString();
        uplinkRetryIntervalMs = s.value("Uplink/RetryIntervalMs", 5000).toInt();
        uplinkDayCutHour = s.value("Uplink/DayCutHour", 0).toInt();
        uplinkStatBasis = s.value("Uplink/StatBasis", "enter").toString();
        uplinkTzName = s.value("Uplink/TzName", "Asia/Shanghai").toString();
        uplinkUtcOffsetSeconds = s.value("Uplink/UtcOffsetSeconds", 28800).toInt();
    }

    /// 根据 IP/User/Pass/Port 重建 RTSP URL
    void rebuildRtspUrl() {
        rtspUrl = QString("rtsp://%1:%2@%3:%4/live/camera")
            .arg(cameraUser, cameraPass, cameraIP).arg(cameraPort);
    }
};

#endif // APP_CONFIG_H
