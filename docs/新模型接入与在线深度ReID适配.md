# 新模型接入与在线深度 ReID 适配（head 检测 + 微调 OSNet）

> 面向本次训练交付的两套模型：
> - **阶段一 head 头部检测**：YOLO11n，单类，ONNX `1×5×8400`，fp16 engine（RTX 2060 + TRT 10.3）
> - **阶段二 OSNet 现场微调 ReID**：OSNet x1.0 + MSMT17 预训练，输出 `1×512×16×8`
>
> 之前 `DeepReID` 只在「离线检测」里使用，**在线主流程仍走手工颜色/纹理特征**。
> 本次改动把微调后的深度 ReID 贯通到在线链路，并为 head 检测模型补齐过滤/裁剪适配。

---

## 一、关键改动一览

| 方向 | 改动 | 位置 |
|---|---|---|
| 在线接入深度 ReID | `ReIDExtractor` 增加深度后端委派（`extract/compare` 自动改走 OSNet） | [`ReIDExtractor::setDeepReid()`](CameraViewer/src/reidextractor.h:99) |
| 阈值切换 | 深度启用时用 `DeepMatchThreshold`，否则用 `MatchThreshold` | [`MainWindow::effectiveReidThreshold()`](CameraViewer/src/mainwindow.cpp:1946) |
| 全局注入 | 一处加载，注入 Tracker/ZoneMonitor/PassengerBook/比对线程 | [`MainWindow::initDeepReid()`](CameraViewer/src/mainwindow.cpp:1934) |
| 批量比对适配 | 深度后端直接余弦相似度 + 匈牙利全局指派（跳过分量动态加权） | [`IdentityMatchWorker::runBatch()`](CameraViewer/src/identitymatchworker.cpp:78) |
| 特征归一化 | 非手工维度（如 512 深特征）整体 L2 归一化（余弦前提） | [`ReIDExtractor::temporalFuse()`](CameraViewer/src/reidextractor.cpp:470) |
| head 检测过滤 | 新增 `detectorModelKind=person\|head`，head 用独立尺寸/长宽比参数 | [`MainWindow::applyDetectorFilters()`](CameraViewer/src/mainwindow.cpp:1962) |
| 头→身裁剪 | head 框按系数扩展后再送 ReID（更接近训练分布） | [`ReIDExtractor::setBoxExpandForReid()`](CameraViewer/src/reidextractor.h:159) |
| 输出兼容 | `DeepReID` 同时支持 4D `1×C×H×W` 与 2D `1×C` 导出 | [`DeepReID::embed()`](CameraViewer/src/reiddeep.cpp:97) |

---

## 二、为什么需要这些改动

1. **在线未用深度模型**：`config.ini` 里 `ReID/UseDeepReid=true` 此前只影响离线窗；
   在线上下车比对仍用手工特征（实测同人/异人分数重叠），微调模型的价值无法体现在主链路。
2. **head 框会被人体过滤参数误杀**：人体默认 `minBoxHeightPx=90`、`minHeightRatio=0.08`、
   宽高比 `[0.12, 1.00]`；头部框小且近似方形（可能 >1.0），会**全部被过滤**导致 0 检测。
3. **训练分布是人体裁剪**：深度 ReID 数据来自 `visitor_V*`（人体裁剪），
   直接用头框裁剪与训练分布不一致，故引入「头→身」适度扩展。

---

## 三、配置项（config.ini）

```ini
[Inference]
ModelPath=./model/head_yolo11n_trt10_x86.engine
; 目标类型：person / head（head 使用下方独立过滤参数）
ModelKind=head
HeadMinPersonHeightRatio=0.0100
HeadMinPersonAreaRatio=0.00020
HeadMinBoxWidthPx=10
HeadMinBoxHeightPx=10
HeadMinBoxAspect=0.35
HeadMaxBoxAspect=3.00

[ReID]
UseDeepReid=true
DeepModelPath=./model/osnet_x1_0_head_finetuned.onnx
DeepMatchThreshold=0.60
; 头框 → 人体框 的 ReID 裁剪扩展系数（1.0=不扩展）
HeadBoxExpand=1.35
```

- `ModelKind` 由 [`AppConfig::detectorModelKind`](CameraViewer/src/app_config.h:52) 读取，
  在 [`applyDetectorFilters()`](CameraViewer/src/mainwindow.cpp:1962) 中决定用哪套过滤参数
  （在线采集与离线检测共用）。
- 深度后端启用后，`ZoneMonitor`/`PassengerBook`/`IdentityMatchWorker` 的匹配阈值
  统一取 `DeepMatchThreshold`；关闭深度则回退 `MatchThreshold`。

---

## 四、兼容与回滚

- **回滚到旧模型**：把 `config.ini` 改回
  `ModelPath=./model/yolo26n_trt10_x86.engine`、`ModelKind=person`、
  `UseDeepReid=false`（或 `DeepModelPath=./model/osnet_x1_0_msmt17.onnx`）即可，
  旧模型文件均已保留。
- **深度模型加载回退链**：`DeepModelPath` → 运行目录 head 微调 → 运行目录 MSMT17 →
  源码目录 head 微调 → 源码目录 MSMT17；全部失败则自动回退手工特征（不中断采集）。
- **engine 与硬件绑定**：`*.engine` 与 GPU 架构 + TensorRT 版本绑定，换机需重跑 `trtexec`
  重建 fp16 engine（构建时不要传已废弃的 `--workspace`）。ONNX 不受影响，engine 缺失会自动回退 ONNX。

---

## 五、构建与验证

主程序：

```sh
cd build-main-debug
/home/froduo/Qt/6.2.4/gcc_64/bin/qmake ../CameraViewer/CameraViewer.pro -spec linux-g++ CONFIG+=debug
make -j4
```

命令行复核工具（`ReIDExtractor` 现已依赖 `reiddeep.cpp`，脚本已更新）：

```sh
sh build_clip_check.sh      # 产物 ./offline_clip_check
```

运行时日志核对：

- `[MainWindow] ReID 后端=深度OSNet(512D) (deep=on, headBoxExpand=1.35)`
- `[MainWindow] 深度 ReID 已加载: ... (生效阈值=0.60)`
- `[MainWindow] 检测类型=head（头部检测），过滤 hRatio>=... aRatio>=... minPx=10x10 aspect∈[0.35,3.00]`
- 若出现 `Found N detections but NONE are valid ... after size filtering`，
  说明过滤仍偏严，可下调 `HeadMinBoxHeightPx/HeadMinPersonAreaRatio`。

内存安全回归（ASan，需额外加入 `reiddeep.cpp`）：

```sh
QT=/home/froduo/Qt/6.2.4/gcc_64
$QT/libexec/moc CameraViewer/src/zonemonitor.h    -o /tmp/moc_zm.cpp -I CameraViewer/src
$QT/libexec/moc CameraViewer/src/regionmanager.h  -o /tmp/moc_rm.cpp -I CameraViewer/src
g++ -std=c++17 -g -O1 -fsanitize=address -fno-omit-frame-pointer \
    tools/doorline_asan_test.cpp \
    CameraViewer/src/{zonemonitor,regionmanager,persontracker,reidextractor,reiddeep,backgroundmodel}.cpp \
    /tmp/moc_zm.cpp /tmp/moc_rm.cpp -I CameraViewer/src -I $QT/include -I $QT/include/QtCore \
    -I $QT/include/QtGui -I /usr/local/include/opencv4 -L $QT/lib -lQt6Core -lQt6Gui \
    -lopencv_core -lopencv_imgproc -lopencv_video -lopencv_videoio -lopencv_highgui \
    -lopencv_imgcodecs -lpthread -o /tmp/door_asan
ASAN_OPTIONS=detect_leaks=0 /tmp/door_asan     # 期望：无 ASan 报错，board=1 alight=1
```

---

## 六、调参建议

| 现象 | 调优方向 |
|---|---|
| 检测框偏少/漏人 | 下调 `HeadMinBoxHeightPx`、`HeadMinPersonAreaRatio`、`HeadMinBoxAspect` 下限 |
| 误检多（背景框） | 上调 `HeadMinPersonAreaRatio` / `HeadMinBoxHeightPx`，或提高 `ConfThreshold` |
| 上/下车比对偏严（多为“异常乘客”） | 下调 `DeepMatchThreshold`（如 0.55） |
| 比对偏松（错配） | 上调 `DeepMatchThreshold`（如 0.65~0.70） |
| 头框裁剪信息不足 | 上调 `HeadBoxExpand`（如 1.5），或改用人体检测模型 |
| 在线卡顿 | 关闭深度（`UseDeepReid=false`）或降低检测/跟踪帧率；深度推理在共享 `DeepReID` 内串行加锁 |

---

## 七、「视频里有人却检测不到」根因与修复

### 7.1 现象与定位过程

现象：视频中明显有人，但轨迹为 0 / 不识别。
逐层排查（命令行复核工具 + 同一帧对照）：

| 试验 | 结果 | 结论 |
|---|---|---|
| head **ONNX** + OpenCV-DNN（squash 预处理） | 轨迹 2 条 | **模型本身有效** |
| head **TRT engine** + 原 letterbox 预处理 | `maxScoreAll≈0.04`，0 框 | 引擎预处理/模型不匹配 |
| 重建 engine（fp16 / fp32 / opset12+simplify） | 仍 ≈0.04 | 与引擎精度、图优化无关 |
| 同一 ONNX 逐帧扫描类别通道最大值 | letterbox=**0.233**，squash=**0.724** | **长宽比处理是根因** |

逐通道诊断（[`TrtYolo::postprocess()`](CameraViewer/src/trt_yolo.cpp:257)）显示
`cx/cy/w/h` 通道量级正常（~637/634/298/377），只有**类别通道**极低，
说明引擎计算正确，问题在**输入预处理方式**。

### 7.2 两处真实缺陷（均已修复）

1. **head 框被人体的尺寸/长宽比过滤全部误杀**
   人体默认 `minBoxHeightPx=90`、`heightRatio=0.08`、宽高比 `[0.12,1.00]`；
   头部框小且近似方形 → 全被过滤。已引入 `ModelKind=person|head` 与独立过滤参数
   （[`applyDetectorFilters()`](CameraViewer/src/mainwindow.cpp:1985)）。

2. **TRT 预处理 letterbox 与该 head 模型不匹配**
   该模型在 letterbox 下类别分数上限仅 0.23（<0.45 阈值）→ 永远无框；
   改用等比拉伸(squash)后达 0.49~0.72 → 正常出框。
   已把预处理模式做成可配置：
   - [`TrtYolo::setLetterbox()`](CameraViewer/src/trt_yolo.h:52)（letterbox / squash，
     并新增 `scaleX/scaleY` 支持横纵不同缩放）；
   - [`PersonDetector::setLetterbox()`](CameraViewer/src/persondetector.h:72)；
   - 配置项 `Inference/Letterbox`，且**缺省随 `ModelKind` 联动**
     （head→false，person→true，见 [`AppConfig::load()`](CameraViewer/src/app_config.h:267)）。

### 7.3 验证

```sh
sh build_clip_check.sh
./offline_clip_check <clip.mp4> 3 CameraViewer/model/head_yolo11n_trt10_x86.engine \
    build-CameraViewer-main-Desktop-Debug/regions.json head
# 期望：TRT 预处理: squash；maxScoreAll≈0.49~0.57；轨迹 >=1 条
./offline_clip_check <clip.mp4> 3 CameraViewer/model/yolo26n_trt10_x86.engine \
    build-CameraViewer-main-Desktop-Debug/regions.json person
# 期望：letterbox，行为与旧版一致（无回归）
```

### 7.4 新增诊断能力

[`TrtYolo`](CameraViewer/src/trt_yolo.cpp:296) 现在会：
- 前 3 次推理打印**逐通道最大值**（判断是否存在通道错位/整段失效）；
- 之后每 25 帧打印一次 `maxScoreAll`；
- 当 `maxScoreAll < 0.05` 时输出 `[WARN] engine 输出置信度异常偏低…建议改用 ONNX 或重建 engine`。
