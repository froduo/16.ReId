# CameraViewer - 任务跟踪日志

> **使用方法**: 每次开始工作前加载此文件了解项目状态和待办任务。
> 每次完成一个任务后更新对应的 checkbox 和日志条目。

---

## 📋 项目概览

- **项目名称**: HX-CCD23 人员检测 ReID 系统 (CameraViewer)
- **技术栈**: Qt 6.2.4 + OpenCV 4.8.0 + TensorRT 10.3 + CUDA
- **构建系统**: qmake (`CameraViewer.pro`)
- **构建命令**: `./build.sh` 或 `cd build-CameraViewer-Desktop-Debug && qmake ../CameraViewer/CameraViewer.pro && make -j4`
- **运行路径**: `build-CameraViewer-Desktop-Debug/CameraViewer`
- **相机 RTSP**: `rtsp://192.168.99.10:8554/live/camera`

---

## 🔴 已知问题 (Known Issues)

### ISSUE-001: YOLO ONNX 模型加载失败
- **状态**: 🔴 未解决
- **描述**: OpenCV 4.8.0 不支持 `yolo26n.onnx` 中的 `TopK` 算子层
- **错误信息**: `Can't create layer "onnx_node!/model.23/TopK" of type "TopK"`
- **影响**: 人员检测功能完全不可用，应用只能显示视频流
- **可选解决方案**:
  1. 升级 OpenCV 到 4.10+ 以支持 TopK 算子
  2. 使用 TensorRT 引擎文件 (`yolo26n_trt10_x86.engine`) 替代 ONNX
  3. 切换推理后端为 ONNX Runtime
  4. 导出不包含 TopK 层的兼容 ONNX 模型

### ISSUE-002: TensorRT 引擎文件缺失
- **状态**: 🔴 未解决
- **描述**: 构建目录中只有 `yolo26n.onnx`，缺少 `yolo26n_trt10_x86.engine`
- **默认配置路径**: `./model/yolo26n_trt10_x86.engine`
- **影响**: 无法使用 TensorRT 后端推理
- **解决方案**: 需要使用 `trtexec` 将 ONNX 转换为 TensorRT 引擎

### ISSUE-003: CMakeLists.txt 与实际源码不匹配
- **状态**: 🔴 未解决
- **描述**: `CameraViewer/CMakeLists.txt` 引用了不存在的 `camera_thread.cpp/h`，缺少多个实际源文件
- **影响**: 无法使用 CMake 构建（当前使用 qmake）
- **解决方案**: 更新 CMakeLists.txt 以匹配 `.pro` 文件中的源文件列表

---

## ✅ 已完成任务 (Completed Tasks)

### TASK-001: 基础视频采集框架
- **完成日期**: 2026-09-11
- **描述**: RTSP 视频流采集、显示、手动/自动保存
- **涉及文件**: `cameracontroller.cpp/h`, `grabthread.cpp/h`, `saveimageworker.cpp/h`

### TASK-002: YOLO 检测管线
- **完成日期**: 2026-09-11
- **描述**: PersonDetector (OpenCV DNN + TensorRT), PersonTracker (IoU), ReIDExtractor (颜色直方图)
- **涉及文件**: `persondetector.cpp/h`, `persontracker.cpp/h`, `reidextractor.cpp/h`, `trt_yolo.cpp/h`

### TASK-003: 区域绘制与管理
- **完成日期**: 2026-09-14
- **描述**: 多边形/矩形区域绘制、归一化坐标存储、方向感知穿越检测
- **涉及文件**: `imageview.cpp/h`, `regionmanager.cpp/h`, `mainwindow.cpp/h`

### TASK-004: 访客身份管理 (ReID)
- **完成日期**: 2026-09-14
- **描述**: 跨 track 的 ReID 匹配、访客记录表格、进入/离开事件处理
- **涉及文件**: `mainwindow.cpp` (VisitorRecord, matchOrCreateVisitor, updateVisitorExit)

### TASK-005: 离线检测功能
- **完成日期**: 2026-09-14
- **描述**: 离线图片检测表单
- **涉及文件**: `offlinedetectform.cpp/h`

### TASK-006: 系统配置界面
- **完成日期**: 2026-09-14
- **描述**: 参数配置表单（模型路径、阈值、保存路径等）
- **涉及文件**: `settingform.cpp/h`, `settingform.ui`

### TASK-007: 背景分割 CUDA 加速（核心图像程序 GPU 化）
- **完成日期**: 2026-09-29
- **描述**: `BackgroundModel` 新增 OpenCV CUDA 后端（`USE_OPENCV_CUDA`）：帧差分、灰度化、
  阈值化、形态学开/闭运算与前景占比统计全部在 GPU 上完成，仅上传原始帧/下载二值掩码，
  大幅降低 CPU 占用。
- **涉及文件**: `backgroundmodel.cpp/h`、`CameraViewer.pro`（链接 cudaarithm/cudaimgproc/cudafilters）
- **关键修复**:
  - 形态学滤波器按 `CV_8UC1` 创建，前景掩码统一导出为 `CV_8UC1`，修复
    `opencv_cudafilters` 的 `src.type() == type_` 断言崩溃（CPU/GPU 两条路径行为一致）。
  - `cv::cuda` 不提供 `accumulateWeighted`，背景学习改用等价的 `addWeighted`
    （`dst = lr*src + (1-lr)*dst`）。
  - `m_useGpu` 移出 `#ifdef USE_OPENCV_CUDA`，保证关闭 CUDA 后 CPU 回退仍可编译。
- **验证**: 合成帧单元测试显示 GPU（`usingGpu=1`）与 CPU（`usingGpu=0`）两路掩码均为
  `CV_8UC1` 且无异常；ReID 端到端特征提取维度 = 61478、数值有限。

### TASK-008: 检测逻辑重构（中心点轨迹判方向 + 人体框交集判保存窗口）
- **完成日期**: 2026-09-29
- **描述**: 按最终业务逻辑重写 `ZoneMonitor`：
  1. 在整个图像中识别乘客（PersonDetector 全帧检测），跟踪其矩形检测框与中心点；
  2. **保存窗口由人体框与检测区域的交集驱动**：矩形检测框与检测区域**开始有交集**时
     开始保存该乘客整段图像，矩形检测框**离开**检测区域（不再相交）时停止保存；
  3. **方向由中心点轨迹判定**：中心点**从上往下**（y 逐渐增大）穿过检测区域 = 上车；
     中心点**从下往上**（y 逐渐减小）穿过检测区域 = 下车；
  4. 上车时登记乘客（图片 + 特征），下车时用下车整段图像特征与所有上车乘客图像做 GPU
     特征比对，判定是哪一位乘客下车（复用 `PassengerBook::matchAndRemove`）。
- **涉及文件**: `zonemonitor.cpp/h`（状态机重写，`CaptureState{Idle,Capturing}`、
  `ChannelZone` 增加外接矩形边界）、`mainwindow.cpp`（事件接线与图片保存保持不变）
- **关键实现**:
  - 由「中心点是否在多边形内」改为「人体框 ∩ 区域外接矩形」面积 > 0 作为保存开关；
  - **方向判定仅由乘客中心点 y 轨迹决定**（不依赖人体框是否与区域有交集）：
    起始 y 取「进入检测区域之前」一帧（人体框与区域无交集）的中心点 y，结束 y 取
    停止保存时的中心点 y；`start<mid<end` 且 y 逐渐增大 ⇒ 上车；
    `start>mid>end` 且 y 逐渐减小 ⇒ 下车（`mid`=区域中心线 y）。擦边/折返不被计入；
  - 轨迹在保存途中丢失时按累计方向强制结算，避免漏事件。
- **验证**: 合成轨迹单元测试（640×480，区域 y∈[0.4,0.6]）：
  - 场景1 中心点 0.12→0.88 → `boarding=1`，保存 6 帧 → 通过；
  - 场景2 中心点 0.88→0.12 → `alighting=1`，保存 6 帧 → 通过；
  - 场景3 从上方进入后折返向上 → 无任何事件（正确丢弃）→ 通过；总结果 `PASS`。

### TASK-009: 下车身份比对线程化（不阻塞实时管线）
- **完成日期**: 2026-09-29
- **背景 / 帧率结论**: 相机源 1920×1080 @ 25fps；`GrabThread` 内联执行 TensorRT 检测，
  日志中 500 帧耗时约 20.0s ⇒ 处理帧率约 **20–25 FPS**（重负载时降到约 20fps）。
  原先 [`PassengerBook::matchAndRemove()`](CameraViewer/src/passengerbook.cpp:71) 在主线程
  同步执行 O(在车乘客 × 探测帧 × 模板帧) 次 61478 维特征比对，会阻塞 UI 与实时管线。
- **描述 / 方案**: 新增 [`IdentityMatchWorker`](CameraViewer/src/identitymatchworker.h:23)
  （QObject 置于独立 QThread）：
  - 下车事件只做主线程"取快照 + 投递任务"，比对在后台线程串行执行；
  - 台账（`PassengerBook`）增删仍留在主线程（新增 `onboardTemplates()` 深拷贝快照与
    `removePassenger()`），避免跨线程共享可变状态；
  - 比对完成后经 Qt 队列连接回到主线程结算（命中/回退/异常判定、图片保存、访客记录）。
- **涉及文件**: `identitymatchworker.cpp/h`（新增）、`passengerbook.cpp/h`、`mainwindow.cpp/h`、
  `CameraViewer.pro`
- **验证**: 单元测试确认"提交→工作线程计算→结果回主线程"链路：
  `gotId=1, gotSim=1.000, matched=1, onMainThread=1 → PASS`；主工程重新编译链接通过。

### 建议：准确率优先的进一步优化路线（对比速度可放松）
- **P1 深度 ReID 嵌入（收益最大）**: 将 61478 维手工特征（HSV/梯度/身形）替换为
  512 维深度学习 ReID 嵌入（OSNet / LightReID 等），用 TensorRT 推理并 L2 归一化，
  余玄相似度判别力远高于颜色直方图。
- **P2 相似度计算上 GPU**: 所有在车模板与探测帧构成矩阵，一次性批量算余弦相似度
  （cuBLAS / OpenCV CUDA），把后台比对从"逐对比对"变为"一次矩阵乘"。
- **P3 提高采样密度**: `SAMPLE_INTERVAL` 降为 1、`MAX_FEATURE_SAMPLES` 提升到 60，
  整段序列更完整（代价是特征提取更重，可放到工作线程）。
- **P4 双向序列相似度**: 现有 `sequenceSimilarity` 为 probe→templ 单向；改为双向取最大/平均，
  减少方向性偏差。
- **P5 时空约束辅助**: 结合最近上车时间、区域/门位、轨迹连续性做候选裁剪与打分融合，
  进一步降低误配。

---

## 📝 待办任务 (Pending Tasks)

### TASK-010: 修复 YOLO 模型加载
- **优先级**: 🔴 高
- **描述**: 解决 ISSUE-001，使人员检测功能可用
- **子任务**:
  - [ ] 确定最佳解决方案（升级 OpenCV / TensorRT 引擎 / ONNX Runtime）
  - [ ] 实施选定方案
  - [ ] 验证检测功能正常工作

### TASK-011: 更新 CMakeLists.txt
- **优先级**: 🟡 中
- **描述**: 解决 ISSUE-003，使项目可用 CMake 构建
- **子任务**:
  - [ ] 将所有实际源文件添加到 CMakeLists.txt
  - [ ] 添加 TensorRT 和 CUDA 依赖
  - [ ] 验证 CMake 构建成功

### TASK-012: 生成 TensorRT 引擎文件
- **优先级**: 🟡 中
- **描述**: 解决 ISSUE-002，使用 trtexec 转换 ONNX 为 TensorRT 引擎
- **子任务**:
  - [ ] 确认 trtexec 路径和版本
  - [ ] 执行 ONNX → TensorRT 转换
  - [ ] 验证引擎文件可正常加载

### TASK-013: ReID 特征提取改进
- **优先级**: 🟢 低
- **描述**: 当前使用 HSV 颜色直方图作为 ReID 特征，可考虑使用深度学习模型
- **子任务**:
  - [ ] 评估轻量级 ReID 模型（如 OSNet）
  - [ ] 集成到 ReIDExtractor
  - [ ] 对比改进前后匹配准确率

---

## 📅 开发日志 (Development Log)

### 2026-10-08（CameraViewer_orbbec · Orbbec Gemini 335 适配分支）
- 11:33–11:39 - 由 RTSP 版派生 `CameraViewer_orbbec`：新增 `OrbbecSource`（OrbbecSDK v2，
  彩色 BGR + 深度 mm、D2C 对齐），`CameraController` 支持 `openOrbbec()/grabFrames()`，
  `GrabThread` 缓存深度并发出 `depthFrameGrabbed`，`app_config` 增加 `useOrbbec/depthMinMM/depthMaxMM` 等
- 15:12 - 修复 `orbbecsource.cpp` 三处编译错误：
  `DepthFrame::getDepthScale()` → `getValueScale()`；`const char*` 拼接改为 `std::string`；
  无效的 `depthMM(depthMM > 0)` 掩码取子矩阵改为逐像素收集有效深度值
- 15:20 - 修复模型复制：qmake `for()` 循环内 `QMAKE_POST_LINK += && ...` 与 `*.commands +=`
  均不生效（Makefile 仅剩 `mkdir`），改用独立脚本 `copy_models.sh`；
  构建输出 `model/` 现包含 head_yolo11n(.onnx/.engine) 与 osnet_x1_0_head_finetuned/msmt17.onnx
- 15:30 - 深度视图接入主界面（TASK-014）：
  - 新增只读 `ImageView* m_depthView`，与彩色 `m_imageView` 一起放入垂直 `QSplitter`
    （`m_mediaSplitter`，彩色:深度≈3:2，可拖动分隔条），插入原 `imageLayout` 图像槽位，保持 14:11 拉伸
  - 连接 `GrabThread::depthFrameGrabbed → MainWindow::onDepthFrameGrabbed`（启动采集与自动重连两处）
  - 可见性：仅 Orbbec 且 `m_camera.hasDepth()` 时显示；RTSP 模式隐藏
  - 尺寸自适应：每帧 `setImage()` 后调用 `resetView()`，随窗口/分屏尺寸等比铺满
  - 构建通过，产物：`build-CameraViewer_orbbec-Desktop-Debug/CameraViewer_orbbec`
- 15:44 - 运行 Orbbec 程序并修复相机打开失败（TASK-015）：
  - 现象：点击“开始采集”报 `ob_pipeline_start_with_config: Current stream profile is not support hardware d2c process`
  - 定位：独立探测 `tests/test_align.cpp` 显示 HW 对齐 start 失败，SW/OFF 对齐可正常 start
  - 修复 `OrbbecSource::open()`：将「设置对齐模式 + 启动」作为整体逐级重试（HW→SW→OFF），
    且每种模式都从零重建 `Pipeline`/`Config`（失败后复用旧 Pipeline 不可靠）
  - 验证 `tests/test_source.cpp`：打开成功 Gemini 335 sn=CP08863000NK 1280x720，深度 848x480，
    彩色/深度各 30/30 帧；GUI 点击开始采集后日志 “已打开: Orbbec Gemini 335 ... depth=on”，
    检测管线正常（TensorRT head 模型，detect≈50ms）
  - 遗留：本机（Gemini 335 默认 profile）不支持硬件 D2C；SW 对齐虽可启动但深度仍为原生
    848x480（未与彩色 1280×720 逐像素对齐），如需像素级对应需另用 `ob::Align` 处理
- 16:20 - 用 `16.reId/head image`（人工裁剪人体框，1825 图 / 60 身份，尺寸不一）再训练 ReID（TASK-016）：
  - 新增 `datasets/build_reid_dataset_from_flat.py`：扁平目录 + 文件名 `V#` 前缀 → ImageFolder
    数据集 `datasets/reid_person_head/{train,val}`（丢弃 <4 张的 V9/V70 → 58 身份，train 1276 / val 547）
  - `train_reid.py --root datasets/reid_person_head --epochs 60 --p 8 --k 4` 微调 OSNet
    （最佳 val gap=0.397），导出 ONNX `[1,3,256,128] → [1,512,16,8]`（DeepReID 兼容）
  - 新增 `datasets/evaluate_reid_compare.py`；在新 val 集对比：新微调 gap +0.397 / rank1 0.909，
    优于旧微调（+0.334 / 0.801）与 MSMT17 原始（+0.010 / 0.569）
  - 部署 ONNX + .pth 至 `CameraViewer` / `CameraViewer_dongqia` / `CameraViewer_orbbec` 源 `model/`
    与 main/orbbec 构建 `model/`（旧模型备份为 `*.prev_<时间戳>.onnx`）；重启 Orbbec 程序确认加载新模型
- 备注：Orbbec 版与 RTSP 版共用单实例锁 `/tmp/.CameraViewer.lock`，二者不能同时运行；
  冒烟测试时 RTSP 版（PID 8230）在运行，Orbbec 版被“已在运行”提示框阻塞（非崩溃）

### 2026-09-29
- 15:38 - 完成 `BackgroundModel` 的 CUDA 加速并修复形态学类型断言崩溃，主工程重新构建通过
- 15:40 - 单元测试验证：GPU 与 CPU 路径前景掩码均为 `CV_8UC1`；ReID 特征提取（1×61478）无异常
- 15:58 - 重构检测逻辑（TASK-008）：人体框∩区域驱动图像保存窗口、中心点轨迹判定上下车；
  主工程重新构建通过；合成轨迹单元测试（上车/下车/折返）全部 PASS
- 16:08 - 下车身份比对线程化（TASK-009）：新增后台比对线程；主工程重新构建通过；
  比对链路单元测试 PASS
- 16:19 - 修正方向判定：改为**完全由乘客中心点 y 轨迹**判定（上→下=上车，下→上=下车），
  人体框∩区域仅作保存窗口；起始 y 取进入区域前一帧（与区域无交集）中心点。
  单元测试新增"区域比人体框还矮"场景，全部 PASS

### 2026-09-14
- 11:32 - 版本 v1.0.20260914113237 构建并测试
- 11:33 - 测试自动保存和手动保存功能（正常工作）
- 11:33 - 测试区域绘制功能（正常工作）
- 11:33 - 确认 YOLO ONNX 模型加载失败（TopK 算子不支持）
- 11:38 - 重启应用继续长时间运行测试
- 11:44 - 应用持续运行中（无人检测，仅视频流显示+区域叠加）

### 2026-09-11
- 初始版本开发完成
- 实现基础视频采集、检测、跟踪、ReID 功能

---

## 📂 项目文件结构

```
CameraViewer/
├── CMakeLists.txt          # [过时] CMake 构建文件
├── CameraViewer.pro        # [当前] qmake 构建文件
├── style.qss               # 样式表
├── src/
│   ├── main.cpp
│   ├── mainwindow.cpp/h    # 主窗口逻辑
│   ├── mainwindow.ui       # 主窗口 UI
│   ├── cameracontroller.cpp/h  # RTSP 相机控制
│   ├── grabthread.cpp/h    # 帧采集线程
│   ├── imageview.cpp/h     # 自定义图像显示控件
│   ├── persondetector.cpp/h    # YOLO 人员检测
│   ├── persontracker.cpp/h     # IoU 跟踪器
│   ├── reidextractor.cpp/h     # ReID 特征提取
│   ├── regionmanager.cpp/h     # 区域管理
│   ├── trt_yolo.cpp/h      # TensorRT YOLO 推理
│   ├── saveimageworker.cpp/h   # 后台图像保存
│   ├── offlinedetectform.cpp/h # 离线检测表单
│   ├── settingform.cpp/h   # 系统配置表单
│   ├── settingform.ui      # 配置表单 UI
│   ├── app_config.h        # 应用配置结构
│   └── logger.h            # 日志宏
├── build/                  # qmake 构建输出
└── model/
    └── yolo26n.onnx        # ONNX 模型（缺少 TensorRT 引擎）
```

---

> **最后更新**: 2026-10-08 15:44 CST
> **当前版本**: v1.1.7（CameraViewer_orbbec）
