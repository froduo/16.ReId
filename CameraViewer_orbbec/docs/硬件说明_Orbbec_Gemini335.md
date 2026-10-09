# 硬件说明 —— Orbbec Gemini 335 深度相机

> 本文为 `CameraViewer_orbbec` 程序的**实测硬件规格**文档（数据由 `tests/ob_info` 与
> `tests/test_source` 在本机实测得到，非手册抄录），覆盖 RGB 像素、深度像素、分辨率、
> 帧率、内参/FOV、深度单位、D2C 对齐与数据格式约定。

---

## 1. 设备与连接

| 项目 | 值 |
|---|---|
| 型号 | Orbbec **Gemini 335** |
| 序列号 (SN) | `CP08863000NK` |
| 固件 (FW) | `1.4.60` |
| PID / VID | `0x0800` / `0x2BC5` |
| 连接方式 | **USB 3.2**（UVC） |
| SDK | OrbbecSDK **v2.10.6**（`third_party/OrbbecSDK_v2.10.6_..._linux_x86_64`）|
| 安装方位 | 车门正上方**俯视**（上下车检测场景）|

---

## 2. 彩色（RGB）流

### 2.1 程序默认使用的 profile
程序默认启用**彩色 + 深度**，未指定分辨率时使用设备默认：**MJPG 1280×720 @ 30fps**。

### 2.2 可用彩色 profile（节选）

| 分辨率 | 支持格式 | 帧率 |
|---|---|---|
| 1920×1080 | MJPG / RGB / RGBA / BGR / BGRA / Y8 / Y16 / YUYV | 30 / 15 / 10 / 6 |
| 1280×720 | 同上 | 60 / 30 / 15 / 10 / 6 |
| 960×540 | 同上 | 60 / … |

### 2.3 彩色内参（针孔模型）

| 分辨率 | fx | fy | cx | cy | HFOV | VFOV |
|---|---|---|---|---|---|---|
| **1280×720** | 688.53 | 688.75 | 644.63 | 360.29 | 85.8° | 55.2° |
| 1920×1080 | 1032.80 | 1033.12 | 966.94 | 540.44 | 85.8° | 55.2° |
| 960×540 | 516.40 | 516.56 | 483.47 | 270.22 | 85.8° | 55.2° |

畸变系数（Brown-Conrady，随分辨率同比缩放）：
`k1=0.0060  k2=-0.0424  k3=0.0296  p1=-0.0005  p2=-0.0000`

### 2.4 像素格式与内存布局

- 传输格式：**MJPG**（默认）或未压缩 RGB/BGR/RGBA/BGRA/YUYV 等。
- **程序内部统一为 `cv::Mat` `CV_8UC3`（BGR，8bit×3通道）**，MJPG 经 `cv::imdecode` 解码。

---

## 3. 深度（Depth）流

### 3.1 程序默认 profile
**Y16 848×480 @ 30fps**（物理原始分辨率；对齐后见 §4）。

### 3.2 可用深度 profile（节选，均为 Y16）

| 分辨率 | 帧率 | fx=fy | cx | cy | HFOV | VFOV |
|---|---|---|---|---|---|---|
| **848×480** | 30/60/15/10/6 | 412.06 | 423.34 | 238.68 | 91.6° | 60.4° |
| 1280×800 | 30/15/10/6 | 621.98 | 639.00 | 398.00 | 91.6° | 65.5° |
| 1280×720 | 30/15/10/6 | 621.98 | 639.00 | 358.00 | 91.6° | 60.1° |
| 640×480 | 90/60/30/15/10/6 | 373.19 | 319.40 | 238.00 | 81.2° | 65.5° |
| 640×360 | 90/60/30/… | 312.16 | 319.00 | 179.00 | 91.4° | 59.9° |
| 480×270 / 424×266 / 424×240 / 848×100 | 多档 | … | … | … | … | … |

> 深度畸变系数均为 0（该模组深度出厂已标定/免畸变）。

### 3.3 深度数值约定

| 项目 | 值 |
|---|---|
| 原始像素类型 | **`uint16`（Y16/Z16）** |
| **depthValueScale** | **1.0**（即：**原始数值 = 毫米**，无需换算） |
| 无效像素 | `0` |
| 量程（硬件） | 约 0.2–5 m（程序配置有效范围 `depthMinMM=300` ~ `depthMaxMM=3000`）|
| 程序内部类型 | `cv::Mat` **`CV_16U`（毫米，0 表示无效）** |

---

## 4. D2C（Depth-to-Color）对齐 —— 重要

### 4.1 实测结论
- **硬件对齐 `ALIGN_D2C_HW_MODE` 在当前 profile 下不被支持**：
  `Current stream profile is not support hardware d2c process`。
- **`config->setAlignMode(ALIGN_D2C_SW_MODE)` 无效**：启动后深度帧仍为 **848×480**，
  并未与彩色 1280×720 逐像素对应（实测 SW 与 OFF 结果相同）。
- **必须使用 SDK 的 `ob::Align` 滤镜**才能真正对齐：

```cpp
ob::Align align(OB_STREAM_COLOR);        // 目标 = 彩色
align.setMatchTargetResolution(true);    // 输出分辨率 = 彩色分辨率
auto out = align.process(frameset);      // out 的深度帧 = 1280×720
```

### 4.2 对齐后
| 项目 | 对齐前（原始深度） | 对齐后（程序实际输出） |
|---|---|---|
| 深度分辨率 | 848×480 | **1280×720（= 彩色）** |
| 像素对应 | 否 | **是（逐像素一一对应）** |
| 单位 | mm | mm（scale=1.0） |

程序已按此实现：`OrbbecSource` 优先尝试 HW，失败则自动改用 `ob::Align` 滤镜，
并暴露 `alignMode() / alignedToColor() / depthWidth() / depthHeight() / depthScale()`。

---

## 5. 帧同步与取帧

- `pipe->enableFrameSync()`：按时间戳同步彩色/深度。
- `config->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE)`：
  保证一个帧集内彩色与深度都齐备（否则 `getColorFrame()` 可能为空）。
- 程序以**阻塞式 `waitForFrameset(timeout)`** 取帧，超时 500ms。

---

## 6. 程序内的数据格式约定（`OrbbecSource::getFrames`）

| 输出 | 类型 | 说明 |
|---|---|---|
| `colorBGR` | `CV_8UC3` | BGR 彩色（MJPG 已解码） |
| `depthMM` | `CV_16U` | 毫米；**与 colorBGR 同分辨率（对齐后）**；0=无效 |
| `colorTsUs`/`depthTsUs` | `uint64` | 设备时间戳（微秒） |

---

## 7. 复现 / 验证命令

```bash
cd CameraViewer_orbbec
SDK=third_party/OrbbecSDK_v2.10.6_202609291004_575347c_linux_x86_64

# 1) 硬件信息（设备/全部 profile/内参/FOV/对齐）
g++ -std=c++17 tests/ob_info.cpp -I$SDK/include \
    -L$SDK/lib -lOrbbecSDK \
    -Wl,-rpath,$PWD/$SDK/lib \
    -Wl,-rpath,$PWD/$SDK/lib/extensions/depthengine \
    -Wl,-rpath,$PWD/$SDK/lib/extensions/filters -o tests/ob_info
./tests/ob_info

# 2) 采集连通性 + D2C 对齐验证
g++ -std=c++17 tests/test_source.cpp src/orbbecsource.cpp -Isrc -I$SDK/include \
    -I/usr/local/include/opencv4 \
    -L$SDK/lib -lOrbbecSDK -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs \
    -Wl,-rpath,$PWD/$SDK/lib \
    -Wl,-rpath,$PWD/$SDK/lib/extensions/depthengine \
    -Wl,-rpath,$PWD/$SDK/lib/extensions/filters \
    -Wl,-rpath,/usr/local/lib -o tests/test_source
./tests/test_source
# 期望输出：align=SW(Align滤镜)；color=1280x720 depth=1280x720 aligned=yes(pixel-wise)
```

---

## 8. 关键参数速查（供其它模块引用）

```
彩色: 1280x720 @30fps, BGR 8UC3
      fx=688.53 fy=688.75 cx=644.63 cy=360.29  (HFOV 85.8°, VFOV 55.2°)
深度: 对齐后 1280x720, CV_16U 毫米, scale=1.0, 0=无效
      原始内参(848x480): fx=fy=412.06 cx=423.34 cy=238.68 (HFOV 91.6° VFOV 60.4°)
对齐: HW 不支持 → ob::Align(OB_STREAM_COLOR)+MatchTargetResolution
```
