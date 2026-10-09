# 动恰3DV6 客流时段聚合上行 + 上下车精准识别（对接说明）

本目录是**原程序[`CameraViewer`](../CameraViewer)的独立副本**，原程序保持不变。
本次在此基础上新增了：

1. 《动恰3DV6 AI HTTP 新对接协议20260911》中「客流时段聚合上行协议」的**设备端客户端**；
2. 单人「上车 / 下车」事件的**精准识别追踪与上行**（用于验证某一个人的上车/下车时刻）；
3. **删除历史上下车记录**功能（界面按钮 + 命令行）；
4. 一套**验证工具**（C++ 自检 + Python 回环/联网脚本）。

> 协议原文：`../动恰3DV6 AI HTTP 新对接协议20260911.pdf`（已复制到程序目录），
> 本实现严格对齐其 0~6 章。

---

## 1. 新增/改动文件

| 文件 | 说明 |
|------|------|
| [`src/dongqia_protocol.h`](src/dongqia_protocol.h) | 协议数据结构（分桶取值、`IntervalAggregate`、`PersonFlowEvent`、工具函数） |
| [`src/httpuplink.h`](src/httpuplink.h) / [`src/httpuplink.cpp`](src/httpuplink.cpp) | HTTP 上行客户端：POST 报文、2xx+非空体判定、revision 取大、幂等、离线 spool、固定间隔重试 |
| [`src/flowaggregator.h`](src/flowaggregator.h) / [`src/flowaggregator.cpp`](src/flowaggregator.cpp) | 时段聚合器 + 单人上下车时间线；保证协议恒等式与分桶之和 |
| [`src/main.cpp`](src/main.cpp) | 新增 `--selftest-uplink` 自检模式（不启动 GUI/网络） |
| [`src/mainwindow.h`](src/mainwindow.h) / [`src/mainwindow.cpp`](src/mainwindow.cpp) | 集成：把 `ZoneMonitor` 的上车/下车事件喂给聚合器，并接上行 |
| [`src/app_config.h`](src/app_config.h) | 新增 `[Uplink]` 配置组 |
| [`CameraViewer.pro`](CameraViewer.pro) | 增加 `QT += network` 与新增源文件 |

原程序的其余文件（相机采集、YOLO 检测、跟踪、ReID 抽取、区域管理等）**未做逻辑改动**。

---

## 2. 构建

依赖与原程序一致（Qt 6.2.4 + OpenCV + TensorRT 10.3）。使用副本自带的 qmake 工程：

```bash
# 在项目根目录执行
bash build_dongqia.sh
```

或手动：

```bash
mkdir -p build-dongqia && cd build-dongqia
/home/froduo/Qt/6.2.4/gcc_64/bin/qmake ../CameraViewer_dongqia/CameraViewer.pro -spec linux-g++ CONFIG+=debug
make -j4
```

产物：`build-dongqia/CameraViewer`。

---

## 3. 配置（`config.ini` 的 `[Uplink]` 组）

程序运行时会自动生成/读取 `config.ini`（位于可执行文件目录）。默认值：

```ini
[Uplink]
Enabled=true
BaseUrl=https://footfall.dongqia.com
IntervalPath=/api/passenger-flow/interval-aggregate
PersonPath=/api/passenger-flow/person-event
User=dongqia01
Pass=123456
MasterSn=CAM-A01
DeviceSns=                 ; 逗号分隔；空则取 MasterSn
SpoolDir=./uplink_spool    ; 离线积压本地目录
RetryIntervalMs=5000       ; 失败重试固定间隔
DayCutHour=0               ; 营业日「日切」时间
StatBasis=enter            ; 统计基准方向：enter / leave
TzName=Asia/Shanghai
UtcOffsetSeconds=28800
```

> 说明：平台根地址 `http://footfall.dongqia.com` 会 301 跳到 HTTPS，故默认使用 `https://`。

---

## 4. 协议实现要点（对齐 PDF）

| 协议要求 | 本实现 |
|----------|--------|
| `POST /api/passenger-flow/interval-aggregate`，`application/json` | [`HttpUplink::makeRequest()`](src/httpuplink.cpp:262) |
| 回 2xx 且**响应体非空**才算成功，否则重发 | [`HttpUplink::handleSuccess()`](src/httpuplink.cpp:305) |
| 一件报文 = 一分钟；`interval_s=60`；`end_ts_s=start_ts_s+60` | [`FlowAggregator::buildAggregate()`](src/flowaggregator.cpp:434) |
| 按 `(master_sn, batch_date, start_ts_s)` 整件替换 | [`IntervalAggregate::idempotentKey()`](src/dongqia_protocol.h:167) |
| 同一分钟多次到达按 `revision` 取大 | [`HttpUplink::enqueueInterval()`](src/httpuplink.cpp:118) |
| 幂等：重复投递终态不变 | 幂等键 + `m_bestRevision` 去重 |
| 每个时段只发最新版（等待重试期间被修正则覆盖待发件） | [`HttpUplink::enqueueInternal()`](src/httpuplink.cpp:171) |
| 一件一件发，前一件确认后再发下一件 | [`HttpUplink::onReplyFinished()`](src/httpuplink.cpp:349) |
| 设备侧最多留 2 天，超期清理 | [`HttpUplink::pruneSpool()`](src/httpuplink.cpp:520) |
| `raw_stats.count = final_stats.count + final_stats.repeat + non_customer.count`（严格） | 构造即保证，见 [`FlowAggregator::onEnter()`](src/flowaggregator.cpp:124) |
| 各维度分桶之和 = 所属块 count；`not_measured` 单独一格不并入 `unknown` | 无检测数据时全部落 `not_measured`，见 [`orderedBucketRows()`](src/flowaggregator.cpp:400) |
| 驻留时长按**进店时间**归时段 | [`FlowAggregator::onLeave()`](src/flowaggregator.cpp:196) |
| 重置（当天清零）用普通全 0 报文表达 | 无需特殊分支，revision 取大自动覆盖 |

---

## 5. 验证

### 5.1 一键脚本（推荐）

```bash
python3 ../verify_dongqia_uplink.py --all
```

包含三部分：

1. **C++ 自检（真实代码路径）**：运行 `CameraViewer --selftest-uplink`，
   注入合成事件，校验协议不变量，并验证「精准识别某人的上车/下车」：
   - `SUMMARY_V1=V1 上车1次/下车1次 …`
   - `PE|visitor=1|type=boarding|dir=B->A …`
   - `PE|visitor=1|type=alighting|dir=A->B …`
2. **本地回环**：脚本内起模拟服务端，按协议实现接收规则，真发 HTTP，验证
   首件入库 / revision 取大 / 整件替换 / 幂等 / 丢弃迟到件。
3. **真实平台**：登录 `footfall.dongqia.com` 并探测上行端点可达性。

### 5.2 只看自检

```bash
QT_QPA_PLATFORM=offscreen ./build-dongqia/CameraViewer --selftest-uplink
# 末行输出 SELFTEST_RESULT=PASS
```

### 5.3 精准识别某人的上车/下车

运行时，每个人的上下车事件都会：

- 写入 `visitor_log/visitor_events.csv`（`BOARDING` / `ALIGHTING`，含时间与方向）；
- 通过 `FlowAggregator::personTracked` 打印其**完整时间线**：
  `V3 上车2次/下车2次 | 进入4/离开4 | 首次 … | 最近 … | 累计驻留 …s`；
- 以 `person_flow_event` 报文上行（含 `visitor_id`、方向 `B->A`/`A->B`、
  `enter_ts_s`/`exit_ts_s`、`dwell_seconds`、`reid_similarity`、抓拍 Base64）。

方向定义（对齐 [`ZoneMonitor`](src/zonemonitor.h:17)）：通道顶端 A、底端 B，
**B→A = 上车**，**A→B = 下车**；跨帧确认 `confirmFrames=3` 防抖。

### 5.4 其它调试/自检开关

| 开关 | 说明 |
|------|------|
| `--selftest-uplink` | 上行协议与单人上下车自检（离线，不联网） |
| `--selftest-detect <图片>` | 用一张真实图像跑 检测→跟踪→ReID，用于复现/定位问题 |
| `--autostart` | 启动后自动点击「启动采集」（便于无头 `offscreen` 复现） |

无头复现示例：

```bash
QT_QPA_PLATFORM=offscreen gdb -batch -ex run -ex "bt 40" \
  --args ./build-dongqia/CameraViewer --autostart
```

---

## 6. 崩溃修复（本次发现并修复）

运行中曾出现 OpenCV 异常并直接 abort：

```
cv::Exception ... (-209:Sizes of input arguments do not match) ... in 'arithm_op'
#13 MainWindow::drawActiveIdOnCvMat(...) at mainwindow.cpp:631 -> cv::addWeighted(...)
```

**根因**：跟踪框越界/退化时（例如卡尔曼预测出框产生负坐标，实测
`bbox={x=1280,y=-28,w=83,h=10}`），标签背景矩形与画面求交后为空，
`cv::Mat roi = frame(bgRect)` 得到 0×0 空 ROI，随后
`cv::addWeighted(overlay, 0.6, roi, 0.4, 0, roi)` 在空矩阵上抛异常。

**修复**（仅改副本）：[`drawActiveIdOnCvMat()`](src/mainwindow.cpp:605) 增加越界防御——
跟踪框与画面无交集时跳过标签绘制；背景矩形为空或 ROI 为空时跳过背景混合；
[`OfflineDetectForm`](src/offlinedetectform.cpp:293) 同样处理。

> 说明：该缺陷位于与原程序共用的代码路径，原程序同样存在；按你的要求
> **未改动原程序**，仅在副本中修复。
> 验证：`--autostart` 无头连续运行 60s（期间 35 帧检测到人并绘制标签）未再出现异常。

---

## 7. 删除历史上下车记录（新增功能）

### 7.1 界面入口
主界面工具条新增按钮 **「删除历史记录」**（位于「清除区域」右侧）。
点击后弹窗确认，可选择：

- **删除记录+抓拍图片**：删除记录文件，并递归删除 ReID 抓拍目录 `person_*`
  与访客帧目录 `visitor_V*`；
- **仅删除记录**：只删记录，保留图片；
- **取消**。

删除内容：

| 类别 | 内容 |
|------|------|
| 内存 | 访客表 `m_visitors`、track↔visitor 映射、淡出标签、上车/下车计数、`ZoneMonitor` 状态、`FlowAggregator` 时间线与聚合 |
| 界面 | 访客表格清空，计数标签归零 |
| 磁盘记录 | `visitor_log/visitor_events.csv`、`visitor_log/visitor_records.json` |
| 抓拍（可选） | `save_images/reid/person_*`、`save_images/<date>/visitor_V*` |

> 不会删除待上行的离线积压 `uplink_spool/`（那是尚未送达平台的报文，删了会丢数据）。

核心实现：`MainWindow::clearHistoryRecords(bool deleteImages)`
（见 [`mainwindow.cpp`](src/mainwindow.cpp:337)），按钮处理 `onClearHistory()`
（见 [`mainwindow.cpp`](src/mainwindow.cpp:302)）。

### 7.2 命令行（无头/自动化）
```bash
./build-dongqia/CameraViewer --clear-history          # 仅删除记录
./build-dongqia/CameraViewer --clear-history-images   # 记录 + 抓拍图片
# 输出：CLEAR_HISTORY_RESULT=已删除记录文件 N 个，抓拍目录 M 个。
```

### 7.3 已验证
- 记录+图片模式：`记录文件=2, 抓拍目录=2`，`person_*` / `visitor_V*` 被删，同目录无关图片 `auto_keep.jpg` 保留；
- 仅记录模式：`记录文件=2, 抓拍目录=0`，图片全部保留；
- 删除后 `visitor_events.csv` 不再包含任何 `BOARDING/ALIGHTING` 历史行，
  `visitor_records.json` 的 `visitors` 为空数组。

---

## 8. 验证结论与待办

- ✅ 账号 `dongqia01 / 123456` 登录**成功**（`POST /api/iot-auth/auth/token`，返回 JWT）。
- ✅ 设备端报文生成、传输语义、单人上下车识别：**全部通过**（16/16）。
- ⚠️ 文档约定的接收端点 `POST /api/passenger-flow/interval-aggregate` 目前返回
  `404 No static resource …`，即**服务端尚未部署该接收端点**（平台当前仅有
  `iot-auth`、`iot-dataIngestion`、`iot-system` 等已挂载接口）。

**待办 / 建议**：请与动恰确认设备上报的真实接收地址与路径（协议 0 节说明二者均可由设备侧配置），
拿到后只需修改 `config.ini` 的 `BaseUrl` / `IntervalPath` / `PersonPath`，无需改代码即可联调。
服务端接收逻辑可直接参照协议 0 节「接入清单」与第 6 章「常见误用一览」实现。
