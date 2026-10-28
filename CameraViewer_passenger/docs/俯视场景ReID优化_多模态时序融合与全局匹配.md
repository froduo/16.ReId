# 俯视场景 ReID 优化：多模态时序融合 + 动态加权 + 匈牙利全局匹配

> 背景：俯视/鱼眼场景下人体被压缩为「头顶 + 肩部」投影，外观区分度天然低于正面视角。
> 本次针对三点进行优化：**多帧时序融合**、**更多特征维度**、**更复杂的匹配策略
> （动态加权 + 匈牙利全局匹配）**。

---

## 一、更多特征维度（多模态并行提取）

[`ReIDExtractor`](CameraViewer/src/reidextractor.h:44) 单帧特征由三类分量拼接，
其中**身形分量由 2 维扩展为 6 维多模态几何/体积代理**
（[`computeShapeVec()`](CameraViewer/src/reidextractor.cpp:119)）：

| 维度 | 含义 | 作用 |
|---|---|---|
| 0 `w/h` | 长宽比 | 体态胖瘦 |
| 1 `h/H` | 归一化高度 | 远近/身高 |
| 2 `w/W` | 归一化宽度 | 体宽 |
| 3 `w·h/(W·H)` | 面积占比 | 体积代理 |
| 4 前景填充率 | 前景像素 / 外接框面积 | 紧凑度/体积代理（背景扣除后计算） |
| 5 肩宽比 | 上方 40% 行内最大连续前景宽度 / 框高 | **肩宽**（俯视下最稳定的人体尺度） |

- 颜色：上/下半身 HSV 直方图（各 960 箱）；
- 纹理：上/下半身梯度方向直方图（各 18 箱）；
- 新增：分量级评分 API [`compareComponents()`](CameraViewer/src/reidextractor.h:95)，
  已内置 **180° 旋转不变**（上车/下车同人旋转约 180°，逐分量取更优）。

> 特征维度由 1958 → 1962，旧模板特征会在
> [`PassengerBook::load()`](CameraViewer/src/passengerbook.cpp:265) 中按维度不兼容自动丢弃（已有日志）。

---

## 二、多帧时序融合（EMA + 缓存）

用**指数滑动平均**替代等权平均：`fused = α·cur + (1-α)·prev`，越新的帧权重越高，
既抑制偶发噪声/误检帧，又更贴近目标“最近状态”。

- 工具：[`ReIDExtractor::temporalFuse()`](CameraViewer/src/reidextractor.h:80)（α 默认 0.4）；
- 缓存：[`TemporalFeatureCache`](CameraViewer/src/reidextractor.h:186)（按 key 维护 O(1) 融合特征）；
- 追踪器：[`PersonTracker::computeAverageReid()`](CameraViewer/src/persontracker.cpp:131)
  改为 EMA 融合，历史帧数 5 → 10；
- 台账：[`BoardingRecord::fused`](CameraViewer/src/passengerbook.h:33) 缓存时序融合模板，
  并提供在线细化 [`addBoardingFeature()`](CameraViewer/src/passengerbook.h:96)（EMA 更新，限长 30）；
- 比对：[`IdentityMatchWorker`](CameraViewer/src/identitymatchworker.cpp:78) 对每个探测的多帧序列做 EMA
  后再参与匹配，模板同样使用融合特征。

---

## 三、动态加权 + 匈牙利全局匹配

### 3.1 动态加权（动态模态权重）

对每个探测，先计算其在各候选模板上的**分量得分**（颜色/纹理/身形），
再按各分量在本批候选间的**区分度（极差）**自适应放大权重：

```
w_c = base_c · (0.25 + spread_c)      spread_c = max_j(score_cj) − min_j(score_cj)
```

区分度高的分量“说话”，全体都高分/低分的分量被抑制——从而在不同光照/着装条件下
自动偏向最有效的模态。基础权重仍可在 [`AppConfig`](CameraViewer/src/app_config.h:77)
中配置（`reidWeightColor/Texture/Shape`）。

### 3.2 匈牙利全局匹配

[`hungarian.h`](CameraViewer/src/hungarian.h:1) 实现 O(n³) 的 KM 指派算法
（[`hungarianAssignMax()`](CameraViewer/src/hungarian.h:82) 最大化总相似度）。
当**一批有多个下车探测**时，构建 `探测 × 模板` 相似度矩阵并做**全局最优指派**，
替代逐个贪心（贪心会因“先到先得”互相抢占模板，产生次优配对）。

### 3.3 精修

对全局指派的配对，再用「探测序列 × 模板序列」的 best-avg 相似度与融合相似度加权：

```
final = 0.55 · fusedSim + 0.45 · bestAvg
```

最终 `final >= threshold` 且被指派 → `matched`。

---

## 四、批量攒批流程（主链路）

```
ZoneMonitor 门线跨越 → MainWindow alightingEvent
        │  入队 jobId + PendingAlight，启动 300ms 攒批定时器
        ▼
 flushAlightBatch()  ── 组装本批探测(probes) + 在车模板(onboardTemplateBundle)
        │
        ▼
 IdentityMatchWorker::submitBatch()  （后台线程，串行）
        ├─ 时序融合（探测/模板）
        ├─ 分量评分 → 动态加权 → 相似度矩阵
        ├─ 匈牙利全局指派
        └─ best-avg 精修 → 逐探测 emit finished(jobId, bestId, sim, matched)
        ▼
 MainWindow 收到 finished → applyAlightingMatch()（命中/回退/异常结算）
```

相关实现：
- 攒批：[`MainWindow::flushAlightBatch()`](CameraViewer/src/mainwindow.cpp:527)
  （窗口常量 `ALIGHT_BATCH_WINDOW_MS = 300`）；
- 批量匹配：[`IdentityMatchWorker::runBatch()`](CameraViewer/src/identitymatchworker.cpp:78)；
- 模板导出：[`PassengerBook::onboardTemplateBundle()`](CameraViewer/src/passengerbook.cpp:139)。

> 单探测场景等价于 1×M 指派，退化即为 argmax，行为与旧版一致（向后兼容）。

---

## 五、构建与验证

```sh
cd build-main-debug
/home/froduo/Qt/6.2.4/gcc_64/bin/qmake ../CameraViewer/CameraViewer.pro -spec linux-g++ CONFIG+=debug
make -j4
```

- 主程序 [`CameraViewer`](CameraViewer/CameraViewer.pro:1) 编译通过；
- 命令行复核工具 `sh build_clip_check.sh` 编译通过。

运行时可观察日志：

- `[IdentityMatch] jobId=... 批量P=.. 模板T=.. | 全局指派=P.. best=P..(sim=..→精修..) ... top候选: ...`
  → 核对全局指派是否合理、精修前后分数变化；
- `[PassengerBook] P.. 模板在线细化: frames=.. (EMA 融合已更新)`。

## 六、可调项

| 项 | 位置 | 默认 |
|---|---|---|
| EMA 系数 α | [`temporalFuse()`](CameraViewer/src/reidextractor.h:80) 调用处 / [`TemporalFeatureCache`](CameraViewer/src/reidextractor.h:186) | 0.4 |
| 基础分量权重 | [`AppConfig`](CameraViewer/src/app_config.h:77) | 0.30/0.35/0.35 |
| 动态加权开关 | [`setDynamicWeightingEnabled()`](CameraViewer/src/identitymatchworker.h:56) | 开 |
| 攒批窗口 | [`ALIGHT_BATCH_WINDOW_MS`](CameraViewer/src/mainwindow.h:200) | 300ms |
| 判定阈值 | [`AppConfig::reidMatchThreshold`](CameraViewer/src/app_config.h:74) | 0.65 |

---

## 七、内存安全修复（`free(): invalid next size (fast)`）

用 ASan 复现并定位到两处内存缺陷，均已修复：

### 7.1 `computeShapeVec()` 行列索引写反（越界写）
多维身形向量 `s` 是 **1×6 的行向量**，索引应为 `at<float>(0, i)`（行0、第 i 列），
但实现里误写成 `at<float>(i, 0)`（第 i 行）——当 `i>=1` 时越界写入，
破坏了相邻堆块的头部，最终在 `free()` 时抛出
`free(): invalid next size (fast)`（正是离线检测报错的根因）。
见 [`computeShapeVec()`](CameraViewer/src/reidextractor.cpp:124)。

### 7.2 `regions()` 返回值被误绑定引用（use-after-free）
`RegionManager::regions()` 原本按值返回 `QVector<DetectionRegion>`；
若调用方写 `const auto &r0 = rm.regions().first();`，会把引用绑定到**临时对象**，
语句结束后临时对象析构 → 悬垂引用（ASan: heap-use-after-free）。
两处修复：
1. 把 `regions()` 改为返回 **const 引用**（[`RegionManager::regions()`](CameraViewer/src/regionmanager.h:32)），
   从根上消除该模式；
2. 修正 [`offline_clip_check.cpp`](offline_clip_check.cpp:72) 中按值接收的写法。

### 7.3 验证
新增 ASan 驱动测试 [`tools/doorline_asan_test.cpp`](tools/doorline_asan_test.cpp)，
模拟一个人自顶向下/自底向上跨越虚拟门线并丢失轨迹，覆盖
`ZoneMonitor` 采样/跨越/结算 + `PersonTracker` + `ReIDExtractor` 融合路径：

```sh
QT=/home/froduo/Qt/6.2.4/gcc_64
$QT/libexec/moc CameraViewer/src/zonemonitor.h  -o /tmp/moc_zm.cpp -I CameraViewer/src
$QT/libexec/moc CameraViewer/src/regionmanager.h -o /tmp/moc_rm.cpp -I CameraViewer/src
g++ -std=c++17 -g -O1 -fsanitize=address -fno-omit-frame-pointer \
    tools/doorline_asan_test.cpp CameraViewer/src/{zonemonitor,regionmanager,persontracker,reidextractor,reiddeep,backgroundmodel}.cpp \
    /tmp/moc_zm.cpp /tmp/moc_rm.cpp -I CameraViewer/src -I $QT/include -I $QT/include/QtCore -I $QT/include/QtGui \
    -I /usr/local/include/opencv4 -L $QT/lib -lQt6Core -lQt6Gui \
    -lopencv_core -lopencv_imgproc -lopencv_video -lopencv_videoio -lopencv_highgui -lopencv_imgcodecs -lpthread \
    -o /tmp/door_asan
ASAN_OPTIONS=detect_leaks=0 /tmp/door_asan     # 期望：无 ASan 报错，board=1 alight=1
```
