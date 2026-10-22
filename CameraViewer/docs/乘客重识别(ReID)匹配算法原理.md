# 乘客重识别（ReID）匹配算法原理

> 本文整理 CameraViewer 中「同一个乘客」从出现、穿越门线到最终身份确认全流程使用的
> 重识别（ReID）匹配算法。涉及特征提取、相似度度量、以及跟踪器 / 门线 / 台账 /
> 目录全图四个层面的匹配策略。

---

## 1. 总体流程

```
相机帧 → YOLO 检测 → PersonTracker 跟踪(含 ReID 复联)
                       │
                       ├── 帧级外观特征提取 ReIDExtractor（手工 1962 维 / 深度 OSNet 512 维）
                       │
                       ├── ZoneMonitor 门线穿越判定 → 上/下车事件
                       │
                       └── 身份确认（两套链路）：
                             ① 旧链路 PassengerBook + IdentityMatchWorker（台账 + 匈牙利全局匹配）
                             ② 新链路 PersonFolder + FolderMatchWorker（按 person_N 分目录 + 全图比对）
```

同一位乘客可能在画面中多次出现、跨越不同 track，系统通过以下机制把它“认成同一个人”：

| 层 | 作用 | 关键匹配方式 |
|----|------|--------------|
| 跟踪器 [`PersonTracker`](src/persontracker.cpp:385) | 帧间保持同一 track，短暂丢失后复联 | IoU → 卡尔曼预测 IoU → ReID 特征 + 丢失画廊 |
| 门线 [`ZoneMonitor`](src/zonemonitor.cpp:184) | 新 track 关联回历史访客 | ReID 与历史访客特征比对（重入） |
| 台账 [`PassengerBook`](src/passengerbook.cpp:74) + [`IdentityMatchWorker`](src/identitymatchworker.cpp:85) | 下车时确定是哪位上车的乘客 | EMA 融合 + 动态加权 + 匈牙利全局匹配 + best-avg 精修 |
| 目录全图 [`FolderMatchWorker`](src/foldermatchworker.cpp:96) | 上车目录 ↔ 下车目录归档后比对 | 全目录 OSNet 512 维 best-avg 余弦相似度 |
| 换号续接 [`MainWindow::savePersonAppearanceFrames()`](src/mainwindow.cpp:2360) | 新乘客出现时决定“换号/续接” | 时间相邻 + ReID 相似度 |

---

## 2. 特征提取

特征提取统一入口：[`ReIDExtractor::extract()`](src/reidextractor.cpp:168)。

### 2.1 深度后端（OSNet，可选）

- 模型：OSNet x1.0（MSMT17 预训练 / 现场微调），ONNX 推理，见 [`DeepReID`](src/reiddeep.h:22)。
- 输入：人体框裁剪（head 检测模式下可用 [`reidHeadBoxExpand`](src/app_config.h:180) 扩展为人体比例）。
- 输出：**512 维 L2 归一化全局特征**，相似度用余弦（归一化特征点积），映射到 [0,1]。
- 后端优先级：TensorRT(GPU) → OpenCV-CUDA → CPU。
- 深度特征区分力显著强于手工特征，默认关闭 [`reidUseDeepReid`](src/app_config.h:166)，现场微调并标定阈值后启用。

### 2.2 手工多模态特征（默认）

当深度后端未启用或推理失败时，回退到手工特征，维度 `TOTAL_DIM = 1962`，由三部分组成：

#### ① 颜色分量（1920 维）
- 对 ROI 转 HSV，计算三维直方图，分箱 `H=15, S=8, V=8`，每半身 960 箱。
- 上、下半身各一份 → `COLOR_DIM = 1920`。
- L2 归一化。详见 [`computeColorHistogram()`](src/reidextractor.cpp:37)。

#### ② 纹理分量（36 维）
- 灰度图 Sobel 求梯度幅值与方向，方向取模 180°（消除正负向），按幅值加权统计方向直方图。
- 分箱 `TEX_BINS = 18`，上、下半身各一份 → `TEX_DIM = 36`。
- L2 归一化。详见 [`computeTextureHistogram()`](src/reidextractor.cpp:74)。

#### ③ 身形分量（6 维）
俯视鱼眼场景下人体被压缩为“头顶 + 肩部”，颜色/纹理区分度下降，故补充 6 维几何/体积代理，
见 [`computeShapeVec()`](src/reidextractor.cpp:125)：

| 维度 | 含义 |
|------|------|
| 0 | 长宽比 w/h |
| 1 | 归一化高度 h/H |
| 2 | 归一化宽度 w/W |
| 3 | 面积占比 (w·h)/(W·H) |
| 4 | 前景填充率 fg/(w·h)（体积代理） |
| 5 | 肩部宽度比（肩宽/身高） |

### 2.3 特征预处理细节

1. **背景扣除**：绑定背景模型后，仅统计人体前景像素（掩码），并把人体框按前景连通域收紧
   （面积比 ≥ 0.20 才采纳，见 [`reidextractor.cpp`](src/reidextractor.cpp:205)）。
2. **上/下半身拆分**：颜色与纹理按 H/2 拆分为上下半身，便于后续做旋转不变。
3. **拼接**：`[颜色上|颜色下|纹理上|纹理下|身形]` 拼接为 1962 维。

---

## 3. 相似度计算

### 3.1 深度特征相似度
[`DeepReID::compare()`](src/reiddeep.h:39) 对两个 L2 归一化的 512 维特征做点积（余弦），
映射到 [0,1]。

### 3.2 手工特征相似度（分量融合）
入口 [`ReIDExtractor::compare()`](src/reidextractor.cpp:441)，分三分量计算再加权融合：

- **颜色**：Bhattacharyya 距离，上/下半身权重 0.55 / 0.45，`sim = 1 - d`。
- **纹理**：Bhattacharyya 距离，上/下半身权重 0.5 / 0.5，`sim = 1 - d`。
- **身形**：逐维相对差异 `1 - |x-y|/max(|x|,|y|)`，维度权重
  `[0.28, 0.18, 0.16, 0.12, 0.08, 0.18]`。
- **融合**：`fused = (wc·color + wt·texture + ws·shape) / (wc+wt+ws)`。

默认分量权重（[`reidextractor.h`](src/reidextractor.h:147)）：颜色 0.30 / 纹理 0.35 / 身形 0.35；
运行期可由配置 [`reidWeightColor/Texture/Shape`](src/app_config.h:145) 覆盖。

### 3.3 180° 旋转不变
上车与下车时同一人约旋转 180°（上下半身交换）。[`compareComponents()`](src/reidextractor.cpp:374)
会交换颜色/纹理的上下半身块构造“旋转版本”，与原始方向**取较大者**作为该分量得分，从而对朝向鲁棒。

---

## 4. 各层匹配算法

### 4.1 跟踪器层：帧间保持 + 短暂丢失复联

[`PersonTracker::multiStageMatch()`](src/persontracker.cpp:166) 使用**三阶段贪心匹配**：

| 阶段 | 方法 | 阈值 / 分数 |
|------|------|-------------|
| 1 | IoU 匹配 | `iou >= m_iouThreshold`，score = IoU |
| 2 | 卡尔曼预测框 IoU | 阈值 = `0.5 × iouThreshold`，score = `0.8 × predIoU` |
| 3 | ReID 特征匹配 | track 历史平均特征 vs 检测特征，`sim >= trackerReidThreshold`，score = `0.6 × sim` |

- 所有候选按分数降序贪心指派（每个 track / 检测只匹配一次）。
- track 的 ReID 特征使用**最近 10 帧的 EMA 融合**（[`computeAverageReid()`](src/persontracker.cpp:131)，α=0.4）。

**丢失画廊复联**：track 连续丢失超过 `maxMissed=30` 帧后进入画廊（保留 `trackerGalleryMaxAgeMs=10s`）。
新检测出现时，[`tryReidFromGallery()`](src/persontracker.cpp:342) 用 ReID 相似度 + 位置距离约束
（`dist < 10 × max(bbox.w, bbox.h)`）恢复原 trackId，避免 ID 切换。

### 4.2 门线层：新 track 关联回历史访客

[`ZoneMonitor::matchOrCreateVisitor()`](src/zonemonitor.cpp:184) 在新 track 首次出现时：

1. 若 track 已映射，直接返回。
2. 提取特征，与历史访客特征 `m_visitorFeatures` 逐一比对：
   - 跳过“最后穿越为上车（在车内）”的访客；
   - 跳过当前正在接近窗口内（Capturing）的访客；
   - 取 `sim >= m_reidMatchThreshold` 的最高者作为 **ReID 重入**。
3. 命中则复用原 visitorId 并更新特征；否则创建新 visitor。

### 4.3 台账层：下车身份判定（旧链路，`kLegacyPipelineEnabled`）

#### 上车登记
[`PassengerBook::registerBoarding()`](src/passengerbook.cpp:38) 保存上车特征序列，并用
EMA（α=0.4）生成融合模板 `fused`。运行中还可通过 [`addBoardingFeature()`](src/passengerbook.cpp:160)
在线细化模板（保留最近 30 帧）。

#### 下车比对（批量 + 匈牙利全局匹配）
多个下车事件先攒批（窗口 `ALIGHT_BATCH_WINDOW_MS = 300ms`），由
[`IdentityMatchWorker::runBatch()`](src/identitymatchworker.cpp:85) 一次处理：

1. **时序融合**：探测（下车）与模板（在车）均做 EMA 融合（α=0.4）。
2. **相似度矩阵**（探测 × 模板）：
   - 深度后端：直接余弦相似度；
   - 手工后端：逐分量评分 + **动态加权**——某分量在候选间的“极差”越大，权重越高：
     `w = base × (0.25 + spread)`，使有区分度的模态说话、抑制无区分度模态。
3. **匈牙利全局最优指派**（[`hungarianAssignMax()`](src/hungarian.h:80)）：把“相似度最大化”
   转化为“代价最小化”（`cost = 1 - sim`）求解，O(n³)，避免贪心互相抢占。
4. **best-avg 精修**：仅对被指派的配对，用探测序列 × 模板序列的 best-avg 相似度精修：
   `finalSim = 0.55 × 全局Sim + 0.45 × bestAvg`。
5. **判定**：`matched = 被指派 && finalSim >= threshold`。

结算见 [`MainWindow::applyAlightingMatch()`](src/mainwindow.cpp:547)：命中则移除台账乘客；
未达阈值但跟踪身份指向在车乘客则回退；否则记为异常下车。

> `best-avg`：对探测每一帧，取与模板序列所有帧的最大相似度，再对这些最大值求平均。
> 实现见 [`bestAvgSimilarity()`](src/identitymatchworker.cpp:41) 与
> [`sequenceSimilarity()`](src/passengerbook.cpp:209)。

### 4.4 目录全图层：上车目录 ↔ 下车目录（新链路，默认）

新链路用 `person_N` 目录代替实时台账：乘客一出现即分配 `person_N` 并持续保存裁剪原图与整帧
（[`savePersonAppearanceFrames()`](src/mainwindow.cpp:2360)，每 `personSaveEveryN` 帧存 1 张）。

1. **上车**：登记 `person_N` 方向“上车” + 目录，加入候选集 `m_boardFolders`。
2. **下车**：生成候选（时间更早、未匹配、且停留 ≥ `MIN_DWELL_MS=30s` 的上车目录），提交
   [`FolderMatchWorker`](src/foldermatchworker.cpp:96)。
3. **目录比对**：
   - 每个目录最多读取 `personFolderMaxImages=200` 张图；
   - 每张图用 OSNet 提取 512 维特征；
   - **目录级 best-avg 余弦相似度**：下车每张图取与上车目录所有图的最大余弦，再平均。
   - 记录最高分 `bestSim` 与次高分 `secondSim`，`margin = bestSim - secondSim` 用于衡量置信度。
4. **判定**：`matched = bestSim >= reidFolderMatchThreshold`；命中后由
   [`markMatched()`](src/personfolder.cpp:330) 写双方 `match.txt`（方向 + 匹配对象 + 相似度）。

### 4.5 换号 / 续接层：新乘客出现时决定是否换号

> 本次改造后的规则：**取消“穿越完成 → 3 帧后换号”**，改为**“遇到新乘客出现时才换号”**，
> 并且“时间相邻 + ReID 相似度高”的新乘客**不换号**（续接原编号）。

逻辑见 [`MainWindow::savePersonAppearanceFrames()`](src/mainwindow.cpp:2360) 与
[`PersonFolder::reusePerson()`](src/personfolder.cpp:153)：

1. 新 track（尚无 person 编号）出现时，提取其 ReID 特征。
2. 在“最近消失的 person”缓存（[`m_personReidFeature`](src/mainwindow.h:256) /
   [`m_personLastSeenMs`](src/mainwindow.h:257)）中查找满足以下条件的候选：
   - **时间相邻**：`now - lastSeen <= kPersonRotateGapMs(5s)`；
   - **不在当前画面中**（跳过仍活跃的 person）；
   - **ReID 相似度高**：`sim >= effectiveReidThreshold()`。
3. 命中 → [`reusePerson()`](src/personfolder.cpp:153) 续接原 `person_N`（沿用目录与图片序号，不换号）。
4. 否则 → [`ensurePerson()`](src/personfolder.cpp:126) 分配新 `person_N`（新乘客出现，换号）。
5. 缓存按时间窗口清理：超过 5s 未再出现且不在画面中的 person 特征被移除。

---

## 5. 关键参数汇总

| 参数 | 默认值 | 说明 | 位置 |
|------|--------|------|------|
| `reidMatchThreshold` | 0.65 | 手工 ReID 通用匹配阈值 | [`app_config.h`](src/app_config.h:142) |
| `reidDeepMatchThreshold` | 0.50 | 深度特征余弦判定阈值 | [`app_config.h`](src/app_config.h:171) |
| `reidFolderMatchThreshold` | 0.60 | 目录级 best-avg 判定阈值 | [`app_config.h`](src/app_config.h:174) |
| `trackerReidThreshold` | 0.62 | 跟踪器内部 ReID 复联阈值 | [`app_config.h`](src/app_config.h:123) |
| `trackerGalleryMaxAgeMs` | 10000 | 丢失画廊保留时长(ms) | [`app_config.h`](src/app_config.h:124) |
| `reidWeightColor/Texture/Shape` | 0.30/0.40/0.30 | 手工分量权重 | [`app_config.h`](src/app_config.h:145) |
| `personFolderMaxImages` | 200 | 单目录参与比对最大图片数 | [`app_config.h`](src/app_config.h:177) |
| `personMinTrackFrames` | 3 | 新乘客建档预热帧数 | [`app_config.h`](src/app_config.h:176) |
| `personSaveEveryN` | 3 | 每 N 帧保存 1 张裁剪图 | [`app_config.h`](src/app_config.h:175) |
| `ALIGHT_BATCH_WINDOW_MS` | 300 | 下车比对攒批窗口(ms) | [`mainwindow.h`](src/mainwindow.h:223) |
| `MIN_DWELL_MS` | 30000 | 上→下车最小停留时长(ms) | [`mainwindow.cpp`](src/mainwindow.cpp:440) |
| `kPersonRotateGapMs` | 5000 | 新乘客续接判定的时间相邻窗口(ms) | [`mainwindow.h`](src/mainwindow.h:258) |
| `REID_HISTORY_SIZE` | 10 | 跟踪器 ReID 历史帧数 | [`persontracker.h`](src/persontracker.h:72) |

**生效阈值说明**：[`effectiveReidThreshold()`](src/mainwindow.cpp:2527) 在深度后端就绪且启用时返回
`reidDeepMatchThreshold`，否则返回 `reidMatchThreshold`，用于门线 ReID 重入与换号续接判定。

---

## 6. 相似度计算小结

| 场景 | 相似度定义 |
|------|-----------|
| 深度特征（单帧/单图） | 512 维余弦（点积） |
| 手工特征（单帧） | 颜色/纹理/身形加权融合，含 180° 旋转不变 |
| 序列 × 序列 | best-avg：探测每帧取与模板序列最大相似度再平均 |
| 时序融合 | EMA（α=0.4），新帧权重更高；颜色/纹理块分别 L2 归一化，深度整体 L2 归一化 |
| 全局多对多 | 相似度矩阵 + 匈牙利最大指派（探测=行、模板=列） |
| 目录 × 目录 | 全目录 OSNet best-avg + margin(最高分-次高分)置信度 |
