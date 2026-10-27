# 待办任务清单

> 记录后续需要完成的任务，方便按优先级推进。

## 已完成

- [x] 修复 `ZoneMonitor::matchOrCreateVisitor` visitor 合并 bug（EMA 特征融合 + 自由访客 30s 时间窗）
- [x] 修复 `MainWindow::matchOrCreateVisitor` 特征覆盖（EMA 融合）
- [x] person_8 目录离线检测 + 聚类分组分析（确定混入多个乘客）
- [x] 导出 ReID 种子数据集 `passenger_reid_dataset/`（passenger_001/002/003）
- [x] 新检测模型 `best.pt`（yolov8m, nc=1）转 engine 并导入项目
- [x] 写「person_N 初步匹配」脚本 `tools/match_person_folders.py`（目录级 OSNet 均值特征 + 上车↔下车贪心一对一匹配，输出 `match_result.json` 供人工确认）
- [x] 运行初步匹配：两个 session（164420/170104）共 121 person → 58 对匹配（avg sim=0.916，51 对高置信≥0.85，7 对低置信需人工复核），结果存 `match_result.json`
- [x] 写「按高置信匹配合并导出」脚本 `tools/export_matched_identities.py`（并查集合并 sim≥0.85 的身份，增量导出到 passenger_reid_dataset）
- [x] 导出数据集：passenger_reid_dataset 现共 **63 身份 / 7947 图**（新增 60 身份，跳过 10 个图片<10张的 person）
- [x] 生成人工核对 HTML 对比页 `match_review.html`（58 对，绿色=高置信/红色=需复核）+ 脚本 `tools/render_match_review.py`
- [x] 建立人工审核记录 `review.json`（已确认 14 对同一人，待审核 44 对）
- [x] 写「人工配对」交互式工具 `tools/render_pairing_tool.py` → 生成 `pairing_tool.html`（左=上车/右=下车，点击配对，导出 JSON 结果）
- [x] 人工配对完成：21 对（与自动一致 13 对、人工纠正 8 对），存 `manual_pairs.json`
- [x] 写 `tools/rebuild_dataset_manual.py` 并按人工配对重建数据集：清空旧的自动匹配导出，现 **90 身份 / 7931 图**（21 合并身份 + 66 独立 + 保留 001~003）
- [x] OSNet 微调训练完成（60 epochs，90 身份 train72/val18，best gap=0.310）：导出 `osnet_x1_0_finetuned.onnx`（input 1×3×256×128 → output 1×512×16×8）
- [x] **新增「第三/四状态」判定（上车折返/下车折返）**：`ZoneMonitor` 在门线附近逗留、上/下车穿越后未离开视野即反向穿越时，回滚计数并发射 `boardingInvalidEvent`/`alightingInvalidEvent`（不计入上下车总数）；`MainWindow`/`OfflineDetectForm` 回滚计数+日志；`PersonFolder::setInvalid` 把方向改写为「上车折返/下车折返」从而排除出上/下车配对（解决 person_150/151/190 类「上车穿越后又未离开视野即下车」视频）
- [x] 写 `tools/split_manual_pairs.py`（按错误配对清单把 manual_pairs 拆成 valid/invalid）+ `tools/wrong_pairs_20261023.txt`（已录入 13 对匹配错误）

## 未完成（待办）

- [ ] 【预留·不处理】「下车折返」比对时序撤销：下车事件触发时比对（攒批 `ALIGHT_BATCH_WINDOW_MS=300ms`）通常已执行（折返确认窗口 `m_lingeringSettleMs=8000ms`），`alightingInvalidEvent` 回滚时不撤销已发生的 `passengerBook.removePassenger` 与目录 `markMatched`；暂无样本视频，后续如需彻底保证「下车折返不产生配对」再处理（延迟比对 或 折返时恢复台账/撤销匹配）
- [ ] 三个 session 数据采集：session_20261023_091153 / session_20261023_121959 / session_20261023_132747（进行中：视频还在分类，方向字段陆续更新中）
- [ ] 用新模型对三个 session 预配对 + 人工确认 + 追加数据继续训练（工具已备好，流程见下）
  - 预配对：`match_person_folders.py --sessions <三个session绝对路径> --out match_result_20261023.json --threshold 0.60`（默认已用 osnet_best.pth）
  - 人工确认：`render_pairing_tool.py --sessions <三个session绝对路径> --prefill match_result_20261023.json --out pairing_tool_20261023.html`（预填自动配对，双击撤销/点击补配，导出 JSON 存 manual_pairs_20261023.json）
  - 追加数据：`append_pairs_dataset.py --manual manual_pairs_20261023.json --sessions <三个session绝对路径>`（增量追加，不清空既有 90 身份）
  - 继续训练：`finetune_osnet_dataset.py --root passenger_reid_dataset --weights osnet_best.pth --epochs 60 --p 8 --k 4`

- [x] **torchreid 微调训练脚本**（`tools/finetune_osnet_dataset.py` 已写好，逻辑验证通过：数据加载/权重加载/forward/backward 均正常，featuremaps 输出 512×16×8 与 C++ DeepReID 兼容）
  - ✅ `yolotest` 已装 GPU 版 torch（`torch==2.1.2+cu121`，`cuda=True`，RTX 2060 可用）
  - 等 `passenger_reid_dataset` 攒够 100~200 个乘客后直接跑：`python finetune_osnet_dataset.py --epochs 60 --p 8 --k 4`
- [ ] 梳理"人工确认同一人上下车"文档格式（CSV/JSON/Excel？字段：上车目录、下车目录、是否同一人）
- [ ] 写"按人工确认文档导出 torchreid 训练数据集"脚本
- [ ] 积累 OSNet 微调数据：100~200 个不同乘客，每人 10~30 张上下车裁剪（用 `tools/export_seed_dataset.py` 增量追加到 `passenger_reid_dataset/`）
- [ ] 检测模型加入负样本重训（解决 A 类"非乘客误识别"问题，负样本放 `negative_yolo/`）
- [ ] 【模型版本管理·后期加】新生成的 OSNet 模型带日期后缀，方便查看/追溯版本（本次 2026-10-27 训练先不改，后期实施）：
  - `finetune_osnet_dataset.py` 导出的 `osnet_best.pth` / `osnet_x1_0_finetuned.onnx` 改为 `osnet_best_YYYYMMDD.pth` / `osnet_x1_0_finetuned_YYYYMMDD.onnx`（保留历史，不覆盖）
  - `retrain_and_deploy.sh` / `deploy_finetuned_model.sh` 部署时同步带日期，`model/` 与 `build/model/` 目录保留历史副本
  - `config.ini` 的 `DeepModelPath` 指向最新带日期 onnx（或保持固定名 + 软链指向最新版本，兼容程序加载）
- [x] OSNet 微调后部署（2026-10-23）：复制 `osnet_x1_0_finetuned.onnx` → `model/` 与 `build/model/`；删旧 `osnet_x1_0_head_finetuned.engine`；更新 `config.ini` 的 `DeepModelPath=./model/osnet_x1_0_finetuned.onnx`；标定 `DeepMatchThreshold=0.60`（21 对人工配对 best-avg：同人 0.94±0.11 / 异人 0.26±0.12）

## 任务1 待验证（第三/四状态）
- [ ] 用新代码重新分类 person_150/151/190 等视频，确认被判定为「上车折返」而非「上车+下车」（程序需先重新编译：`cd build && make -j4`）
- [ ] 第四状态「下车折返」暂无样本视频，程序已预留（`alightingInvalidEvent` + `setInvalid(personId,false,...)`），后续找到类似视频再验证

## 任务2 待执行（manual_pairs 拆分 + 追加训练）
- [ ] 【2.2 待分析】person108 上车图片保存过少的原因——先记录，后期分析：疑似整段采样 `trackImages` 不足（轨迹断裂/采样窗口过短），排查 `ZoneMonitor::sampleCapture/startCapture/stopCapture` 与 `SAMPLE_INTERVAL`/`MAX_FEATURE_SAMPLES`
- [ ] 【2.1 待执行】等程序停止、完整版 `manual_pairs_20261023.json` 生成后：
  - 把「以及后面的都匹配错误」的那些对补齐到 `tools/wrong_pairs_20261023.txt`
  - 跑 `python tools/split_manual_pairs.py --manual manual_pairs_20261023.json --wrong tools/wrong_pairs_20261023.txt --valid-out manual_pairs_20261023_valid.json --invalid-out manual_pairs_20261023_invalid.json`
  - 用 `manual_pairs_20261023_valid.json` 跑 `append_pairs_dataset.py` 增量追加训练（不要用含错误配对的原文件）

## OSNet 微调训练优化方案（分析结论，2026-10-23，待实施）

> 针对 `tools/finetune_osnet_dataset.py`（`--epochs 60 --p 8 --k 4`，当前数据集 **173 身份 / 18475 图**）的训练逻辑分析。
> 运行环境：torch 2.1.2+cu121 / CUDA 12.1 / cuDNN 8.9 / RTX 2060 (6GB)。

### 当前训练配置

| 项目 | 当前配置 |
|------|---------|
| 模型 | torchreid `osnet_x1_0`（`loss="softmax"` 头，无显式 BNNeck），从 `osnet_best.pth` 去分类头加载 |
| 数据 | 173 身份 / 18475 图，train:val = 8:2（`split_dirs` 随机划分，**无固定种子**） |
| 采样 | `RandomIdentitySampler` P=8 × K=4 → batch=32 |
| 增强 | Resize(256,128) + 水平翻转 + ColorJitter + RandomErasing(0.3) |
| 损失 | CE(label_smoothing=0.1) + Batch-Hard Triplet(margin=0.3)，权重 1:1 |
| 优化器 | Adam `lr=3.5e-4` `wd=5e-4` + CosineAnnealingLR(T_max=60)，**无 warmup** |
| 数据加载 | `num_workers=0`，图片全部缓存进内存（`self.cache`） |

### 性能瓶颈（按影响排序）

1. **🔴 backbone 被重复计算 2 遍（最大问题）**：`train` 循环里 `logits = model(x)`（内部已跑一遍 featuremaps），紧接着又 `fm = model.featuremaps(x)` 再跑一遍 conv1~conv5，单步约浪费一半 GPU 算力。
2. **🟠 数据增强单线程（`num_workers=0`）**：`ToPILImage→Resize→ColorJitter→RandomErasing→ToTensor` 全在主进程串行执行，GPU 取数时空转；且 `self.cache` 存的是原始分辨率图（进程 RES 约 7.7GB），直接开 `num_workers>0` 会把 cache pickle 复制到每个 worker 导致内存成倍爆炸。
3. **🟡 未开混合精度（AMP）**：RTX 2060（图灵架构）有 fp16 Tensor Core，当前纯 fp32。
4. **🟢 其他零成本项**：未开 `cudnn.benchmark`、未用 `channels_last` 内存布局。

### 加速方案（不影响精度）

- [ ] **消除重复前向**（≈2× 加速，零精度影响）：`train` 循环改为只算一次 featuremaps 并复用（注意 triplet 用的就是「GAP 后、未过 fc」的向量，CE 用「过 fc 后」的向量，需保持该特征分工不变）：
  ```python
  fm = model.featuremaps(x)                    # 只算一次
  v_gap = model.global_avgpool(fm).flatten(1)  # 512 维
  v_fc  = model.fc(v_gap)                       # 用于 CE
  logits = model.classifier(v_fc)
  v_tri = F.normalize(v_gap, dim=1)             # 用于 triplet（与现状一致）
  l_ce, l_tri = ce(logits, y), batch_hard_triplet(v_tri, y)
  ```
- [ ] **开 AMP 混合精度**（≈1.3~2×，显存下降可反哺增大 batch）：
  ```python
  scaler = torch.amp.GradScaler('cuda')
  with torch.amp.autocast('cuda'):
      ...前向、算 loss...
  scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
  ```
  （如 triplet 距离计算出现 nan，可对该段保留 fp32。）
- [ ] **`torch.backends.cudnn.benchmark = True` + `channels_last` 内存布局**（零成本，固定输入尺寸下 conv 更快）：
  ```python
  model = model.to(memory_format=torch.channels_last)
  x = x.to(memory_format=torch.channels_last)
  ```
- [ ] **数据加载优化（二选一）**：
  - 方案 A（推荐）：`cache` 改存 resize 到 256×128 的 uint8 图（内存 7.7GB→约 1.8GB），`build_transform` 去掉 `Resize`，然后开 `num_workers=4` + `persistent_workers=True` + `pin_memory=True`。
  - 方案 B（保守）：保持 `num_workers=0`，把 `Resize` 在 `__init__` 建 cache 时一次性做完，`__getitem__` 只做翻转/色彩/擦除等轻操作。

### 提高模型精度方案

- [ ] **固定随机种子**（可复现，最基础）：`split_dirs`/`RandomIdentitySampler` 均用 `random` 且无种子，每次 train/val 划分与采样不同，结果不可比。加 `random.seed(0); np.random.seed(0); torch.manual_seed(0)`。
- [ ] **数据增强增强**（ReID 收益最直接，上车↔下车光照/视角变化大）：
  - `Resize((256,136)) + RandomCrop((256,128))`（保留长宽比、加位置扰动）
  - 增大 `ColorJitter`（尤其 brightness/contrast，光照鲁棒性关键）
  - 增加小角度 `RandomRotation`、`GaussianBlur`
  - `RandomErasing` 概率 0.3 → 0.5
- [ ] **加 BNNeck（统一 CE 与 triplet 特征空间）**：当前 CE 用「过 fc（含 BN）」特征、triplet 用「GAP 后未过 BN」特征，两监督信号作用在不同分布上互相牵制；加 `nn.BatchNorm1d(512)`，让 triplet 用 BN 后归一化特征、CE 用 BN 前特征。
- [ ] **损失函数增强**：加 Center Loss（λ≈5e-4~5e-3，提升类内紧凑性）；triplet 可试 soft-margin 或 margin 0.3→0.4；`loss = ce + tri` 加 `--triplet-weight` 权重超参。
- [ ] **训练策略**：加 warmup（前 5~10 epoch 线性升至峰值再进 Cosine）；换 AdamW（weight_decay 解耦）；fc 层加 `dropout=0.5` 缓解过拟合。
- [ ] **数据侧**：存在 min=12 张图/身份的低样本（PK 采样时 `random.choices` 重复采样会产生重复正样本），建议 <20 张的身份合并或补采。
- [ ] **评估/导出 TTA**：导出/推理用左右翻转 + 多尺度取平均，最终精度可进一步提升。
- [ ] **验证集改进**：val 仅约 34 身份（20%），`eval_similarity` 噪声较大，可增大 val 比例或分层采样并固定 val 集。

### 综合优先级

1. **先做（性价比最高、低风险）**：消除重复前向 → 开 AMP → `cudnn.benchmark`+`channels_last` → 固定随机种子。
2. **再做（精度）**：数据增强加 RandomCrop + 更强 ColorJitter + RandomErasing 概率提升 → 加 warmup + 换 AdamW → 加 BNNeck →（可选）Center Loss。
3. **长期**：补充低样本身份数据、评估/导出加 TTA。

---

## 2026-10-23 下午（任务1 离线分析 + 修复 / 任务2 匹配细节增强）

### 任务1：离线图像分析 + 程序优化
- [x] **离线分析 person_311**（OSNet 层次聚类验证）：414 张乘客图聚类为 **3 组**（组间相似度 0.448/0.470/0.558，均 <0.6 为不同人）——证实 3 个乘客（上车→下车→上车）被 tracker 串到同一 track=26 混存。
- [x] **离线分析 person_338**（OSNet 聚类验证）：141 张聚类 2 组但时间交错（组间 0.499，为折返姿态变化）——证实为同一人「上车折返」被误判为「上车+下车」。
- [x] **修复 person_338 折返误判**：`ZoneMonitor::settlePending` 离开门线视野时不再立即清除 `m_pendingByVisitor`，改由 `tryInvalidateReversal` 的 `m_lingeringSettleMs`(8s) 窗口超时 + `forgetVisitor`/`pruneStaleVisitors` 统一清理，使「上车后短暂远离门线再折返」能正确判为上车折返。
- [x] **修复 person_311 多人混存**：`MainWindow` 下车事件里，若该 track 已判为「上车」（`PersonFolder::isBoarding`），则 `rotatePerson` 换新 person 编号归档，避免「上车」「下车」两次方向覆盖写进同一目录。

### 任务2：匹配细节增强
- [x] 确认「关闭后保持计数」已实现（TripDataStore SQLite 持久化 + `loadOnboardFromDb` 恢复车厢人数，`onClearCounts` 才清零）。
- [x] 确认「采集信息」检测框已位于「人员进出记录」上方（mainwindow.ui 右侧面板）。
- [x] **匹配细节显示上/下车文件夹图片**：`initMatchDetailDialog` 增加上/下车代表图预览区（点击表格行 `onMatchDetailRowChanged` 加载 person 目录「乘客图片」中间一张）。
- [x] **新增「人工确认」列**：匹配细节表格第 5 列下拉框（未确认/正确/错误），`onManualConfirmChanged` 写回 `PersonEventRow.manualConfirm` 并持久化到 session_state.json。
- [x] **人工匹配工具导出 OSNet 训练数据**：`onExportTrainData` 把「人工确认正确」的匹配对合并上下车裁剪图导出为 torchreid 身份目录（train_XXXX，每身份 ≤80 图）。
- [x] 编译通过（`build-CameraViewer_passenger-Desktop-Debug`，make -j4，退出码 0）。

### 待验证
- [ ] 用新编译程序重新跑 person_311 / person_338 对应视频段，确认多人分离 + 折返判定正确（需实际视频回放验证）。
- [ ] 「下车折返」撤销比对的时序问题仍为预留（见上「未完成」），本次未处理。

---

## 2026-10-23 晚间（会话目录复用 + 匹配细节界面增强）

- [x] **修复「每次打开都新建文件夹」**：`m_sessionStatePath` 原来在 `initPersonFolder()` 之后才设置，导致 `ensureArchiveSessionDir` 读不到上次归档目录而每次新建。已调整初始化顺序（路径先于 initPersonFolder 设置），并在 `closeEvent` 补 `saveSessionState()`，实现「启动复用上次目录 + 恢复最后关闭的数据」。
- [x] **匹配细节点击表格加载图片**：`currentCellChanged`（刷新时误触发）改为 `cellClicked`；从第 0/1 列 `Qt::UserRole` 读 personId。
- [x] **人工确认列改为三个单选按钮**：`QComboBox` 下拉 → 3 个 `QRadioButton`（未确认/正确/错误）+ `QButtonGroup` 互斥。
- [x] **上车/下车图片各占一行**：单张代表图 → `QScrollArea` 横向滚动展示该 person 文件夹全部裁剪图（≤60 张）。
- [x] **采集信息移出 visitorPanel 容器**：用 ElementTree 将 `groupBox_info` 移到新 `rightColumnLayout`（采集信息在上、visitorPanel 人员进出记录在下），不再嵌在人员进出记录容器内。
- [x] 编译通过（make -j4，退出码 0）。


