# passenger_clip_tool —— 乘客上下车录像离线批处理

把「乘客上下车录像」批量处理为**按乘客分文件夹**的上下车过程片段。

## 功能流水线
1. **自动解压**输入目录下的录像压缩包（`.rar`，支持 `partNN` 分卷）；
2. 对目录内**所有视频**逐帧执行「检测 + 跟踪 + ReID」；
3. 用 **深度 ReID（OSNet）** 把不同轨迹聚合为「同一个乘客」（**跨视频全局身份**，同一乘客跨分段保持同一 ID）；
4. 按每位乘客的**出现时段**裁剪原始视频，导出为独立片段；
5. 不同乘客保存到**不同文件夹** `passenger_<ID>/`，并生成 `manifest.csv` 清单。

## 构建
```bash
mkdir -p build-passenger-tool && cd build-passenger-tool
/home/froduo/Qt/6.2.4/gcc_64/bin/qmake ../tools/passenger_clip_tool.pro
make -j$(nproc)
```
产物：`build-passenger-tool/passenger_clip_tool`

## 用法（推荐用一键脚本）
```bash
./tools/run_passenger_clip.sh
# 或指定目录
./tools/run_passenger_clip.sh <输入目录> <输出目录>
```
等价的直接调用：
```bash
./build-passenger-tool/passenger_clip_tool \
    --input  "/run/user/1000/gvfs/sftp:host=192.168.200.8/share/vision/25.乘客视频" \
    --output "./passenger_clips" \
    --model  "./model/yolo26n_trt10_x86.engine" \
    --deep   "./model/osnet_x1_0_head_finetuned.onnx" \
    --gap 12 --pad 3
```

### 参数
| 参数 | 默认 | 说明 |
|---|---|---|
| `--input,-i` | 必填 | 输入目录（含 `.rar` 分卷或已解压视频；**递归**扫描） |
| `--output,-o` | `./passenger_clips` | 输出目录 |
| `--model` | `./model/yolo26n_trt10_x86.engine` | 检测模型（`.engine` 走 TensorRT，`.onnx` 走 OpenCV DNN） |
| `--conf` | `0.45` | 检测置信度阈值 |
| `--step` | `1` | 检测抽帧步长（1=每帧；调大可提速） |
| `--same` | `0.80` | **同一乘客判定阈值** `[0,1]`（越大越严格；0.62 会把不同人误并为同一人） |
| `--deep` | 空 | 深度 ReID 模型(.onnx)；留空则关闭深度（回退手工特征） |
| `--pad` | `1.0` | 每个片段前后留白（秒） |
| `--gap` | `2.0` | 同一乘客两次出现间隔超过该秒数则分段 |
| `--no-extract` | — | 跳过解压（输入目录已是解压后的视频） |

## 输出结构
```
passenger_clips/
├── passenger_1/
│   └── ch01_..._clip0_<start>-<end>.mp4
├── passenger_2/ ...
└── manifest.csv
```
`manifest.csv` 列：`video,passenger_id,direction,clip_file,start_frame,end_frame,start_sec,end_sec`
- `direction`：`boarding_down`（净向下）/ `alighting_up`（净向上）；重生成的历史清单可能为 `n/a`。

## 关键实现要点
- **多锚点身份匹配**：每位乘客保留 ≤8 个代表性特征，匹配取最大相似度；避免了「单一质心 EMA 漂移 → 身份坍缩」（把所有人都并成同一人）。
- **跨视频全局身份**：`passengers` 在 `main` 中维护并传入每个视频；但**出现时段 `ranges` 每个视频开始必须清空**，否则会把上一视频的时段重复导出到本视频（已修复）。
- **清单内存累积后一次性写盘**：避免长时间流式写 `manifest.csv` 导致后续视频丢行（已修复）。

## 注意事项
- **RAR5 解压**：`bsdtar`(libarchive) 只能列出、**无法正确解压 RAR5**（会产出 0 填充的大文件）。必须用 `unrar`/`7z`。项目内已在 `.conda_envs/unrar/bin/unrar` 安装，一键脚本会自动加入 PATH。
- **分卷完整性**：`.rar` 分卷须从 `part01`（该归档首卷）起齐全；缺失后续卷会导致解压中断。
- 大文件处理耗时较长（1080p/25fps，约 24 分钟/68 分钟视频，TensorRT+CUDA），程序每 10s 打印进度。
- `--same` 影响聚合松紧：同一人被拆成多人 → 调小；不同人被并成一人 → 调大。

## 本轮实测记录（2026-10-10）
- 输入：`25.乘客视频` 内 3 个 `ch01_...topspeed.part12/13/14.rar`。
- 实测为**同一多卷归档的第 12/13/14 卷**（part12 即首卷）；成功解出 **2 个完整视频**：
  `ch01_20260930_133328_144128topspeed.mp4`、`ch01_20260930_144128_154927topspeed.mp4`（各约 1.07GB，1920×1080@25fps，约 68 分钟）。
  第 3 个 `ch01_20260930_154927_165727topspeed.mp4` 需要 **缺失的 `part15`**，无法解压。
- 处理结果（`passenger_clips/`）：**共 168 个片段、44 个乘客 ID**，磁盘约 1.5GB；抽样 ffprobe 校验全部可播放。
  - `ch01_...133328_144128topspeed`：25 位乘客 / 93 片段
  - `ch01_...144128_154927topspeed`：累计 56 位乘客（新增 31 位）/ 75 片段
