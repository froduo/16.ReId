# CameraViewer 部署到 192.168.10.88（test 盘）说明

> 目标机：`192.168.10.88`（用户 `tianheng` / 密码 `tianheng123`）
> 程序：`16.reId/CameraViewer`（Qt6 抓拍 / 人员检测 / ReID）
> 说明：本包已在源码内附带一键脚本 `deploy/deploy_cameraviewer.sh` 与运行配置 `deploy/config.ini`。

---

## 0. 前置条件（88 上需具备）

| 组件 | 要求 | 检测方式 |
|---|---|---|
| 编译工具 | g++ / make | `g++ --version` |
| Qt6 | qmake + Widgets/Network/Sql（如 6.2.4） | `qmake6 -v` 或 `/home/*/Qt/6.*/gcc_64/bin/qmake -v` |
| OpenCV 4 | 建议 CUDA 版，`/usr/local/include/opencv4` | `pkg-config --modversion opencv4` |
| CUDA | `/usr/local/cuda` | `ls /usr/local/cuda` |
| TensorRT 10.3 | `/usr/local/TensorRT-10.3` | `ls /usr/local/TensorRT-10.3` |
| NVIDIA 驱动 | 与显卡匹配（显卡自原机迁移，需在 88 装对应驱动） | `nvidia-smi` |
| 图形桌面 | GUI 程序需要 X（`DISPLAY`） | `echo $DISPLAY` |

> 若本机 OpenCV 未编译 CUDA，脚本会自动给 `.pro` 打补丁（关闭 `USE_OPENCV_CUDA` 与 cuda 链接库），不影响运行，只是背景分割走 CPU。
> CUDA 与 TensorRT 无法用 apt 自动装，需按 NVIDIA 官方方式安装（版本建议与原机一致：TRT 10.3）。

---

## 1. 把源码/部署包送到 88

任选一种（在**源机器**上执行）：

- **scp**（若两机互通且有 ssh）：
  ```sh
  scp -r 16.reId/CameraViewer tianheng@192.168.10.88:/tmp/
  # 或使用本目录打好的包：
  scp 16.reId/CameraViewer_deploy_20261013.tar.gz tianheng@192.168.10.88:/tmp/
  ```
- **U 盘 / SMB 共享**：把 `CameraViewer` 目录（或 `CameraViewer_deploy_20261013.tar.gz`）拷到 88。
- 若用 tar 包，在 88 上解压：
  ```sh
  tar xzf /tmp/CameraViewer_deploy_20261013.tar.gz -C /tmp/
  ```

---

## 2. 在 88 上执行部署

先确认 “test 盘” 的挂载点（脚本会自动找，找不到就手动指定）：

```sh
lsblk -o NAME,SIZE,FSTYPE,LABEL,MOUNTPOINT      # 找到 label/挂载点含 test 的盘
```

然后运行（推荐 `bash`，不是 `sh`）：

```sh
cd /tmp/CameraViewer
bash deploy/deploy_cameraviewer.sh
```

脚本会依次：定位 test 盘 → 拷贝源码到 `<test盘>/CameraViewer` → 检查依赖 → 打 CUDA 补丁(如需) → `qmake` + `make` → 拷模型/`config.ini`/`style.qss` → 启动程序。

常用变体：

```sh
# 只编译不运行
bash deploy/deploy_cameraviewer.sh --no-run

# 强制指定 test 盘挂载点
TEST_ROOT=/media/tianheng/test bash deploy/deploy_cameraviewer.sh

# 已在源码目录内，原地编译
bash deploy/deploy_cameraviewer.sh --in-place

# 尝试 apt 安装基础依赖（CUDA/TRT 仍需手动）
bash deploy/deploy_cameraviewer.sh --install-deps
```

编译成功后产物：`<test盘>/CameraViewer/build/CameraViewer`。

---

## 3. 手动运行（脚本已启动可跳过）

GUI 必须在**图形桌面会话**里运行：

```sh
cd <test盘>/CameraViewer/build
./CameraViewer
```

若通过 SSH 且本机有 X：`DISPLAY=:0 ./CameraViewer`。

---

## 4. 配置说明（`deploy/config.ini`）

脚本会把 `deploy/config.ini` 写入 `build/config.ini`（已存在则不覆盖，需要覆盖用 `FORCE_CFG=1`）。
当前关键项（与源机 main 运行目录一致）：

```ini
[Inference]
ModelPath=./model/passenger_yolo26n_trt10_x86.engine
OnnxPath=./model/passenger_yolo26n.onnx
ModelKind=person
PreferOnnx=false
Letterbox=true
ConfThreshold=0.30

[ReID]
UseDeepReid=true
DeepModelPath=./model/osnet_x1_0_head_finetuned.onnx
DeepMatchThreshold=0.60
```

模型文件来自源码 `CameraViewer/model/`（含 `passenger_yolo26n*`、`passenger_yolo11n*`、`yolo26n*`、`osnet_x1_0_head_finetuned.onnx` 等），脚本会整体拷入 `build/model/`。
> `*.pt` / `*.pth` / `*.prev_*` 未打进部署包（运行不需要），如仍需可用源目录补拷。

---

## 5. 常见问题

- **qmake 报找不到 Qt**：`export QMAKE_BIN=/home/tianheng/Qt/6.2.4/gcc_64/bin/qmake` 后重跑。
- **链接 `-lnvinfer` 失败**：TensorRT 未装或版本不符；确认 `/usr/local/TensorRT-10.3/targets/x86_64-linux-gnu/lib` 存在。
- **链接 `-lopencv_cuda*` 失败**：脚本应已自动注释；若手改了 `.pro`，检查 `DEFINES += USE_OPENCV_CUDA` 是否已注释。
- **engine 反序列化失败**：`.engine` 与 GPU 架构/TRT 版本绑定，需在 88 上用 `trtexec` 重新生成（源目录 `model/` 有同名 `.onnx`）。
- **程序起不来无界面**：当前是 SSH 会话无 `DISPLAY`，请在桌面终端运行或 `DISPLAY=:0`。
- **想改回旧模型**：改 `build/config.ini` 的 `ModelPath=./model/yolo26n_trt10_x86.engine` 即可。

---

## 6. 传输受限说明

本次部署**未**由我（助手）直接 ssh/scp 到 88 —— 本会话策略禁止 `ssh`，且本机无 88 的挂载点。
因此采用“离线包 + 一键脚本”方式：你在源机把包送到 88，然后在 88 上跑脚本。
