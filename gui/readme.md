# GUI: Pi3 point cloud interaction

本目录提供 **Pi3 点云 + 相机视锥** 的交互界面（Open3D）：**Ego** 模式绕参考相机旋转点云与相机；**Global** 模式绕场景中心轨道观察。行为与 Pi3 服务端的 ego / global 视角逻辑对齐。

**依赖：** `pip install open3d`（本仓库未固定版本）。

**前置：** 已启动 Pi3 HTTP 服务（如 `spagent/external_experts/Pi3/pi3_server.py`），并知道其地址（下文示例为 `http://localhost:20021`，请按实际修改）。

---

## 脚本入口

在**仓库根目录**下执行：

```bash
python gui/gui_manipulation/pi3_pointcloud_interactive.py [选项]
```

---

## 推荐用法（与 `pi3_pointcloud_interactive.py` 一致）

### 1）仅本地文件，不访问服务（已有 PLY + 相机 JSON）

若 `outputs/result_front_125.ply` 与同目录下的 `result_front_125_cameras.json`（或 `<ply 文件名去掉 .ply>_cameras.json`）已存在，**不要加 `--fetch`**：

```bash
python gui/gui_manipulation/pi3_pointcloud_interactive.py \
  --ply outputs/result_front_125.ply
```

- 默认相机文件路径：与 PLY 同目录、文件名为 `<ply_stem>_cameras.json`。
- 自定义相机 JSON：`--cameras-json /path/to/custom_cameras.json`。

---

### 2）首次或缺少文件：通过 `Pi3Client` 调 `/infer`，再打开 GUI

脚本使用 **`--fetch`** 调用服务端推理（`generate_views=False`），写入 PLY 与 `*_cameras.json`，然后启动界面。

**还没有 PLY、不想手写输出路径：** 省略 `--ply`，文件写入 **`--out-dir`**（默认 `outputs`），PLY 文件名由服务端返回的 `ply_filename` 决定：

```bash
python gui/gui_manipulation/pi3_pointcloud_interactive.py \
  --fetch -f outputs/gui_demo_image.txt \
  --server http://localhost:20021
```

**指定输出目录：**

```bash
python gui/gui_manipulation/pi3_pointcloud_interactive.py \
  --fetch -f outputs/gui_demo_image.txt \
  --out-dir /path/to/out \
  --server http://localhost:20021
```

**已确定本地 PLY 路径（不存在则创建）：**

```bash
python gui/gui_manipulation/pi3_pointcloud_interactive.py \
  --ply outputs/result_front_125.ply \
  --fetch -f outputs/gui_demo_image.txt \
  --server http://localhost:20021
```

**图像列表文件 `-f`：** 每行一张图路径（相对仓库根或绝对路径）；`#` 开头与空行忽略。须与重建所用视角一致。

---

### 3）带 `--fetch` 但本地已有 PLY 和 JSON

若同时传入 **`--ply`** 且该 PLY 与默认（或 `--cameras-json`）相机文件**都已存在**，脚本默认 **不再请求** `/infer`，直接打开 GUI。需要强制重新推理时：

```bash
python gui/gui_manipulation/pi3_pointcloud_interactive.py \
  --ply outputs/result_front_125.ply \
  --fetch -f outputs/gui_demo_image.txt \
  --force-fetch \
  --server http://localhost:20021
```

---

## 常用参数说明

| 参数 | 说明 |
|------|------|
| `--ply` | PLY 路径。与 `--fetch` 联用时若省略，则使用 `--out-dir` + 服务端 `ply_filename`。仅本地打开时**必填**。 |
| `--out-dir` | 仅在 **`--fetch` 且未指定 `--ply`** 时生效；输出目录，默认 `outputs`。 |
| `--cameras-json` | 相机元数据 JSON；默认 `<ply_stem>_cameras.json`（与 PLY 同目录）。 |
| `--fetch` | 通过 `Pi3Client` 请求 `/infer`，按需写入 PLY 与相机 JSON。 |
| `--force-fetch` | 与 `--fetch` 同用时，即使本地已有 PLY+JSON 也重新推理。 |
| `-f` / `--images-file` | `--fetch` **必填**：图像路径列表文件。 |
| `--server` | Pi3 服务根 URL，默认 `http://localhost:20021`。 |
| `--conf-threshold` | 传给推理的置信度阈值，默认 `0.1`。 |
| `--rtol` | 传给推理的深度边缘 rtol，默认 `0.03`。 |
| `--voxel` | 点云体素下采样尺寸；默认 `0.002`，`<=0` 可关闭下采样（以脚本实现为准）。 |

更细的说明见脚本文件顶部 **module docstring**，或执行：

```bash
python gui/gui_manipulation/pi3_pointcloud_interactive.py --help
```

---

## 可选：不用 GUI，仅用 Python 调用 `Pi3Client` 生成 PLY + JSON

若希望完全自行保存结果，可在仓库根目录用 `Pi3Client` 调用 `/infer`，再 `save_results`，并把 `camera_poses` 存成与上面相同命名的 `*_cameras.json`。一般可直接用上一节的 **`--fetch`** 一条命令替代。

---

## 文件位置

| 作用 | 路径 |
|------|------|
| 交互查看器 | `gui/gui_manipulation/pi3_pointcloud_interactive.py` |
