# 黔椒智检

「黔椒智检」是面向传送带场景的贵州辣椒 AI 视觉检测系统。当前主任务只输出两个层次：辣椒品种（子弹头 / 条子）与品级（一级 / 二级）。后端保留原有批次、历史记录和 A/B/C 数据库字段仅为旧功能兼容；摄像头主界面以检测框内的品种与一/二级为准。

当前版本无需 Redis、消息队列或 Docker。FastAPI 同进程通过 OpenCV 只采集显式绑定的外接摄像头，YOLO11 持续定位多枚辣椒，分层分类器实时计算品种、条件品级和联合概率，检测框持续绘制在 MJPEG 主画面上直到目标离开。Mock 服务仍可用于测试历史数据页，但不是实时摄像头的默认推理链。

## 已实现功能

- 16:9 实时识别主界面：真实 MJPEG 相机画面、多辣椒持续检测框、品种/品级与联合置信度
- 中央固定 ROI 帧差检测：目标经过时自动截图，带冷却和自动重连
- 实时模型链：单类 YOLO11 定位 + `p(品种)` + 两个 `p(品级|品种)` + 联合概率 argmax
- 五项客观指标：色泽、完整度、形态、尺寸、缺陷控制
- 独立品质评分服务：可配置权重、A/B 阈值和评级解释
- 硬性降级规则：任意 `severity=severe` 缺陷直接 C
- 检测记录入库、筛选、搜索、分页及原始图/标注图详情
- 批次分析：等级占比、分数分布、平均指标、缺陷统计
- 模板化 AI 质量分析报告，预留真实 LLM 替换边界
- WebSocket 自动广播与前端断线重连
- 后端 Mock 模式：开始、暂停、单样本和清空当前批次
- SQLite 持久化评级规则，保存后立即用于后续检测
- multipart 图片上传、后端静态图片服务与原图回退标注图
- FastAPI Swagger 接口文档与集成测试

## 系统架构

```text
外接摄像头 ── OpenCV ── MJPEG /api/stream ───────┐
     │                                             │
     └─ ROI 持续触发 ── YOLO11 多目标定位 ── 分层分类 ─┤
                                                   ▼
外部 AI视觉服务 ── multipart POST /api/detections ─▶ FastAPI Detection Service
              │
     ┌────────┼──────────┐
     ▼        ▼          ▼
评分引擎   SQLite     WebSocket 广播
     │        │          │
     └────────┴──────────┤
                        ▼
             Next.js 实时质检驾驶舱
                │        │        │
             质检记录   批次分析   规则配置
```

摄像头链路中的目标检测与分层分类均为真实模型；截图、SQLite 入库与 WebSocket 实时推送也在同一链路内。外部视觉服务仍可通过 multipart API 提交结果，无需改动前端。

## 目录结构

```text
qianjiao-ai-inspection/
├── backend/
│   ├── app/
│   │   ├── api/                 # REST 与 WebSocket 路由
│   │   ├── database/            # SQLite 会话与旧表无损迁移
│   │   ├── models/              # Batch / Detection / GradingRule
│   │   ├── schemas/             # Pydantic 请求和响应模型
│   │   ├── services/
│   │   │   ├── camera_service.py       # 摄像头、MJPEG 与帧差触发
│   │   │   ├── camera_pipeline.py      # 截图进入检测流程
│   │   │   ├── detection_service.py    # 评分、入库与广播统一链路
│   │   │   ├── storage_service.py      # 上传图片存储
│   │   │   ├── quality_service.py
│   │   │   ├── mock_service.py
│   │   │   ├── stats_service.py
│   │   │   └── analysis_service.py
│   │   ├── websocket/           # 连接管理与广播
│   │   └── main.py
│   └── tests/
├── uploads/                     # 摄像头截图和 API 上传图片
├── frontend/
│   ├── app/                     # Next.js App Router 页面
│   ├── components/              # 驾驶舱、记录详情、公共组件
│   ├── hooks/                   # WebSocket 自动重连
│   ├── lib/                     # API 客户端与格式化工具
│   ├── public/images/           # Mock 相机与标注画面
│   └── types/                   # TypeScript 领域类型
├── setup.bat
└── start-demo.bat
```

## 快速启动（Windows）

本项目已在 Windows、Python 3.10、Node.js 18、OpenCV 4.8/4.10 环境验证。

首次使用可双击：

```text
setup.bat
```

安装完成后双击：

```text
start-demo.bat
```

脚本会启动两个终端并打开实时质检台：

- 前端：[http://localhost:3000/inspection](http://localhost:3000/inspection)
- 后端接口文档：[http://localhost:8000/docs](http://localhost:8000/docs)
- 健康检查：[http://localhost:8000/api/health](http://localhost:8000/api/health)
- 摄像头状态：[http://localhost:8000/api/camera/status](http://localhost:8000/api/camera/status)
- MJPEG 画面：[http://localhost:8000/api/stream](http://localhost:8000/api/stream)

首次运行会安装 `opencv-python-headless` 和 `python-multipart`。这里的 headless 仅移除了 OpenCV 自带 GUI 窗口，`VideoCapture`、JPEG 编码和摄像头采集仍可用。

### 摄像头权限与设备选择

1. 在 Windows **设置 → 隐私和安全性 → 相机** 中打开“允许桌面应用访问相机”。
2. 将 USB/工业相机连接到电脑，并关闭正在独占它的相机调试软件。
3. 系统只打开一个明确指定的外接设备索引，绝不轮询其他索引或回退到电脑内置摄像头。当前这台 macOS 电脑已绑定 USB 相机 `Web Camera`（序列号 `202604081837`），其 OpenCV 索引为 `0`：

```bash
export QJ_CAMERA_INDEX="0"
export QJ_CAMERA_NAME="Web Camera"
export QJ_CAMERA_SERIAL="202604081837"
```

Windows 可在启动后端前指定对应的外接设备索引；Windows 当前只执行“单索引、不回退”策略：

```powershell
$env:QJ_CAMERA_INDEX="1"         # 外接摄像头的 OpenCV 设备索引
$env:QJ_CAMERA_NAME="Web Camera" # 状态页显示的设备名称
$env:QJ_MOTION_THRESHOLD="0.06"  # 数值越小越灵敏
$env:QJ_TRIGGER_COOLDOWN="0.8"   # 连续目标最小间隔秒数
```

启动后访问 `/api/camera/status`。`external_verified=true`、`online=true` 且 `fps>0` 表示指定 USB 摄像头已通过身份校验并成功采集；返回的 `camera_index` 是当前唯一允许打开的设备。`/api/stream` 应持续返回 `multipart/x-mixed-replace; boundary=frame`。未连接指定 USB 设备或权限不足时页面显示稳定诊断帧，服务保持离线，且不会尝试任何其他摄像头。

`POST /api/camera/trigger?batch_id=...` 是预留给光电/红外等硬件传感器的手动触发入口：它会抓取当前帧并进入同一检测链路；当前默认启用的是中央 ROI 像素变化自动触发。

### 手动启动后端

```powershell
cd backend
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

### 手动启动前端

```powershell
cd frontend
npm install
npm run dev
```

如本机 npm 官方源不稳定，可使用 pnpm：

```powershell
pnpm install --registry=https://registry.npmmirror.com
pnpm dev
```

## 现场 Demo 流程

1. 打开 `/inspection`，确认顶部显示“系统运行中”。
2. 真实摄像头画面会立即显示；物体经过中央 `TRIGGER ZONE` 时，帧差算法自动截图并检测。
3. 如需完全可控的兜底演示，点击“开始模拟检测”。后端会立即生成第一个样本，之后每 2～4 秒生成一个。
4. 观察画面依次显示“捕获图像 → 分析外观特征 → 计算品质评分 → 检测完成”。
5. 观察当前等级、评级依据、A/B/C 数量和平均指标实时变化，无需刷新。
6. 点击“暂停”后，可进入“批次分析”查看图表并生成 AI 质量分析。
7. 进入“质检记录”，筛选等级并点击任一行查看原始图、标注图和完整证据。
8. 若需重置，先暂停模拟，再点击“清空当前批次”。

Mock 样本期望分布约为 A 级 55%、B 级 30%、C 级 15%，并随机包含黑斑、破损、色泽异常、形态异常、霉变和虫害。约 10% 样本会生成任意类型的 `severe` 缺陷，用于稳定演示强制降级。

## 接入真实摄像头 / AI 模型

前端无需修改。视觉服务只需在每次完成目标检测后，以 `multipart/form-data` 调用：

```http
POST http://localhost:8000/api/detections
Content-Type: multipart/form-data
```

字段：

- `image`：必填原始图片（JPEG / PNG / WebP，最大 12MB）
- `payload`：必填 `DetectionCreate` JSON 字符串
- `annotated_image`：可选标注图片；不传时自动使用原图 URL

PowerShell 示例：

```powershell
$payload = '{"batch_id":"BATCH-GZ-20260828-01","color_score":92,"integrity_score":85,"shape_score":88,"size_score":90,"defect_score":86,"confidence":0.94,"defects":[{"type":"black_spot","name":"表面黑斑","severity":"mild","confidence":0.87}]}'
curl.exe -X POST "http://localhost:8000/api/detections" `
  -F "image=@C:\captures\pepper-001.jpg;type=image/jpeg" `
  -F "payload=$payload"
```

`payload` 完整结构示例：

```json
{
  "batch_id": "BATCH-GZ-20260828-01",
  "variety": "贵州辣椒",
  "length": 12.8,
  "width": 2.9,
  "color_score": 92,
  "integrity_score": 85,
  "shape_score": 88,
  "size_score": 90,
  "defect_score": 86,
  "confidence": 0.94,
  "processing_time": 68,
  "defects": [
    {
      "type": "black_spot",
      "name": "表面黑斑",
      "severity": "mild",
      "confidence": 0.87,
      "area_ratio": 0.03
    }
  ]
}
```

后端会自动完成：综合评分 → override 规则 → 入库 → WebSocket 广播 → 前端实时更新。

后端会保存图片并生成 `/uploads/...` URL。若 AI 服务另有标注图，可通过可选的 `annotated_image` 文件一并上传。返回中的数据库 `id` 是整数主键，对外展示编号是 `sample_code`；详情路由继续使用整数 `id`。

当前外接摄像头默认走 `YOLO11 定位 → 分层分类 → SQLite → WebSocket`。`backend/app/services/camera_pipeline.py` 对同一帧中的每个辣椒框分别生成品种/品级结果，目标留在画面时只更新框，离开后才清除。

## 评分规则

默认公式：

```text
quality_score = 色泽 × 30% + 完整度 × 30% + 形态 × 15% + 尺寸 × 10% + 缺陷 × 15%
```

- `quality_score >= 85`：A级 / 优质
- `65 <= quality_score < 85`：B级 / 合格
- `quality_score < 65`：C级 / 待处理

`defects[].severity` 只允许 `mild`、`moderate`、`severe`。任何类型只要出现 `severe`，都会忽略正常等级结果并强制判定为 C，`grade_reason` 会返回触发的缺陷类型。

规则可在 `/settings/grading` 修改并持久化到 SQLite。核心实现位于 `backend/app/services/quality_service.py`，没有写死在 API 路由中。

## 当前真实模型

后端默认使用物理隔离的 clean v5 链路：`yolo11n_pepper_strict_v5_f4/weights/best.pt` 负责多辣椒定位，ImageNet YOLO11n-cls 骨干提取 3584 维多尺度特征，分层 SVM 分别计算 `p(品种)` 和 `p(品级|品种)`，最后对联合概率做温度校准和 argmax。在线裁剪严格复现离线 canonical 规范：256×256、辣椒长边占 88%、RGB `(64,68,68)` 背景。

模型启动时会校验检测器、分类器、ImageNet 骨干和 selection 封存回执的 SHA256，任一不一致都拒绝加载。当前 clean v5 只有物理隔离验证集结果：品种准确率 99.32%、条件品级准确率 86.30%、四类联合准确率 85.62%。**这些不是独立盲测结果**；接口状态会返回 `provenance.validation_only=true`、`strict_test_evaluated=false` 和 `test_metrics=null`，防止把验证数据误报为测试成绩。

如需回退到历史 v4 双特征分类器，可通过环境变量同时指定三个已封存文件：

```powershell
$env:QJ_DETECTOR_MODEL="...\runs\yolo11n_pepper\weights\best.pt"
$env:QJ_CLASSIFIER_MODEL="...\runs\hierarchical_v4\best_hierarchical_v4_svm_strict.joblib"
$env:QJ_CLASSIFIER_SELECTION="...\runs\hierarchical_v4\v4_selection\svm_selection_strict.json"
```

自定义检测器还必须提供 `QJ_DETECTOR_SHA256`；自定义 selection 可使用 `QJ_CLASSIFIER_SELECTION_RECEIPT` 指向回执，或显式设置 `QJ_CLASSIFIER_SELECTION_SHA256`。历史 v4 仅为兼容通道，其 pepper 特征骨干的数据来源不如 clean v5 可追溯。

## 验证命令

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest -q

cd ..\frontend
npm run lint
npm run typecheck
npm run build
```
