# 辣椒图片 DataLoader 使用说明

`pepper_dataloader.py` 用于加载“两个品种 × 两个品级”的辣椒单体图片数据集，并返回标准的 PyTorch `DataLoader`。

## 1. 数据目录规范

```text
辣椒单体_透明PNG/成品/
├── 子弹头_差/<原图ID>/*.png
├── 子弹头_好/<原图ID>/*.png
├── 条子_差/<原图ID>/*.png
└── 条子_好/<原图ID>/*.png
```

程序会从类别目录名自动解析标签，无需手工生成 one-hot 编码。

| 任务 | 类别 | 整数标签 |
|---|---|---:|
| 品级 `quality` | 差 | 0 |
| 品级 `quality` | 好 | 1 |
| 品种 `species` | 子弹头 | 0 |
| 品种 `species` | 条子 | 1 |

四个数据组合分别是 `(species, quality) = (0,0)`、`(0,1)`、`(1,0)`、`(1,1)`。训练时建议使用两个分类头，分别预测品种和品级。

## 2. 安装依赖

```bash
python -m pip install torch torchvision pillow
```

## 3. 创建 DataLoader

```python
from pathlib import Path

from pepper_dataloader import create_dataloaders, describe_dataloaders

data_root = Path("辣椒单体_透明PNG/成品")

train_loader, val_loader, test_loader = create_dataloaders(
    dataset_root=data_root,
    batch_size=32,
    image_size=224,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
    num_workers=4,       # Windows 或调试时可设为 0
    balance_train=True,  # 按四种组合做加权采样
)

describe_dataloaders((train_loader, val_loader, test_loader))
```

划分不是直接随机拆分 838 张单体图片，而是按原图 ID 分组。同一张原图中切出的辣椒只会出现在同一个 split 中，避免训练集和验证/测试集之间的数据泄漏。

对当前 838 张图片使用上述默认参数时，划分结果为训练集 578 张、验证集 130 张、测试集 130 张；三者的原图 ID 交集为空。修改 `seed` 会改变具体划分，但同一个 `seed` 的结果可重现。

## 4. Batch 接口规范

每次迭代返回一个字典：

```python
batch = next(iter(train_loader))

# Tensor[B, 3, H, W]，torch.float32
images = batch["image"]

# Tensor[B]，torch.long；0=差，1=好
quality_targets = batch["quality"]

# Tensor[B]，torch.long；0=子弹头，1=条子
species_targets = batch["species"]

# list[str]
image_ids = batch["image_id"]
```

图像会保持长宽比填充成正方形，缩放到 `image_size × image_size`，然后按 ImageNet 均值和标准差归一化。PNG 的 alpha 透明区域默认合成为黑色背景；可通过 `background_rgb=(255, 255, 255)` 改为白色。

## 5. 训练代码示例

```python
import torch
from torch import nn

quality_loss_fn = nn.CrossEntropyLoss()
species_loss_fn = nn.CrossEntropyLoss()

model.train()
for batch in train_loader:
    images = batch["image"].to(device, non_blocking=True)
    quality_targets = batch["quality"].to(device, non_blocking=True)
    species_targets = batch["species"].to(device, non_blocking=True)

    # 模型应返回两个 [B, 2] logits，不要先做 softmax。
    quality_logits, species_logits = model(images)

    quality_loss = quality_loss_fn(quality_logits, quality_targets)
    species_loss = species_loss_fn(species_logits, species_targets)
    loss = quality_loss + species_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

`nn.CrossEntropyLoss` 需要的正是 `torch.long` 整数类别索引，所以不要对 `quality` 和 `species` 进行 one-hot 编码，也不要对输入损失函数的 logits 预先执行 softmax。

## 6. 快速检查

```bash
python pepper_dataloader.py
```

脚本会打印三个 split 的图片数量、四种组合的数量，以及一个 batch 的 shape 和 dtype。
