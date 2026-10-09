# CS231n 学习代码

这是我在学习 CS231n 和计算机视觉基础时整理的练习代码。仓库按学习主题分类，主要记录从基础 CNN、ResNet、Vision Transformer 到表征学习的实现与实验。

代码以学习过程为主，保留了原有的 `TODO`、注释和实验写法，没有进行额外的复杂工程化封装。

## 目录结构

```text
cs231n-learning/
├── 01_cnn_basics/
│   └── train_cnn_cifar10.py
├── 02_resnet/
│   ├── resnet_from_scratch.py
│   └── pretrained_resnet_encoder.py
├── 03_vision_transformer/
│   ├── vit.py
│   ├── transformer.py
│   └── train_vit.py
├── 04_representation_learning/
│   ├── vit_representation.py
│   ├── linear_probe.py
│   ├── dinov2_representation.py
│   └── dino_toy.py
└── projects/
    ├── README.md
    └── visual_world_model/  # Reacher 视觉世界模型小项目（15 个脚本）
```

## 内容说明

- `01_cnn_basics`：在 CIFAR-10 上训练简单 CNN。
- `02_resnet`：从零实现 ResNet18，以及使用 torchvision 预训练 ResNet 提取图像特征。
- `03_vision_transformer`：ViT 的 patch embedding、Transformer block 和 CIFAR-10 训练代码。
- `04_representation_learning`：预训练 ViT/DINOv2 表征分析、1-NN 评估、linear probe 和简化版 DINO 思路练习。
- `projects/visual_world_model`：Reacher 视觉世界模型学习项目，包括数据采集、视觉自编码器、目标坐标监督、潜空间动力学和 rollout 评估。详见 [项目说明](projects/visual_world_model/README.md)。后续小项目可继续在 `projects/` 下添加独立目录。

## 环境安装

建议使用 Python 3.10 或更高版本，并在虚拟环境中安装依赖：

```bash
pip install -r requirements.txt
```

视觉世界模型项目还需额外安装：

```bash
pip install -r projects/visual_world_model/requirements.txt
```

## 运行示例

进入相应目录后运行脚本，例如：

```bash
cd 01_cnn_basics
python train_cnn_cifar10.py
```

部分脚本会自动下载 CIFAR-10、torchvision 预训练权重或 DINOv2 模型，需要网络连接。下载的数据、模型权重和缓存不会提交到 Git 仓库。

## 当前附件说明

原课程学习代码成功恢复并整理了 10 个文件，后续合并了视觉世界模型小项目的 15 个脚本。Vision Transformer 代码还会导入 `attention.py`，但该文件未包含在当前可访问的附件中，因此相关脚本需要在补充该文件后才能完整运行。这里没有自行补写缺失实现，以保持学习代码的真实性。
