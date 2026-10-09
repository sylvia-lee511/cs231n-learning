# Reacher 视觉世界模型学习项目

在 Gymnasium 的 `Reacher-v5` 环境中采集图像和随机动作，学习图像表征，再训练潜空间动力学模型预测后续状态。项目保留了学习过程中的不同实验版本，未记录未经验证的训练成绩。

## 文件说明

| 文件 | 内容 |
| --- | --- |
| `collect_data.py` | 原始图像、动作与 episode 信息采集 |
| `collect_data_modified.py` | 增加与图像帧对齐的目标世界坐标 |
| `dataset.py` | 原始相邻帧数据集 |
| `dataset_modified.py` | 图像预处理、带目标坐标的帧对数据集、动作转移数据集 |
| `model.py` | 视觉自编码器、目标编码器、有动作与无动作的潜空间动力学模型 |
| `model_modified.py` | 带世界坐标预测头的视觉自编码器 |
| `overfit.py` | 小样本过拟合实验（早期版本） |
| `train_autocoder.py` | 视觉重建实验，包含前景与目标相关损失 |
| `train_autocoder_armmask.py` | 机械臂前景掩码加权重建实验 |
| `train_autoencoder_targetcoord.py` | 增加图像目标坐标监督的实验 |
| `train_autoencoder_worldtarget.py` | 增加仿真器目标世界坐标监督的实验 |
| `train_target_encoder.py` | 单独训练目标世界坐标预测编码器 |
| `train_dynamics.py` | 有动作条件的潜空间动力学训练 |
| `train_noaction_dynamics.py` | 无动作条件的对照实验 |
| `evaluate_rollout.py` | 多步 rollout 评估，比较有动作、无动作及 identity 基线 |

文件名保留原样（包括 `autocoder` 和 `_modified`），便于对应原学习记录。

## 环境与路径

从仓库根目录安装：

```bash
pip install -r projects/visual_world_model/requirements.txt
```

数据采集需要 MuJoCo 环境及可用的图像渲染环境。训练脚本会优先使用可用的 CUDA，否则使用 CPU。

为适配原代码中的相对路径，从仓库根目录先进入 `projects/`，之后都在此处运行：

```bash
cd projects
python visual_world_model/collect_data_modified.py
```

数据生成在 `projects/data_visual_world_model/reacher_visual_random.npz`。原始采集脚本不保存 `target_xy_world`，世界坐标监督实验需要使用 modified 版本生成的数据。两个采集脚本使用相同的输出文件名，重新采集会覆盖该文件。

## 主要实验顺序

以下命令表示主要实验流程，运行前须准备对应数据及训练产生的权重：

```bash
python visual_world_model/train_autoencoder_worldtarget.py
python visual_world_model/train_dynamics.py
python visual_world_model/train_noaction_dynamics.py
python visual_world_model/evaluate_rollout.py
```

后面三个脚本读取 `visual_world_model/best_visual_autoencoder_worldtarget.pt`；rollout 还需 `best_latent_dynamics.pt` 与 `best_noaction_dynamics.pt`。这些权重由训练产生，不包含在仓库中。默认采集 200 个 episode，训练、验证、测试使用 0–159、160–179、180–199 的划分。

## 实验版本与验证范围

本次合并仅复制原始 15 个脚本并添加说明、依赖与忽略规则。所有脚本通过语法检查；未执行完整数据采集、训练或 rollout，不代表所有版本可直接连贯运行。

已发现的版本差异：

- `overfit.py` 引用了 `dataset.py` 中未提供的 `ReacherFrameDataset`。
- `train_autocoder.py`、`train_autocoder_armmask.py`、`overfit.py` 和 `model.py` 的直接运行示例按两个返回值解包；当前 `model.py` 的 `VisualAutoencoder.forward` 返回三个值。
- 世界坐标训练使用 `model_modified.py`，动力学与 rollout 使用 `model.py`。两者目标预测头的输出激活不同，跨版本加载权重时需确认所使用的头；动力学训练主要使用 encoder 提取特征。

上述差异保留为学习记录。数据目录、NumPy 数据缓存和模型权重已加入根目录 `.gitignore`，交付包只包含源码与说明。
