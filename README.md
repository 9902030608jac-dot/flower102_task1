# Flowers102 Task 1: Fine-tuning ImageNet-pretrained CNNs

本项目完成深度学习课程期中作业任务 1：微调 ImageNet 预训练卷积神经网络，在 `102 Category Flower Dataset` 上完成 102 类图像分类实验。

注意：题目标题写作“宠物识别”，但基本要求明确指定 `102 Category Flower Dataset`。因此本工程、实验和报告结果全部以 Flowers102 图像分类为准；报告中可说明这是题目文字表述与数据集描述不一致导致的命名问题。

## Directory

```text
flower102_task1/
├── README.md
├── requirements.txt
├── requirements_server.txt
├── configs/
│   ├── baseline_resnet18.yaml
│   ├── baseline_resnet34.yaml
│   ├── scratch_resnet18.yaml
│   ├── se_resnet18.yaml
│   └── sweep_hparams.yaml
├── scripts/
│   ├── train.py
│   ├── eval.py
│   ├── download_assets.py
│   ├── run_baseline.sh
│   ├── run_ablation.sh
│   ├── run_attention.sh
│   ├── run_hparam_sweep.sh
│   ├── run_extra_hparam_factorial.sh
│   ├── run_extended_budget.sh
│   ├── summarize_results.py
│   └── analyze_confusions.py
├── src/
│   ├── datasets.py
│   ├── transforms.py
│   ├── models/
│   │   ├── resnet_baseline.py
│   │   ├── attention.py
│   │   ├── se_resnet.py
│   │   └── cbam_resnet.py
│   ├── engine.py
│   ├── metrics.py
│   ├── utils.py
│   └── logger.py
├── outputs/
└── report_assets/
```


## GitHub Submission Policy

本仓库用于提交代码、配置文件、实验结果表、训练曲线、混淆矩阵和报告源码。以下内容不提交到 GitHub：

- `data/`: Flowers102 数据集文件与压缩包；
- `assets/`: torchvision/ImageNet 预训练权重缓存；
- `outputs/checkpoints/`: 训练得到的 `.pt` 模型权重；
- `outputs/logs/`: 运行日志或在线记录工具缓存。

上述排除规则已写入 `.gitignore`。首次 clone 后，请按照本文的 Assets Download 章节重新下载数据集和 ImageNet 预训练权重。训练完成的模型权重请从网盘下载后放回对应的 `outputs/checkpoints/{experiment_name}/best.pt` 路径。权重上传清单见 `MODEL_WEIGHTS.md`。

## Environment

优先检查服务器自带环境。如果 base 或已有 conda 环境已经包含可用的 CUDA 版 PyTorch，就不必重新下载 PyTorch，这会节省很多时间。

```bash
python --version
python -c "import sys; print(sys.executable)"
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('torch_cuda', torch.version.cuda); print('device_count', torch.cuda.device_count())"
python -c "import torchvision; print('torchvision', torchvision.__version__)"
```

如果 `torch.cuda.is_available()` 输出 `True`，且 Python 版本为 3.10+、torch/torchvision 能正常 import，可以直接在当前环境安装缺失依赖：

```bash
pip install -r requirements.txt
```

如果服务器环境较乱，或者没有 PyTorch，再新建干净环境：

```bash
conda create -n flower102 python=3.10 -y
conda activate flower102
pip install -r requirements.txt
```

### Current Remote Server Recommendation

当前远程主机 `/root/miniconda3` 的 base 环境已经检查过：

```text
Python 3.12.3
torch 2.8.0+cu128
torchvision 0.23.0+cu128
numpy 2.2.6
tqdm 4.67.1
scikit-learn 1.7.2
matplotlib 3.10.7
PyYAML 6.0.2
Pillow 12.0.0
```

缺少的主要是：

```text
pandas
timm
swanlab
```

因此推荐 clone base，而不是重新创建一个从零安装 PyTorch 的环境：

```bash
source /root/miniconda3/bin/activate base
conda create -n flower102 --clone base -y
conda activate flower102
cd /root/autodl-tmp/flower102_task1
pip install -r requirements_server.txt
```

`requirements_server.txt` 不包含 `torch` 和 `torchvision`，这样可以避免覆盖服务器已有的 CUDA 版 PyTorch。完整从零环境才使用 `requirements.txt`。

如果你不 clone base，而是直接在 base 环境运行，也可以：

```bash
source /root/miniconda3/bin/activate base
cd /root/autodl-tmp/flower102_task1
pip install -r requirements_server.txt
```

但课程实验更推荐单独环境 `flower102`，避免污染 base。

## Assets Download

本仓库不包含 Flowers102 数据文件，也不包含 ImageNet 预训练权重。资源下载被单独拆到 `scripts/download_assets.py`，不和核心训练脚本混在一起。

首次训练前建议显式运行：

```bash
python scripts/download_assets.py --config configs/baseline_resnet18.yaml --models resnet18 resnet34
```

该脚本会下载两类资源：

| Resource | Default path | Description |
| --- | --- | --- |
| Flowers102 dataset | `data/flowers-102/` | 由 `torchvision.datasets.Flowers102(root="data", download=True)` 创建 |
| ImageNet pretrained weights | `assets/torch/hub/checkpoints/` | 由 torchvision model weights 机制缓存 |

路径来自配置文件：

```yaml
assets:
  torch_home: assets/torch

data:
  root: data
```

因此默认完整位置是：

```text
flower102_task1/data/flowers-102/
flower102_task1/assets/torch/hub/checkpoints/
```

只下载数据集：

```bash
python scripts/download_assets.py --config configs/baseline_resnet18.yaml --skip_models
```

只下载模型权重：

```bash
python scripts/download_assets.py --config configs/baseline_resnet18.yaml --skip_data --models resnet18 resnet34
```

改数据集目录：

```bash
python scripts/download_assets.py --data_root /path/to/data --models resnet18 resnet34
```

改模型缓存目录：

```bash
python scripts/download_assets.py --torch_home /path/to/torch_cache --models resnet18 resnet34
```

训练和评估脚本也会读取 `assets.torch_home`，所以预下载脚本与训练脚本使用同一个模型缓存位置。若资源已经存在，torchvision 会复用本地文件，不会重复下载。

## Dataset

代码使用 `torchvision.datasets.Flowers102`：

- `split="train"`
- `split="val"`
- `split="test"`
- 类别数固定为 102，标签范围为 `0..101`
- 无需手动整理文件夹

`src/datasets.py` 本身是数据加载模块，不建议直接运行。正确的资源准备入口是：

```bash
python scripts/download_assets.py --config configs/baseline_resnet18.yaml
```

训练脚本内部也会调用同样的数据集构造逻辑；如果你没有提前运行 `download_assets.py`，首次训练时仍可能触发下载。但为了实验流程清晰、便于服务器排错，推荐先单独运行下载脚本。

默认数据增强：

- train: `RandomResizedCrop(224)`, `RandomHorizontalFlip`, optional `ColorJitter`, ImageNet Normalize
- val/test: `Resize(256)`, `CenterCrop(224)`, ImageNet Normalize

如果显存不足，把 YAML 或命令行里的 `batch_size` 改为 16 或 8。

## Training

Baseline:

```bash
cd flower102_task1
python scripts/download_assets.py --config configs/baseline_resnet18.yaml --models resnet18 resnet34
bash scripts/run_baseline.sh
```

单独训练 ResNet-18 baseline:

```bash
python scripts/train.py --config configs/baseline_resnet18.yaml
```

命令行覆盖参数:

```bash
python scripts/train.py \
  --config configs/baseline_resnet18.yaml \
  --experiment.name baseline_resnet18_lr1e4 \
  --train.epochs 50 \
  --train.backbone_lr 5e-5 \
  --train.head_lr 5e-4 \
  --data.batch_size 32
```

也支持常用显式参数：

```bash
python scripts/train.py --config configs/baseline_resnet18.yaml \
  --epochs 50 --batch_size 32 --backbone_lr 5e-5 --head_lr 5e-4 \
  --weight_decay 1e-4 --arch resnet18 --pretrained true --attention none --logger none
```

Pretraining ablation:

```bash
bash scripts/run_ablation.sh
```

Attention comparison:

```bash
bash scripts/run_attention.sh
```

Hyperparameter sweep:

```bash
bash scripts/run_hparam_sweep.sh
```

Extra factorial hyperparameter sweep:

```bash
bash scripts/run_extra_hparam_factorial.sh
```

`run_hparam_sweep.sh` 覆盖报告最初的两组代表性组合：`lr1+30 epoch` 和 `lr2+50 epoch`。为了更清楚地区分“epoch 增加”和“learning rate 变小”的影响，`run_extra_hparam_factorial.sh` 额外补齐四个组合：

- `hparam_resnet18_lr1_epochs50`: ResNet-18, `backbone_lr=1e-4`, `head_lr=1e-3`, `epochs=50`
- `hparam_resnet18_lr2_epochs30`: ResNet-18, `backbone_lr=5e-5`, `head_lr=5e-4`, `epochs=30`
- `hparam_resnet34_lr1_epochs50`: ResNet-34, `backbone_lr=1e-4`, `head_lr=1e-3`, `epochs=50`
- `hparam_resnet34_lr2_epochs30`: ResNet-34, `backbone_lr=5e-5`, `head_lr=5e-4`, `epochs=30`

该脚本通过命令行参数覆盖 `configs/baseline_resnet18.yaml` 和 `configs/baseline_resnet34.yaml` 中的训练设置，不直接修改 YAML 配置文件。脚本结束后会运行 `scripts/analyze_confusions.py` 和 `scripts/summarize_results.py`，用于刷新混淆矩阵分析产物和 `outputs/metrics/results_summary.csv/md`。

### Extended-budget optimization experiments

在完成主实验之后，如果需要继续提高实验设计的公平性和说服力，可以运行可选脚本：

```bash
bash scripts/run_extended_budget.sh
```

该脚本不会被 baseline、ablation、attention 或 hparam sweep 默认调用，只有手动执行时才会启动训练。它针对当前实验中最需要补充的两个问题设计：

1. Scratch 消融实验：主实验中 scratch 与 pretrained 使用接近的训练预算，便于控制变量，但随机初始化模型没有 ImageNet 特征作为起点，通常需要更长训练轮数和更大的 backbone learning rate。因此脚本补充 `extended_scratch_resnet18_epochs100_lr1e3` 和 `extended_scratch_resnet34_epochs100_lr1e3`，设置 `epochs=100`、`backbone_lr=1e-3`、`head_lr=1e-3`。这些结果用于评估 scratch 模型在更充分优化预算下的表现，而不是替代主报告中的同预算消融结论。

2. Attention 对比实验：SE/CBAM 模块是在 ImageNet 预训练 ResNet backbone 上新增的随机初始化模块，可能需要更多 epoch 才能稳定适配 Flowers102。脚本补充 `extended_attention_se_resnet18_epochs50` 和 `extended_attention_cbam_resnet18_epochs50`，设置 `epochs=50`。如果 30 epoch 的 attention 模型未超过 baseline，不能简单断定 attention 机制无效；更严谨的表述应是“在当前插入方式和训练预算下未观察到稳定提升”。

脚本最后会生成：

- `outputs/metrics/extended_budget_summary.csv`
- `outputs/metrics/extended_budget_summary.md`
- `outputs/metrics/extended_budget_only.csv`
- `outputs/metrics/extended_budget_only.md`

其中 `extended_budget_summary.*` 由通用汇总脚本生成，可能包含所有已完成实验；`extended_budget_only.*` 只保留四个 extended-budget 实验，更适合直接写入报告的扩展实验小节。

如果这些扩展实验被实际运行，建议在报告中单独作为 extended-budget results 或附录结果呈现，避免与主实验的同预算对比混淆。

## Evaluation

```bash
python scripts/eval.py \
  --config configs/baseline_resnet18.yaml \
  --checkpoint outputs/checkpoints/baseline_resnet18_pretrained/best.pt \
  --split test
```

## Outputs

每次训练或评估实验会写入独立目录。下面这部分是训练/评估流程的基础输出，由 `scripts/train.py` 和相关 `run_*.sh` 脚本直接生成：

- `outputs/checkpoints/{experiment_name}/best.pt`: best validation accuracy checkpoint
- `outputs/checkpoints/{experiment_name}/last.pt`: latest checkpoint
- `outputs/metrics/{experiment_name}/config.yaml`: 实际运行配置
- `outputs/metrics/{experiment_name}/metrics.csv`: 每个 epoch 的 train/val loss、accuracy、learning rate
- `outputs/metrics/{experiment_name}/test_metrics.json`: best checkpoint 在 test split 上的结果
- `outputs/metrics/{experiment_name}/summary.json`: 汇总字段
- `outputs/figures/{experiment_name}/confusion_matrix.png`: 测试集原始混淆矩阵图
- `outputs/figures/{experiment_name}/confusion_matrix.npy`: 测试集原始混淆矩阵数组，供后处理分析使用
- `outputs/figures/{experiment_name}/training_curves.png`: train/val loss、accuracy 与 learning rate 曲线
- `outputs/metrics/results_summary.csv`: 所有实验汇总
- `outputs/metrics/results_summary.md`: 可直接粘贴进报告的 Markdown 表格

为了避免混淆矩阵只作为装饰图，本项目额外提供 `scripts/analyze_confusions.py` 对已经保存的 `confusion_matrix.npy` 做后处理分析：

```bash
python scripts/analyze_confusions.py --top_k 10
```

该脚本不会重新训练模型，也不会重新评估 checkpoint，只读取 `outputs/figures/{experiment_name}/confusion_matrix.npy` 并生成以下分析产物：

- `outputs/figures/{experiment_name}/normalized_confusion_matrix.png`: 按真实类别归一化后的混淆矩阵，更适合观察各类别相对错误率
- `outputs/figures/{experiment_name}/top_confusion_pairs.csv`: 错误次数最多的类别混淆对，可用于报告中的 top confused pairs 表格
- `outputs/figures/{experiment_name}/per_class_accuracy.csv`: 每个类别的测试样本数、正确数和 per-class accuracy，可用于分析低准确率类别

这些文件属于结果分析补强产物，不是训练脚本最原始的必要输出。如果 clone 仓库后只复现实验训练，缺少这些文件时可以运行上面的分析命令重新生成。

`run_ablation.sh` 会额外生成：

- `outputs/metrics/pretrained_ablation.csv`

`run_attention.sh` 会额外生成：

- `outputs/metrics/attention_comparison.csv`

`run_extra_hparam_factorial.sh` 会额外补齐四个超参数实验目录，并刷新总汇总表：

- `outputs/metrics/hparam_resnet18_lr1_epochs50/`
- `outputs/metrics/hparam_resnet18_lr2_epochs30/`
- `outputs/metrics/hparam_resnet34_lr1_epochs50/`
- `outputs/metrics/hparam_resnet34_lr2_epochs30/`
- `outputs/metrics/results_summary.csv`
- `outputs/metrics/results_summary.md`

`run_extended_budget.sh` 会额外生成：

- `outputs/metrics/extended_budget_summary.csv`
- `outputs/metrics/extended_budget_summary.md`
- `outputs/metrics/extended_budget_only.csv`
- `outputs/metrics/extended_budget_only.md`

## Logging

默认 `logger.type=none`，不需要任何账号即可训练。

本项目当前默认关闭在线实验记录，`logger.type=none`。报告曲线由本地 `metrics.csv` 自动绘制并保存到 `outputs/figures/{experiment_name}/training_curves.png`。如需使用 swanlab，可在 YAML 中设置：

```yaml
logger:
  type: swanlab
```

## Recommended Experiments

最终报告建议至少运行：

| Group | Experiment |
| --- | --- |
| Baseline | ResNet-18 pretrained |
| Baseline | ResNet-34 pretrained |
| Hyperparameter | ResNet-18, backbone_lr=1e-4, head_lr=1e-3, epochs=30 |
| Hyperparameter | ResNet-18, backbone_lr=1e-4, head_lr=1e-3, epochs=50 |
| Hyperparameter | ResNet-18, backbone_lr=5e-5, head_lr=5e-4, epochs=30 |
| Hyperparameter | ResNet-18, backbone_lr=5e-5, head_lr=5e-4, epochs=50 |
| Hyperparameter | ResNet-34, backbone_lr=1e-4, head_lr=1e-3, epochs=30 |
| Hyperparameter | ResNet-34, backbone_lr=1e-4, head_lr=1e-3, epochs=50 |
| Hyperparameter | ResNet-34, backbone_lr=5e-5, head_lr=5e-4, epochs=30 |
| Hyperparameter | ResNet-34, backbone_lr=5e-5, head_lr=5e-4, epochs=50 |
| Pretraining Ablation | ResNet-18 scratch |
| Pretraining Ablation | ResNet-34 scratch |
| Attention | SE-ResNet-18 pretrained |
| Attention Optional | CBAM-ResNet-18 pretrained |
| Extended Optional | Scratch ResNet-18/34, epochs=100, backbone_lr=head_lr=1e-3 |
| Extended Optional | SE/CBAM ResNet-18, epochs=50 |

报告表格推荐字段：

- Method
- Backbone
- Pretrained
- Attention
- Epochs
- Backbone LR
- Head LR
- Best Val Acc
- Test Acc
- Params

## Model Notes

Baseline 使用 torchvision ResNet-18/34，加载 `IMAGENET1K_V1` 权重后替换 `fc` 为 102 类输出。优化器采用差分学习率：backbone 使用较小 learning rate，分类头使用较大 learning rate。

SE-ResNet 保留原始 ResNet 主体结构，在 `layer1/layer2/layer3/layer4` 每个 stage 后插入手写 `SEBlock`。当 `pretrained=true` 时，先加载原始 ImageNet ResNet 权重，再随机初始化新增 SE 模块。新增 attention module 的作用是学习与 Flowers102 任务相关的通道重标定。

CBAM 是可选增强，同样在每个 stage 后插入手写 channel attention 和 spatial attention。

实验结论需要区分“同训练预算下的控制变量对比”和“充分调参后的最优性能”。主实验中的 scratch 与 pretrained 对比保持相近训练轮数和优化器设置，用于观察 ImageNet 预训练在相同预算下的收益；但随机初始化网络通常需要更长 epoch 和更大的 backbone learning rate。类似地，SE/CBAM 的新增模块随机初始化，若训练轮数不足，可能因为优化不充分而未能体现潜在收益。因此后续优化时应优先参考 `scripts/run_extended_budget.sh`，补充 scratch 长训练实验和 attention 长训练实验。

## Model Weights

训练完成后的模型权重位于 `outputs/checkpoints/{experiment_name}/best.pt`，文件较大，不提交到 GitHub。请将所有需要提交的 `best.pt` 上传到百度云、Google Drive 或其他网盘，并在下方填写公开链接。上传时不要包含 `data/` 目录或 Flowers102 数据集压缩包。

- Baidu Netdisk: TBD
- Google Drive: https://drive.google.com/drive/folders/16jzCBtfwesFbTHLzad11ZXdOw51ncH9-?usp=drive_link

完整权重清单和恢复示例见 `MODEL_WEIGHTS.md`。

## Reproducibility

`src/utils.py` 中实现了 `set_seed`，覆盖 `random`、`numpy`、`torch`、`torch.cuda` 与 cuDNN deterministic 设置。训练启动时会打印 device、torch/torchvision 版本、模型、预训练状态、attention 类型、数据集大小、batch size、optimizer 和 learning rate。
