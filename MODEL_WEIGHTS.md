# Model Weights

训练好的模型权重不放入 GitHub 仓库，请上传到百度云、Google Drive 或其他网盘后，将公开链接填写到本文件和 README 中。

## Important

- 不要上传 `data/` 目录。
- 不要上传 `data/flowers-102/102flowers.tgz`。
- 不要上传 `assets/torch/` 中的 ImageNet 预训练缓存。
- GitHub 仓库只提交代码、配置、README、实验指标和报告用图。

## Checkpoints To Upload

| Experiment | Local checkpoint path | Size |
| --- | --- | --- |
| baseline_resnet18_pretrained | `outputs/checkpoints/baseline_resnet18_pretrained/best.pt` | 129 MB |
| baseline_resnet34_pretrained | `outputs/checkpoints/baseline_resnet34_pretrained/best.pt` | 244 MB |
| hparam_resnet18_lr1_epochs30 | `outputs/checkpoints/hparam_resnet18_lr1_epochs30/best.pt` | 129 MB |
| hparam_resnet18_lr2_epochs50 | `outputs/checkpoints/hparam_resnet18_lr2_epochs50/best.pt` | 129 MB |
| hparam_resnet34_lr1_epochs30 | `outputs/checkpoints/hparam_resnet34_lr1_epochs30/best.pt` | 244 MB |
| hparam_resnet34_lr2_epochs50 | `outputs/checkpoints/hparam_resnet34_lr2_epochs50/best.pt` | 244 MB |
| ablation_resnet18_pretrained | `outputs/checkpoints/ablation_resnet18_pretrained/best.pt` | 129 MB |
| ablation_resnet18_scratch | `outputs/checkpoints/ablation_resnet18_scratch/best.pt` | 129 MB |
| ablation_resnet34_pretrained | `outputs/checkpoints/ablation_resnet34_pretrained/best.pt` | 244 MB |
| ablation_resnet34_scratch | `outputs/checkpoints/ablation_resnet34_scratch/best.pt` | 244 MB |
| attention_baseline_resnet18 | `outputs/checkpoints/attention_baseline_resnet18/best.pt` | 129 MB |
| attention_se_resnet18 | `outputs/checkpoints/attention_se_resnet18/best.pt` | 129 MB |
| attention_cbam_resnet18 | `outputs/checkpoints/attention_cbam_resnet18/best.pt` | 129 MB |

## Cloud Link

- Baidu Netdisk: TBD
- Google Drive: https://drive.google.com/drive/folders/16jzCBtfwesFbTHLzad11ZXdOw51ncH9-?usp=drive_link

## Restore Example

下载权重后，将对应 `best.pt` 放回相同目录，例如：

```bash
mkdir -p outputs/checkpoints/baseline_resnet18_pretrained
# 将下载的 best.pt 放到 outputs/checkpoints/baseline_resnet18_pretrained/best.pt
python scripts/eval.py \
  --config configs/baseline_resnet18.yaml \
  --checkpoint outputs/checkpoints/baseline_resnet18_pretrained/best.pt \
  --split test
```
