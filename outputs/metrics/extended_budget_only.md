| experiment_name | model | pretrained | attention | epochs | batch_size | backbone_lr | head_lr | best_val_acc | test_acc | params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| extended_attention_cbam_resnet18_epochs50 | resnet18 | True | cbam | 50 | 64 | 0.0001 | 0.001 | 93.0392 | 89.3479 | 11272750 |
| extended_scratch_resnet18_epochs100_lr1e3 | resnet18 | False | none | 100 | 64 | 0.001 | 0.001 | 54.902 | 48.8697 | 11228838 |
| extended_attention_se_resnet18_epochs50 | resnet18 | True | se | 50 | 64 | 0.0001 | 0.001 | 92.549 | 89.9658 | 11272358 |
| extended_scratch_resnet34_epochs100_lr1e3 | resnet34 | False | none | 100 | 32 | 0.001 | 0.001 | 54.5098 | 48.9998 | 21336998 |
