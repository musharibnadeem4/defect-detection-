# Training runs (validation only)

Seed 42; AdamW, weight decay 0.0001, batch 32; stage 1 (head only) 2 epochs at lr 0.001, stage 2 (all layers) up to 8 epochs at lr 0.0001 with cosine schedule; early stopping on validation PR-AUC (patience 3); CPU only.

Ranking rule: validation PR-AUC (rounded to 0.001), then recall@0.5, then precision at the tuned threshold, then lower validation log-loss at the best epoch.

| run | imbalance handling | size | params (M) | best epoch / run | val PR-AUC | val recall@0.5 | val log-loss (best epoch) | train time (s) |
|---|---|---|---|---|---|---|---|---|
| resnet18_none_224 | none | 224 | 11.18 | 5 / 8 | 1.000 | 1.000 | 0.0071 | 1073.6 |
| resnet18_none_300 | none | 300 | 11.18 | 6 / 9 | 1.000 | 1.000 | 0.0108 | 1399.8 |
| efficientnet_b0_none_224 | none | 224 | 4.01 | 10 / 10 | 1.000 | 1.000 | 0.0166 | 1183.8 |
| resnet18_weighted_loss_224 | weighted_loss | 224 | 11.18 | 5 / 8 | 1.000 | 1.000 | 0.0418 | 1076.1 |
| resnet18_sampler_224 | sampler | 224 | 11.18 | 3 / 6 | 1.000 | 1.000 | 0.2385 | 628.7 |
| logreg_64px | balanced | 64 | 0.00 | - / - | 0.949 | 0.895 | - | - |

5 of 5 neural runs reach validation PR-AUC 1.000 and recall 1.000: validation (105 images, 19 defective) cannot separate them on those metrics, so the imbalance-strategy comparison is **inconclusive**; the winner is decided by the log-loss tie-break only.
