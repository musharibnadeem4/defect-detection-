# Model comparison

> **Stand-in dataset** (Kaggle casting, subsampled to 700 images, ~18% defective). These numbers say nothing about the client's data.

Models were ranked on **validation** only (PR-AUC, then recall@0.5, then precision at the tuned threshold, then lower validation log-loss). Thresholds are chosen on validation (target recall 0.95) and reused unchanged on test. ★ = chosen model. Positive class = `defective`.

## Validation (used for selection)

| run | params (M) | PR-AUC | ROC-AUC | recall@0.5 | prec@0.5 | tuned thr | prec@tuned | recall@tuned | best epoch |
|---|---|---|---|---|---|---|---|---|---|
| resnet18_none_224 ★ | 11.18 | 1.000 | 1.000 | 1.000 | 1.000 | 0.7760 | 1.000 | 1.000 | 5 |
| resnet18_none_300 | 11.18 | 1.000 | 1.000 | 1.000 | 1.000 | 0.6194 | 1.000 | 1.000 | 6 |
| efficientnet_b0_none_224 | 4.01 | 1.000 | 1.000 | 1.000 | 1.000 | 0.6657 | 1.000 | 1.000 | 10 |
| resnet18_weighted_loss_224 | 11.18 | 1.000 | 1.000 | 1.000 | 0.905 | 0.9274 | 1.000 | 1.000 | 5 |
| resnet18_sampler_224 | 11.18 | 1.000 | 1.000 | 1.000 | 0.594 | 0.9836 | 1.000 | 1.000 | 3 |
| logreg_64px | 0.00 | 0.949 | 0.982 | 0.895 | 0.773 | 0.3144 | 0.500 | 1.000 | - |

## Held-out test

| run | PR-AUC | ROC-AUC | recall@0.5 | prec@0.5 | F1@0.5 | recall@tuned | prec@tuned | F1@tuned | latency median / p95 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| resnet18_none_224 ★ | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 50.9 / 57.2 |
| resnet18_none_300 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 87.3 / 95.5 |
| efficientnet_b0_none_224 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 48.6 / 53.3 |
| resnet18_weighted_loss_224 | 1.000 | 1.000 | 1.000 | 0.950 | 0.974 | 1.000 | 1.000 | 1.000 | 54.1 / 83.5 |
| resnet18_sampler_224 | 0.997 | 0.999 | 1.000 | 0.655 | 0.792 | 0.947 | 0.947 | 0.947 | 55.1 / 60.2 |
| logreg_64px | 0.972 | 0.993 | 1.000 | 0.792 | 0.884 | 1.000 | 0.594 | 0.745 | 0.2 / 0.7 |

## Chosen model: `resnet18_none_224` — bootstrap 95% CIs on test (1000 resamples, 19 defective of 105)

| threshold | recall | precision | F1 |
|---|---|---|---|
| 0.5 | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |
| tuned (0.7760) | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |

## 5-fold CV of the chosen configuration (train+val only, 5 epochs/fold = the chosen run's best epoch, same LR schedule, no early stopping)

- OOF PR-AUC 0.988, ROC-AUC 0.993 on 595 images; per-fold PR-AUC 0.963, 1.000, 1.000, 0.998, 1.000 (mean 0.992 ± 0.015)
- OOF @0.5: recall 0.981, precision 0.761; @validation threshold 0.7760: recall 0.981, precision 0.938
- Predictions: `reports/oof_predictions.csv`

## Leakage probe (pixel RMSE, 64x64 grayscale, 0-255 scale)

- test -> train+val nearest neighbour: min 4.51, median 19.38; test images with a train/val image within RMSE 3.0: 0
- val -> train: min 4.79, median 19.81

Latency: forward pass only, batch size 1, CPU (4 torch threads), median/p95 over 100 runs after warm-up; image decoding/resizing excluded.
