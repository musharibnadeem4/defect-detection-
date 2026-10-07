# Visual Defect Detection (normal vs defective)

[![CI](https://github.com/musharibnadeem4/defect-detection-/actions/workflows/ci.yml/badge.svg)](https://github.com/musharibnadeem4/defect-detection-/actions/workflows/ci.yml)

**Jump to:** [TL;DR](#tldr) · [Quickstart](#quickstart) · [API usage](#api-usage) · [Dataset strategy](#dataset-strategy) · [Model approach](#model-approach) · [Evaluation](#evaluation) · [Error analysis](#error-analysis) · [Latency and trade-offs](#latency-and-trade-offs) · [Production considerations](#production-considerations) · [Known limitations](#known-limitations) · [Reproducing training](#reproducing-training) · [Project structure](#project-structure) · [Dataset license note](#dataset-license-note)

## TL;DR

- A binary visual defect detector (ResNet-18 transfer learning, 224 px) served by FastAPI + ONNX Runtime on CPU, with a decision threshold of **0.776** tuned on validation for at least 95% recall.
- **Every result here comes from a stand-in dataset** (public Kaggle casting images, subsampled to 700 images with 18% defective), because the client dataset was not provided. Nothing below is a claim about the client's data.
- On the held-out test set (105 images, 19 defective) it finds 19/19 defects with 0 false alarms, but with only 19 defects the exact 95% interval for recall is **[0.82, 1.00]**. Cross-validation on 595 images is the more honest number: recall 0.981 and precision 0.938 at the deployed threshold (applied to five differently calibrated fold models, not the deployed model's own precision).
- Main weaknesses found: calibration shifts between training runs, two defects missed with high confidence, and brittleness to noise, blur and brighter images ([error analysis](reports/error_analysis.md)).
- The Docker image is built and smoke-tested by the [CI workflow](.github/workflows/ci.yml) (the badge shows that workflow's current status; this README does not claim it is passing), **and** it was built and run locally with Docker 29.8.2: 837 MB on disk (219 MB content size, compressed), container healthy, running as the non-root `appuser`, 284.9 s for a cold uncached build. Details and the curl smoke tests: [reports/docker_local_check.md](reports/docker_local_check.md).

## Quickstart

```bash
# A. With Docker (artifacts/model.onnx + model_meta.json are committed in this repo)
docker compose up --build
curl -s http://localhost:8000/ready

# B. Without Docker (Python 3.11, inference dependencies only, no torch)
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-inference.txt
python tasks.py serve                                     # http://127.0.0.1:8000
curl -s -X POST http://127.0.0.1:8000/predict -F "file=@examples/images/01_defect_correct_cast_def_0_7896.jpeg;type=image/jpeg"

# C. Development: training, evaluation, all tests
pip install -r requirements.txt
python tasks.py test
```
`tasks.py` is the task runner (`setup data train evaluate analysis export test serve docker-build docker-run examples all`). The
`Makefile` only delegates to it and has not been run (make is not installed on the dev machine).

## API usage

```bash
curl -s http://localhost:8000/health        # liveness
curl -s http://localhost:8000/ready         # 200 only when the model unit is verified and a warm-up inference passed
curl -s http://localhost:8000/model-info    # contents of model_meta.json
curl -s -X POST http://localhost:8000/predict -F "file=@examples/images/01_defect_correct_cast_def_0_7896.jpeg;type=image/jpeg"
curl -s -X POST http://localhost:8000/predict -F "file=@examples/images/14_perturbed_blur_sigma2_cast_def_0_7896.png;type=image/png"   # quality warning
curl -s -X POST http://localhost:8000/predict -F "file=@README.md;type=image/jpeg"                                                      # 400
```
More commands: [examples/curl_commands.md](examples/curl_commands.md). `POST /predict/batch` (field `files`, up to 8 images) returns one result or error per image.

Real response (first row of [examples/predictions.json](examples/predictions.json)):

```json
{"predicted_class": "defective", "confidence": 0.9999651908874512, "defect_probability": 0.9999651908874512,
 "threshold_used": 0.7760080151863646, "model_version": "20261007-004422_resnet18_none_224-0f824ffa",
 "latency_ms": 29.0, "request_id": "bd2a68a156804e8c8ee45d5bdfd3a1d1",
 "input_quality": {"ok": true, "warnings": [], "metrics": {"mean_brightness": 140.8, "sharpness": 162.21}}}
```

| field | meaning |
|---|---|
| `predicted_class` | `defective` if `defect_probability >= threshold_used`, else `normal` |
| `defect_probability` | softmax probability of the `defective` class. Use this one for ranking and thresholds |
| `confidence` | softmax probability of the **predicted** class (`defect_probability` if defective, otherwise `1 - defect_probability`) |
| `threshold_used` | the deployed threshold, 0.776 (from `model_meta.json`), not 0.5 |
| `model_version` | `<training run id>-<sha256 prefix of model.onnx>` |
| `latency_ms` | server time for decode + preprocess + quality check + inference (not network or upload) |
| `request_id` | also returned as the `X-Request-ID` header; a safe client-supplied one is honoured |
| `input_quality` | `ok`, `warnings`, and the measured brightness/sharpness. Informational only, see [Production considerations](#production-considerations) |

**Because the threshold is 0.776, `confidence` can be below 0.5.** Worked example of the rule (arithmetic, not a measured case): an image with
`defect_probability` 0.60 is below 0.776, so it is `normal`, and its `confidence` is 1 - 0.60 = 0.40. A `normal` answer with a low
`confidence` therefore means "close to the line", not "probably defective".

**Errors** all share the body `{"error": {"code", "message", "request_id"}}` (no stack traces):

| status | code | when |
|---|---|---|
| 400 | `empty_file`, `invalid_image` | empty upload; not decodable, corrupt or truncated image |
| 413 | `file_too_large`, `too_many_files` | file over `DEFECT_MAX_UPLOAD_BYTES`; batch over `DEFECT_BATCH_MAX_IMAGES` |
| 415 | `unsupported_media_type` | content type not in `DEFECT_ALLOWED_CONTENT_TYPES` |
| 422 | `image_too_small`, `image_too_many_pixels`, `unsupported_pixel_format`, `validation_error` | min side below `DEFECT_MIN_IMAGE_SIDE`; pixel count over `DEFECT_MAX_IMAGE_PIXELS` (checked from the header, before decoding: decompression-bomb guard); 16-bit/float images (rejected, not silently clipped); missing `file` field |
| 500 | `internal_error` | unexpected server error (details only in the server log) |
| 503 | `not_ready` | artifacts missing, or `model.onnx` does not match the sha256 in `model_meta.json` |

Grayscale, RGB, RGBA and palette images are accepted and converted exactly as in training (`convert("RGB")`, alpha dropped, no EXIF rotation).

**Settings** are environment variables (prefix `DEFECT_`, defaults in [api/settings.py](api/settings.py)):

| variable | default | meaning |
|---|---|---|
| `DEFECT_ARTIFACTS_DIR` | `<repo>/artifacts` (`/app/artifacts` in Docker) | directory with `model.onnx` + `model_meta.json` |
| `DEFECT_MAX_UPLOAD_BYTES` | 5242880 | per-file limit |
| `DEFECT_MIN_IMAGE_SIDE` | 64 | smaller min(width, height) is rejected |
| `DEFECT_MAX_IMAGE_PIXELS` | 25000000 | decompression-bomb guard |
| `DEFECT_ALLOWED_CONTENT_TYPES` | `image/jpeg,image/png,image/bmp,image/tiff` | comma separated |
| `DEFECT_BATCH_MAX_IMAGES` | 8 | limit for `/predict/batch` |
| `DEFECT_ORT_INTRA_OP_THREADS`, `DEFECT_ORT_INTER_OP_THREADS` | 0, 0 (ONNX Runtime default) | ORT thread pools |
| `DEFECT_WARMUP_RUNS` | 2 | warm-up inferences at startup |
| `DEFECT_LOG_LEVEL` | INFO | JSON logs on stdout |

`docker-compose.yml` sets the ONNX Runtime threads to intra-op 2 and inter-op 1, the values the container log showed ([reports/docker_local_check.md](reports/docker_local_check.md)), matching the 2 physical cores benchmarked in [reports/latency.md](reports/latency.md). The bare-process default is 0 / 0 (ONNX Runtime chooses).

## Dataset strategy

Source: the Kaggle "Casting product image data for quality inspection" images: 6,000 JPEGs (3,000 `ok_front`, 3,000 `def_front`), all
300x300, **100% grayscale stored as RGB**, 0 corrupt, no train/test split provided ([reports/dataset_inspection.txt](reports/dataset_inspection.txt)).

**Subsample.** To mimic an imbalanced production setting, 700 images are drawn with a fixed seed (42): 574 normal, 126 defective = 18.0% defective
([reports/dataset_summary.json](reports/dataset_summary.json)).

**Duplicates: pHash alone failed here.** All images show the same part in the same pose, so perceptual hashes of different parts are close. At
Hamming distance <= 4, pHash flagged 4,166 of the 6,000 images (69%) and chained them into one 2,280-image group; 30 groups mixed both classes. The method used
instead: pHash proposes candidate pairs (distance <= 8), a pixel RMSE check on 64x64 thumbnails (<= 3.0) confirms them, and groups are the
connected components.

| setting | groups | images in groups | largest group | groups mixing both classes |
|---|---|---|---|---|
| pHash only, Hamming <= 4 | 480 | 4,166 | 2,280 | 30 |
| pHash only, Hamming <= 8 | 31 | 5,799 | 5,712 | 2 |
| **pHash <= 8 + RMSE <= 3 (used)** | **51** | **103 (1.7%)** | **3** | **0** |

No byte-identical duplicates exist. The threshold sweep (RMSE <= 1, 2, 5) is in the same report; the distances form a continuum, so 3.0 is a judgement call.
Only 11 of the 700 subsampled images belong to a source duplicate group and none has its partner inside the subsample, so the group-aware
split could not be exercised on real duplicates here; it is unit-tested with synthetic groups ([tests/test_splits.py](tests/test_splits.py)).

**Group-aware split.** `StratifiedGroupKFold` cuts the data into 20 folds, folds are assigned to train/val/test (70/15/15), the best of 10 seeded
passes is kept, and the 5 CV folds are a second group-aware partition of train+val only (the test set is never part of CV). No group straddles a split.

| split | normal | defective | defective % |
|---|---|---|---|
| train | 402 | 88 | 18.0 |
| val | 86 | 19 | 18.1 |
| test | 86 | 19 | 18.1 |

Normalisation statistics come from the train split only. Test data is read only by the evaluation, parity and robustness steps.

## Model approach

**Why ResNet-18 transfer learning.** Only 490 training images, so ImageNet features are the sensible starting point; ResNet-18 (11.2M parameters) fine-tunes
on CPU and exports cleanly to ONNX. Training is two-stage: head only (2 epochs, lr 1e-3), then all layers (up to 8 epochs, lr 1e-4, cosine
schedule), AdamW, early stopping on validation PR-AUC. Augmentation: flips, +-15 degree rotation, mild brightness/contrast jitter; no hue/saturation jitter, no crops
(they could cut off edge defects). Details: [reports/training_runs.md](reports/training_runs.md).

| run | params (M) | val PR-AUC | test PR-AUC | test recall / precision at the run's own validation-tuned threshold |
|---|---|---|---|---|
| **resnet18, no handling, 224 px (chosen)** | 11.18 | 1.000 | 1.000 | 1.000 / 1.000 |
| resnet18, no handling, 300 px | 11.18 | 1.000 | 1.000 | 1.000 / 1.000 |
| efficientnet_b0, no handling, 224 px | 4.01 | 1.000 | 1.000 | 1.000 / 1.000 |
| resnet18, weighted loss, 224 px | 11.18 | 1.000 | 1.000 | 1.000 / 1.000 |
| resnet18, balanced sampler, 224 px | 11.18 | 1.000 | 0.997 | 0.947 / 0.947 |
| logistic regression on 64x64 pixels (baseline) | <0.01 | 0.949 | 0.972 | 1.000 / 0.594 |

Source: [reports/model_comparison.md](reports/model_comparison.md). The pixel baseline already reaches 0.972 test PR-AUC, which tells you how easy this stand-in is.

**Class-imbalance handling: inconclusive.** I compared no handling, weighted cross-entropy and a weighted sampler. All three reach validation PR-AUC 1.000 and
recall 1.000 on 105 validation images (19 defective), so validation cannot rank them. "No handling" was chosen by the last tie-break only (validation
log-loss 0.0071 vs 0.0418 for weighted loss and 0.2385 for the sampler). The one visible difference is calibration, not ranking: at threshold 0.5 test
precision is 1.000 (none), 0.950 (weighted) and 0.655 (sampler). The three-way tie also means ResNet-18 @224 beat the 300 px and EfficientNet runs on that tie-break
plus lower cost, not on accuracy. The imbalance here is only 4.6:1; nothing in this project says how the strategies behave at 50:1.

## Evaluation

All thresholds are chosen on validation only; the test set is read once afterwards. Recall and precision use exact Clopper-Pearson 95% intervals;
F1 and AUCs use a bootstrap, which is marked *degenerate* when it collapses to a zero-width interval.

**Held-out test** (105 images, 19 defective; [reports/model_comparison.md](reports/model_comparison.md), [reports/classification_report.txt](reports/classification_report.txt)):

| threshold | TP / FP / FN / TN | recall [95% CI] | precision [95% CI] | F1, PR-AUC, ROC-AUC |
|---|---|---|---|---|
| 0.776 (deployed) | 19 / 0 / 0 / 86 | 1.000 [0.82, 1.00] | 1.000 [0.82, 1.00] | 1.000 each, bootstrap degenerate |
| 0.5 | 19 / 0 / 0 / 86 | 1.000 [0.82, 1.00] | 1.000 [0.82, 1.00] | 1.000 each, bootstrap degenerate |

![test confusion matrices](reports/figures/confusion_matrix.png)

**Pooled out-of-fold (5-fold CV on train+val, 595 images, 107 defective;** [reports/error_analysis.md](reports/error_analysis.md), [reports/candidate_thresholds.csv](reports/candidate_thresholds.csv)):

| threshold | TP / FP / FN / TN | recall [95% CI] | precision [95% CI] |
|---|---|---|---|
| 0.776 (deployed) | 105 / 7 / 2 / 481 | 0.981 [0.934, 0.998] | 0.938 [0.875, 0.975] |
| 0.5 | 105 / 33 / 2 / 455 | 0.981 [0.934, 0.998] | 0.761 [0.681, 0.829] |

Pooled OOF PR-AUC 0.988 [0.971, 1.000], ROC-AUC 0.993 [0.982, 1.000]; per-fold PR-AUC 0.963, 1.000, 1.000, 0.998, 1.000 ([reports/cv_summary.json](reports/cv_summary.json)).

**Read the 0.776 row with care:** it is the deployed threshold applied to predictions from five differently calibrated fold models, not the deployed model's own precision.

**CV and test disagree, and I trust CV more.** Test precision at 0.5 is 1.000 but OOF precision at 0.5 is 0.761. The error analysis traces most of that gap to one CV
fold (25 of the 33 false positives are in fold 2): its model has a normal-class score offset of about +2.65 logits while its ranking is still perfect (PR-AUC 1.000).
The test set has only 19 defects and 86 normals, so a perfect score is compatible with a true recall as low as 0.82, and the deployed model's epoch was selected on
validation. Leakage check: no test image is within the duplicate distance of any train/val image (nearest pixel RMSE 4.51; 0 images within 3.0).

## Error analysis

Full write-up with figures, Grad-CAM, calibration, threshold study and robustness: **[reports/error_analysis.md](reports/error_analysis.md)**. Top findings:

1. **Calibration is unstable between training runs.** Fold 2's normals score a median 2.65 logits higher than the other folds' (-1.60 vs -4.25) although its PR-AUC is 1.000, and that fold model also puts 95 of its own 390 training normals above 0.5. A fixed threshold is not safe across retrains.
2. **Two defects are missed with high confidence** (`cast_def_0_6988`, `cast_def_0_2949`). I could not see a defect in either by eye; Grad-CAM does not land on anything identifiable. The deployed model still calls both normal even though it trained on them ([examples/predictions.md](examples/predictions.md)). Cause unknown (hypothesis: very subtle defect or label problem).
3. **Brittle to noise, blur and brighter images** (test set, fixed thresholds): noise sigma 25 drops precision to 0.22 at 0.5 (0.38 at 0.776); blur sigma 2 to 0.50 (0.72); brightness x1.3 is the only perturbation causing misses (recall 0.842 at 0.776). Rotations by 90/180 degrees and darker or lower-contrast images cause no errors.

Temperature scaling could not fix the fold-2 offset (it cannot move the p = 0.5 boundary, and its fit on the perfectly separable validation set is degenerate).

## Latency and trade-offs

Hardware: Intel i5-7200U (2 physical / 4 logical cores), 15.9 GB RAM, no GPU, Microsoft Windows 11 Pro (10.0.22621). Client and server shared the same cores. Full tables: [reports/latency.md](reports/latency.md).

| threads | PyTorch eager median / p95 (ms) | ONNX Runtime median / p95 (ms) | ORT end-to-end through HTTP median / p95 (ms) |
|---|---|---|---|
| 1 | 79.9 / 87.7 | 43.7 / 59.1 | 51.6 / 54.7 |
| 2 | 55.9 / 63.9 | 27.3 / 34.5 | 37.7 / 44.9 |
| 4 | 55.3 / 64.8 | 26.2 / 36.5 | 36.3 / 87.7 |

- ONNX Runtime is 1.8-2.1x faster than PyTorch eager at batch size 1; HTTP, upload and JSON add about 4 ms. One worker handles about 36-38 requests/s at 1-2 ORT threads (30 at 4 threads: more ORT threads than physical cores hurt).
- Cold start is about 2.5 s from process start to `/ready` for a bare process (benchmark, [reports/latency.md](reports/latency.md)). Inside the container, the application logged `startup` and `ready` events about 0.93 s apart (model load and warm-up only, in an already-running container; [reports/docker_local_check.md](reports/docker_local_check.md)). That is **not** the total container cold start (container creation, process start and imports are not included), which was not measured.
- Image size (`docker images`, Docker 29.8.2, local build): 837 MB disk usage (unpacked), 219 MB content size (compressed). A cold uncached build took 284.9 s.
- **INT8 was rejected.** Dynamic quantization cannot run on this CPU provider (no `ConvInteger` kernel). Static QDQ INT8 is 4x smaller (11.2 vs 44.7 MB) and about 1.5x faster, but logits move by up to 4.08 and probabilities by up to 0.85, 2 of the 105 validation decisions flip at the fixed deployed threshold, and validation precision drops from 1.000 to 0.905 ([reports/quantization.json](reports/quantization.json)). With a threshold that is already fragile to score shifts, that speed-up is not worth re-tuning and re-validating it.
- Resize matters for parity: the inference code uses PIL bilinear like training; `cv2.resize` would shift probabilities by up to 0.19 (INTER_LINEAR) or 0.43 (INTER_CUBIC) ([reports/preprocess_equivalence.json](reports/preprocess_equivalence.json)). PyTorch and ONNX logits agree to 3.2e-05 and all 105 test decisions match ([reports/export_parity.json](reports/export_parity.json)).

## Production considerations

- **Versioned model unit.** `model.onnx`, the threshold, normalisation statistics, class names, input-quality ranges, git commit, training run id and the sha256 of `model.onnx` live together in `model_meta.json`. The service reads everything from it and refuses to become ready if the sha256 does not match. Changing the threshold means re-exporting (new `model_version`). The committed `model_meta.json` records git commit `a2a8418` with `dirty: true`: the model was exported from an uncommitted working tree, so that commit does not identify the exact source; the sha256 in `model_meta.json` is the authoritative identifier. Architecture: [docs/architecture.md](docs/architecture.md).
- **Logging.** One JSON object per request on stdout: request id, method, path, status, latency, predicted class, probability, quality warnings, error code. Image bytes are never logged; startup logs the config and model version.
- **Input-quality guard and its limits.** Mean brightness and Laplacian-variance sharpness are compared with the 1st-99th percentile range of the 490 training images; outside it you get `input_quality.ok = false` and a warning ending "input outside the training range; prediction may be less reliable". It never changes the prediction. It is a coarse risk flag: it warns on 3.9% of training, 4.8% of validation and 7.6% of test images that are normal and correctly classified, and it does **not** flag the two known missed defects ([reports/quality_check_false_alarm_rate.json](reports/quality_check_false_alarm_rate.json)). Behaviour on resolutions other than 300x300 was not evaluated.
- **Monitoring and drift plan (not implemented).** Track, against a baseline taken from validation: the distribution of `defect_probability` (mean, quantiles, fraction above the threshold), the image statistics (brightness, sharpness, fraction with quality warnings, input resolution), error rates and inspector overrides. Alert on shifts in the *normal-class score distribution*: the error analysis found that failure mode (a score offset) in both the CV folds and the perturbation tests, while ranking mostly survived.
- **Retraining and threshold re-validation (procedure, not yet exercised).** Trigger on a camera, lighting or process change, a drift alert, a new defect type, or any retrain. Then: (1) collect and label new data, review disagreements with inspectors; (2) rebuild the manifest and group-aware split; (3) retrain, ideally with several seeds, and look at the spread of calibration, not just PR-AUC; (4) pick the epoch and the threshold on validation (target recall with an exact lower bound), never on test; (5) evaluate once on a fresh test set and compare with the current model; (6) run the parity check and the test suite, export a new versioned unit; (7) roll out in shadow mode first; the previous unit (identified by its sha256) is the rollback.
- **Service limits.** Single worker, no authentication, TLS or rate limiting (put it behind a gateway). Upload size is limited by reading at most limit + 1 bytes, and a reverse proxy should cap request size as well.

## Known limitations

- **Stand-in data:** clean, uniform acquisition, 18% prevalence is artificial, and a pixel logistic regression already scores 0.972 test PR-AUC. Real data will probably be harder.
- **Small samples:** 19 defects on test (exact recall lower bound 0.82), 107 in CV. The model comparison and the imbalance comparison are not decisive.
- **Unstable calibration** across training runs; the threshold must be re-validated after any retrain.
- **Two defects the model cannot see** and no defect annotations, so Grad-CAM (7x7 map) could not be scored and per-defect-type recall is unknown.
- **Sensitive to noise, blur and brightness x1.3**; the quality guard is only a coarse flag.
- **Group-aware splitting** was not exercised on real duplicates (none fall inside the subsample).
- **Docker** is verified by the CI workflow and by manual curl smoke tests of a locally built and run container ([reports/docker_local_check.md](reports/docker_local_check.md)). The CI workflow runs the torch-free subset of the tests (`tests/api`, preprocessing; the two torch-only checks skip) on a Python 3.11 runner, not inside the container; the same subset passed locally in a Windows 3.11 virtualenv (50 passed, 2 skipped). The complete 88-test suite needs torch and ran on Python 3.12. No pytest run happened inside the Linux image. Total container cold start, memory use and behaviour under load in the container were not measured.
- **Dataset licence** not verified (below).

## Reproducing training

Target Python 3.11, but training, evaluation and analysis here were run on **Python 3.12.0**; `requirements.txt` has not been installed or run on 3.11 (only the inference stack has). CPU only (4 threads; one run takes about 10-23 minutes, [reports/training_runs.md](reports/training_runs.md)). Seeds: project seed 42 (subsample, splits, training); CV fold k uses seed 42 + k; deterministic algorithms are on.

```bash
pip install -r requirements.txt
# put the Kaggle images in data/external/casting/casting_data/{ok_front,def_front}
python scripts/inspect_dataset.py > reports/dataset_inspection.txt   # optional, Step 1 report
python tasks.py data        # subsample (700, 18%, seed 42) + manifest + group-aware splits + normalisation stats
python tasks.py train       # baseline, imbalance sweep, extra architectures
python tasks.py evaluate    # test evaluation, comparison table, 5-fold CV (the slow part)
python tasks.py analysis    # fold models, error analysis, Grad-CAM, robustness
python tasks.py export      # artifacts/model.onnx + model_meta.json + PyTorch/ONNX parity check
python tasks.py test
```
Reproducibility check: re-training the 5 CV folds with identical seeds reproduced the earlier out-of-fold probabilities to 2.9e-08 ([reports/oof_reproduction_check.json](reports/oof_reproduction_check.json)). Everything (class names, sizes, paths, hyperparameters) is in [configs/config.yaml](configs/config.yaml).
`artifacts/best.pt` and the other training outputs are not versioned; `python tasks.py all` regenerates them.

## Project structure

```
configs/config.yaml        single source of truth (classes, image size, paths, hyperparameters, thresholds)
src/defect_detection/      data, transforms, model, train, metrics, evaluate, error_analysis, export, preprocess (torch-free), utils
api/                       FastAPI service: main, service, settings, schemas, logging_config
scripts/                   dataset inspection/EDA, splits, training, evaluation, analysis, export, benchmarks, examples, CI response check
tests/                     unit, split-leakage, transform, metric, parity and API tests (tests/api)
reports/                   all measured results (tables, CSV/JSON, figures)
docs/                      architecture (+ diagram), video script, interview prep, submission checklist
examples/                  16 example images, predictions, curl commands
artifacts/                 model.onnx + model_meta.json (the only versioned artifacts; the rest is git-ignored)
.github/workflows/ci.yml   inference tests on Python 3.11, Docker build + container smoke test
Dockerfile, docker-compose.yml, requirements.txt (training/dev), requirements-inference.txt, tasks.py, Makefile
```
Exploratory data analysis is `scripts/eda.py` (figures in `reports/figures/`), not a notebook.

More documentation: [docs/architecture.md](docs/architecture.md) · [docs/video_script.md](docs/video_script.md) · [docs/interview_prep.md](docs/interview_prep.md) · [docs/submission_checklist.md](docs/submission_checklist.md).

## Dataset license note

The images are the Kaggle "Casting product image data for quality inspection" dataset. **I have not verified its licence or redistribution terms.**
They are used only as a stand-in. The 16 images in `examples/images/` are samples of it: check the terms before publishing the repository, and delete that
folder if they do not allow redistribution (the API tests that use it will then fail in CI strict mode or skip locally). The model weights were trained on that
data, so the same question applies to `artifacts/model.onnx`. The [MIT licence](LICENSE) covers the code in this repository only, not the data or anything derived from it.
