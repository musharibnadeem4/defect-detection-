# Submission checklist

> **Read this first.** The assessment brief (`docs/assessment.pdf`) was never present in the workspace, so I could not read it. This list maps the requirements as they
> were given to me step by step, **not** the PDF. Please check it against the PDF yourself. "Verified" says how; "Unverified" says why.

## Status legend
- **Verified**: I ran it and the output is in the repo or was checked in this session.
- **Unverified**: written but not run, or run only partly. The reason is stated.

## Step 1: scaffold and dataset inspection

| requirement | where | status |
|---|---|---|
| Dataset inspection (structure, counts, formats, resolution, channels, corrupt files, duplicates, label conflicts) | `scripts/inspect_dataset.py`, `reports/dataset_inspection.txt` | Verified |
| Stand-in preparation: class mapping, 700 images, 18% defective, fixed seed, manifest with `duplicate_group_id` | `scripts/prepare_standin_dataset.py`, `reports/dataset_summary.json` (`data/` itself is git-ignored) | Verified |
| EDA: class distribution, resolution, 8+8 samples, duplicate report, mean/std | `scripts/eda.py`, `reports/figures/` | Verified (script, not a notebook; the empty `notebooks/` folder was removed) |
| Project structure, config-driven, dataset-agnostic (class names/sizes/paths from `configs/config.yaml`) | `configs/config.yaml`, README "Project structure" | Verified |
| Stand-in disclosed everywhere | README TL;DR, every report header | Verified |
| Pinned requirements, Python 3.11 | `requirements.txt`, `requirements-inference.txt` | **Partly unverified**: the inference stack ran on Python 3.11.2; training, evaluation and analysis ran only on **Python 3.12.0**, and `requirements.txt` (torch, scikit-learn, ...) was never installed on 3.11 |
| git initialised, data and weights ignored | `.gitignore` | Verified (no data, no `.pt`; only `artifacts/model.onnx` and `artifacts/model_meta.json` are allowlisted) |

## Step 2: splitting, training, evaluation

| requirement | where | status |
|---|---|---|
| Group-aware stratified 70/15/15 split + 5-fold CV ids, unit test for leakage | `src/defect_detection/data.py`, `scripts/make_splits.py`, `tests/test_splits.py` | Verified (leakage unit-tested on synthetic groups; not exercised on real duplicates, none fall in the subsample) |
| Transforms (no hue, no crops), train-only normalisation | `src/defect_detection/transforms.py`, `tests/test_transforms.py` | Verified |
| Imbalance comparison (none / weighted loss / sampler) | `reports/training_runs.md` | Verified, **result inconclusive** (all tie on validation) |
| Models: pixel baseline, ResNet-18, EfficientNet-B0; two-stage fine-tuning | `src/defect_detection/model.py`, `train.py`, `reports/training_runs.md` | Verified |
| Threshold chosen on validation only | `src/defect_detection/metrics.py`, `reports/model_comparison.md` | Verified |
| Test evaluation, CIs, confusion matrix, CV with out-of-fold predictions, latency, comparison table | `reports/model_comparison.md`, `reports/oof_predictions.csv`, `reports/figures/confusion_matrix.png` | Verified |

## Step 3: error analysis, calibration, robustness

| requirement | where | status |
|---|---|---|
| Error inventory, galleries, per-fold confusion and score distributions, fold-2 investigation | `reports/errors.csv`, `reports/error_analysis.md`, `reports/figures/` | Verified |
| Grad-CAM on errors and correct detections | `reports/figures/gradcam/`, `reports/gradcam_summary.csv` | Verified (descriptions are subjective; there are no defect masks to score against) |
| Image-property analysis | `reports/error_vs_correct_features.csv` | Verified; **inconclusive** (only 2 false negatives) |
| Threshold and calibration study (OOF only), temperature scaling | `reports/candidate_thresholds.csv`, `reports/error_analysis.md` | Verified (temperature fit is degenerate; stated) |
| Exact Clopper-Pearson intervals | `src/defect_detection/metrics.py`, `tests/test_metrics.py` | Verified |
| Robustness on the test set, read-only | `reports/robustness.csv`, `reports/figures/robustness.png` | Verified |

## Step 4: ONNX, API, Docker, latency, examples

| requirement | where | status |
|---|---|---|
| ONNX export (dynamic batch, pinned opset) + `model_meta.json` (model, threshold, norm stats, sha256, quality ranges) | `src/defect_detection/export.py`, `artifacts/model_meta.json` | Verified. Note: the recorded git state is `dirty: true` at commit `a2a8418`; re-export after committing for a clean record |
| Parity test (script + pytest) and preprocessing equivalence incl. PIL vs cv2 | `scripts/export_onnx.py`, `tests/test_export_parity.py`, `reports/export_parity.json`, `reports/preprocess_equivalence.json` | Verified (105/105 decisions identical, max logit diff 3.2e-05, tensors bit-identical) |
| FastAPI: `/predict`, `/health`, `/ready`, `/model-info`, `/predict/batch`, validation, error codes, quality check, JSON logs, env settings | `api/`, README "API usage" | Verified (38 API tests; real curl against the server on Python 3.12 and 3.11) |
| API tests | `tests/api/` | Verified (also run in a clean Python 3.11 venv without torch) |
| Dockerfile (multi-stage, 3.11-slim, non-root, HEALTHCHECK), `.dockerignore`, compose, separate requirements | `Dockerfile`, `.dockerignore`, `docker-compose.yml`, `requirements-inference.txt` | **Unverified**: Docker is not installed on the dev machine. Compose parses; the image was never built or run locally; image size and container cold start are unmeasured; `libgomp1` in the Dockerfile is an untested precaution |
| Latency report incl. PyTorch vs ONNX Runtime, end-to-end HTTP, concurrency, INT8 | `reports/latency.md`, `reports/quantization.json` | Verified (one machine: i5-7200U, 2 cores / 4 threads; INT8 rejected) |
| Example predictions, curl commands | `examples/` | Verified (16 images; the licence of the images is unverified) |
| Task runner | `tasks.py`, `Makefile` | **Partly verified**: `python tasks.py test` and `python tasks.py serve` were run through the runner; the other targets were run as direct script calls, not through the runner (`export` and `data` would overwrite artifacts, so I did not re-run them). The `Makefile` was never run (no `make` on the machine) |

## Step 5: CI, documentation, polish

| requirement | where | status |
|---|---|---|
| `python-multipart` in inference requirements | `requirements-inference.txt` | Verified (it was added in Step 4; FastAPI needs it for uploads) |
| CI: inference tests on 3.11, Docker build, container run, `/ready` wait, `/health`, `/predict` schema check, image size | `.github/workflows/ci.yml`, `scripts/check_predict_response.py` | **Unverified on GitHub: the workflow has not run.** The YAML parses; the test job's commands passed locally in a clean Python 3.11 venv from an export of only the tracked files (50 passed, 2 skipped); the container job's HTTP steps passed against a server started from that export (not in Docker); strict mode was shown to fail when the model is missing. The Docker build/run steps themselves are untested |
| CI badge | README top | Points at `musharibnadeem4/defect-detection-` (from the `origin` remote); it shows nothing meaningful until the workflow runs |
| README structured around the criteria | `README.md` | Verified (every number checked against a file in `reports/` or `examples/`) |
| Architecture diagram: Mermaid + rendered image | `docs/architecture.md`, `docs/architecture.mmd`, `.png`, `.svg` | Verified (rendered with mermaid-cli 11.4.2 and inspected) |
| Video script (2:30-3:00) | `docs/video_script.md` | Verified; brief's segment lengths summed to 3:20, so five segments were shortened (total 2:55) |
| Interview prep (15 Q&A) | `docs/interview_prep.md` | Written from repo evidence; hypotheses and untested items are marked |
| LICENSE for the code | `LICENSE` | MIT; **holder is the GitHub handle `musharibnadeem4` from the repo remote, not a legal name: edit if needed** |
| .gitignore allowlist, no absolute Windows paths, dead code removed, full suite | see below | Verified |

## Final checks run
- Full test suite (`python tasks.py test`, Python 3.12.0, torch): **88 passed**.
- Torch-free subset in a clean Python 3.11.2 venv (`tests/api`, `tests/test_preprocess.py`, `tests/test_export_parity.py`): **50 passed, 2 skipped** (the two torch-only checks).
- `pyflakes` clean on `src api scripts tests tasks.py`; `vulture` hits were reviewed (remaining ones are FastAPI route handlers and pydantic fields).
- No absolute Windows or user paths in any tracked text file, the SVG, or the ONNX metadata (`git grep`).
- Nothing tracked under `data/` besides `.gitkeep`; no `.pt`/`.pth`/`.ckpt`/`.joblib` tracked.

## Things to do before submitting
1. Push and let the CI workflow run; **fix whatever it reports** (Docker build, `libgomp1`/OpenCV system libraries, healthcheck timing are the likeliest places).
2. Decide on the dataset licence question for `examples/images/` and `artifacts/model.onnx` (44.7 MB committed; consider Git LFS).
3. Check the brief (PDF) against this list.
4. Re-export after the final commit if you want `model_meta.json` to record a clean git state.
5. Check the LICENSE holder name.
