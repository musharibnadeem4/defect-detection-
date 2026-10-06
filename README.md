# Visual Defect Detection (normal vs defective)

> ## ⚠️ Stand-in dataset: read this first
> The real client dataset has **not** been provided. All numbers, figures and models in this
> repository come from a **stand-in**: the public Kaggle
> *"Casting product image data for quality inspection"* dataset (grayscale top-view photos of
> a cast impeller, 300x300). It is subsampled to 700 images with ~18% defective to mimic an
> imbalanced production setting. **Results must not be read as performance on the client's
> data.** The code is dataset-agnostic and is meant to be re-run unchanged once real data
> arrives (see "Switching to real data").

## Layout
```
configs/config.yaml     class names, image size, paths, hyper-parameters, stand-in settings
data/{raw,processed,external}   git-ignored; raw/<class_name>/*.jpg|png is the only input contract
scripts/                inspect_dataset.py, prepare_standin_dataset.py, eda.py (+ _common.py)
src/defect_detection/   data, transforms, model, train, metrics, evaluate, utils (export.py: stub until Step 3)
api/  tests/  notebooks/  reports/figures/
```

## Step 1 usage (inspection + stand-in preparation; no training)
```bash
pip install -r requirements.txt            # Python 3.11
# place the Kaggle data at data/external/casting/casting_data/{ok_front,def_front}
python scripts/inspect_dataset.py          # terminal report on the source dataset
python scripts/prepare_standin_dataset.py  # -> data/raw/{normal,defective}/ + data/processed/manifest.csv
python scripts/eda.py                      # -> reports/figures/
```
Defaults (700 images, 18% defective, seed 42) live in `configs/config.yaml` and can be
overridden with `--n`, `--positive-ratio`, `--seed`.

## Duplicates: how they are detected and why
pHash alone proved unreliable on this data: all images show the same part in the same pose, so
pHash (Hamming <= 4) flagged 69% of the images and chained them into one 2,280-image group
mixing both classes. We therefore use pHash only to *propose* candidate pairs and *confirm* each
with a pixel-level check (RMSE <= 3 on 64x64 grayscale thumbnails), then take connected
components. `manifest.csv` carries `duplicate_group_id` for every image (unique id when
there is no duplicate) so the Step 2 split can be group-aware (e.g. `StratifiedGroupKFold`).
Thresholds are in `configs/config.yaml` (`duplicates.*`). Rotated/flipped copies are not checked.

## Switching to real data
Put images in `data/raw/<class_name>/`, set `classes` in `configs/config.yaml` (last = positive
class), and skip `prepare_standin_dataset.py` (the `standin:` block is stand-in only). The
manifest/duplicate groups can be rebuilt by pointing the scan helpers in `scripts/_common.py`
at the new folders.

## Step 2: splitting, training, evaluation
```bash
python scripts/make_splits.py            # data/processed/splits.csv + artifacts/norm_stats.json
python scripts/train.py --phase all      # baseline, imbalance sweep, extra architectures -> runs/<ts>_<exp>/
python scripts/evaluate.py               # test metrics, comparison table, 5-fold CV + OOF predictions
python -m pytest                         # split-leakage, transform, metric and smoke-training tests
```
Everything (sizes, strategies, epochs, thresholds, experiment grid) is in `configs/config.yaml`.

**Protocol (what touches which data)**
- *Split*: 70/15/15, class-stratified and group-aware on `duplicate_group_id` (StratifiedGroupKFold cut
  into 20 folds, folds handed to splits; the best of `splits.n_tries` seeded passes is kept). `cv_fold`
  is a second group-aware partition of train+val only; test rows are never part of CV.
- *Normalisation* mean/std come from the train split only (`artifacts/norm_stats.json`).
- *Train* data fits weights. *Validation* picks the epoch (early stopping on PR-AUC), the imbalance
  strategy, the model, and the decision threshold. *Test* is read only by `evaluate.py`, after the
  model has been chosen from validation metrics.
- *Model ranking*: validation PR-AUC (rounded to 0.001), then recall@0.5, then precision at the tuned
  threshold, then lower validation log-loss (the last tie-break matters because PR-AUC saturates at
  1.0 on a small validation set).
- *Threshold*: highest precision subject to recall >= `threshold.target_recall` on validation. If many
  thresholds tie, the logit-space midpoint of the tied plateau is used rather than the edge. With
  only ~19 validation defects, recall >= 0.95 means catching every one of them (18/19 = 0.947).
- *CV* replays the chosen run (same LR schedule, stopped at its best epoch, no early stopping, no
  checkpoint selection on the held-out fold) so out-of-fold predictions are unbiased.
- *Latency* is the CPU forward pass at batch size 1 (decode/resize excluded), measured by `evaluate.py`.

Training here ran on CPU only (4 cores), which is why epoch budgets are modest
(2 head epochs + up to 8 fine-tuning epochs, patience 3). Results are in `reports/model_comparison.md`;
all of them are on the stand-in dataset.
