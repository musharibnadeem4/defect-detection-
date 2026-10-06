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
src/defect_detection/   data, transforms, model, train, evaluate, export, utils (stubs until Step 2)
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
