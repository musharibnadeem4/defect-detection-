# Architecture

Two flows share one artifact: the **offline** pipeline builds a versioned model unit (`model.onnx` + `model_meta.json`), the
**online** service loads exactly that unit. The dashed boxes and arrows are **future work, not implemented**.

![architecture](architecture.png)

*(The PNG was rendered from the Mermaid diagram below.)*

## Mermaid source

GitHub renders this block natively:

```mermaid
flowchart LR
  subgraph OFF["OFFLINE: build the model unit"]
    direction TB
    A["Raw images<br/>data/raw/CLASS/*.jpg|png"] --> B["Dedup + manifest<br/>pHash candidates + pixel-RMSE check<br/>duplicate_group_id"]
    B --> C["Group-aware split<br/>70/15/15 + 5-fold CV<br/>StratifiedGroupKFold"]
    C --> D["Train<br/>ResNet-18 transfer learning<br/>2 stages, AdamW, early stop on val PR-AUC"]
    D --> E["Evaluate + error analysis<br/>val: pick threshold (0.776)<br/>test: exact CIs, OOF, Grad-CAM, robustness"]
    E --> F["Export<br/>model.onnx + model_meta.json<br/>sha256, threshold, norm stats, quality ranges"]
  end

  OFF ==>|"one versioned unit: model.onnx + model_meta.json<br/>(sha256 verified at startup)"| ON

  subgraph ON["ONLINE: FastAPI + ONNX Runtime (CPU, no torch)"]
    direction TB
    LOAD["Startup: verify sha256,<br/>load session, warm-up, /ready"]
    CL(["Client"]) -->|"POST /predict (multipart)"| V["Validation<br/>content type, size, decode,<br/>min side, max pixels<br/>-> 400 / 413 / 415 / 422"]
    V --> P["Preprocess (PIL + numpy)<br/>bilinear resize, grayscale to 3 ch,<br/>normalise: identical to training"]
    P --> I["ONNX Runtime<br/>logits"]
    LOAD -.-> I
    I --> T["Softmax + threshold<br/>defect_probability >= 0.776"]
    P --> Q["Input-quality check<br/>brightness, sharpness vs training range<br/>(informational only)"]
    T --> R["JSON response<br/>class, confidence, defect_probability,<br/>threshold, model_version, input_quality"]
    Q --> R
    T --> LOG["Structured JSON logs<br/>request_id, status, latency,<br/>class, probability (never image bytes)"]
    R --> OUT(["Client receives the JSON"])
    V -. "error" .-> ERR["Uniform error body<br/>code, message, request_id"]
  end

  subgraph FUT["FUTURE WORK (not implemented)"]
    direction TB
    MON["Monitoring<br/>score + image-stat drift,<br/>inspector overrides"]
    RET["Review labels, retrain,<br/>re-validate threshold"]
    MON -.-> RET
  end

  ON -.->|"logs, scores"| FUT
  FUT -.->|"new data, retrained model,<br/>re-validated threshold"| OFF

  classDef future stroke-dasharray: 5 5,fill:#f6f6f6,stroke:#888;
  class MON,RET future;
```

## What each stage is, and where it lives

| stage | what happens | code / output |
|---|---|---|
| Dedup + manifest | pHash proposes near-duplicate pairs, a pixel-RMSE check confirms them, every image gets a `duplicate_group_id` | `scripts/prepare_standin_dataset.py`, `scripts/_common.py`; `reports/dataset_inspection.txt` |
| Group-aware split | `StratifiedGroupKFold` folds are assigned to train/val/test (70/15/15) and to 5 CV folds; no group crosses a split | `src/defect_detection/data.py`, `scripts/make_splits.py`; `reports/dataset_summary.json` |
| Train | ResNet-18 (ImageNet weights), head-only stage then full fine-tune, early stopping on validation PR-AUC | `src/defect_detection/train.py`; `reports/training_runs.md` |
| Evaluate + error analysis | threshold chosen on validation only; test read once; exact CIs, out-of-fold predictions, Grad-CAM, calibration, robustness | `src/defect_detection/evaluate.py`, `error_analysis.py`; `reports/model_comparison.md`, `reports/error_analysis.md` |
| Export | ONNX (opset pinned) + metadata (threshold, normalisation, quality ranges, sha256); parity check against PyTorch | `src/defect_detection/export.py`; `reports/export_parity.json` |
| Validation | content type, size, decode, minimum side, maximum pixel count; one error body for every failure | `api/main.py`, `api/service.py` |
| Preprocess | torch-free; PIL bilinear resize (not cv2), grayscale to 3 channels, normalise; bit-identical to the training transform | `src/defect_detection/preprocess.py`; `reports/preprocess_equivalence.json` |
| Inference + threshold | ONNX Runtime on CPU, softmax, `defect_probability >= threshold` (0.776) | `api/service.py` |
| Input-quality check | brightness and sharpness against the 1st-99th training percentiles; informational, never alters the prediction | `src/defect_detection/preprocess.py` |
| Logs | one JSON object per request: id, status, latency, class, probability, warnings (no image bytes) | `api/logging_config.py` |

## Re-rendering the diagram

Copy the Mermaid block above (without the fences) into a file, for example `diagram.mmd`, and render it:

```bash
npx -y @mermaid-js/mermaid-cli@11.4.2 -i diagram.mmd -o docs/architecture.png -s 2 -b white
```
(On a machine with Chrome but no Puppeteer browser download, add `-p puppeteer.json` with `{"executablePath": "<path to chrome>", "args": ["--no-sandbox"]}`.)
