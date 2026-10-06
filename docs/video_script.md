# Video script (target 2:55)

> The segment lengths in the brief add up to 3:20, over the 2:30-3:00 target, so five segments are trimmed (problem 20 to 15 s, data 25 to 20 s, model 30 to 25 s,
> results 25 to 20 s, Docker/latency 20 to 15 s). Error analysis (35 s), the live demo (35 s) and limitations (10 s) keep their length. Total: 175 s = 2:55.
> Base narration is 2.0-2.8 words per second. Sentences marked *(cut if short on time)* are optional extras: with them, segments 2 and 7 run at about 3 words per second, so include them only if you speak fast (the live-demo segment is mostly on-screen output).

## Before recording (pre-flight)

- Server running and warm: `python tasks.py serve` (or `docker compose up`), then `curl -s localhost:8000/ready` once.
- Terminal with a large font, working directory at the repo root, and these commands ready in the history:
  ```bash
  curl -s localhost:8000/ready
  curl -s -X POST localhost:8000/predict -F "file=@examples/images/01_defect_correct_cast_def_0_7896.jpeg;type=image/jpeg"
  curl -s -X POST localhost:8000/predict -F "file=@examples/images/14_perturbed_blur_sigma2_cast_def_0_7896.png;type=image/png"
  curl -s -X POST localhost:8000/predict -F "file=@README.md;type=image/jpeg"
  ```
  Pipe through `python -m json.tool` if you want pretty output, but test it first so it does not eat time.
- Browser tabs open: README (rendered), `reports/figures/samples_grid.png`, `reports/dataset_inspection.txt`, `reports/training_runs.md`,
  `reports/figures/confusion_matrix.png`, `reports/figures/per_fold_score_distributions.png`, `reports/figures/gradcam_montage_FN.png`,
  `reports/figures/robustness.png`, `reports/latency.md`, and the GitHub Actions page.
- **Only show a green Actions run if CI has actually passed by the time you record.** If it has not run yet, say "the workflow is written and has not run yet".

## Timeline

| time | segment | on screen | narration (what to say) |
|---|---|---|---|
| **0:00-0:15** | Problem and stand-in dataset | README TL;DR, then `samples_grid.png` (8 normal, 8 defective) | "I built a normal-versus-defective detector for cast metal parts, served as an API. The client's data wasn't provided, so a public casting dataset stands in: 700 images, 18% defective. This shows the engineering, not client performance." |
| **0:15-0:35** | Key data decisions | the duplicate table in `dataset_inspection.txt` (pHash-only rows vs the used row), then the split table in the README | "Two data decisions. Perceptual hashing alone failed: it flagged 69% of images as duplicates and chained them into one 2,280-image group. So I confirm hash candidates with a pixel check, which leaves 51 real groups. And the split is group-aware, so near-duplicates can't leak between train, validation and test. *(cut if short on time: The subsample keeps 18% defective, seed 42.)*" |
| **0:35-1:00** | Model and imbalance | `training_runs.md` table | "ResNet-18 transfer learning in two stages: head first, then fine-tune everything. I compared 224 and 300 pixels, EfficientNet-B0 and a pixel baseline, plus three ways of handling imbalance. All of them tie on validation, so the imbalance comparison is inconclusive. I chose plain ResNet-18 at 224 on a log-loss tie-break and cost. *(cut if short on time: The threshold, 0.776, is tuned on validation for 95% recall.)*" |
| **1:00-1:20** | Results and honest intervals | `confusion_matrix.png`, then the README test and OOF tables | "On the test set it finds all 19 defects with no false alarms. But with 19 defects, the exact interval for recall is 0.82 to 1.0, so perfect is not proven. Cross-validation on 595 images gives recall 0.98 and precision 0.94 at the deployed threshold, and I trust that number more." |
| **1:20-1:55** | Error analysis | `per_fold_score_distributions.png` (point at fold 2), `gradcam_montage_FN.png`, `robustness.png` | "Three findings. One: fold two has 25 false positives at 0.5 even though its PR-AUC is 1.0. Its normal images score 2.65 logits higher: a calibration offset in that model, not a ranking problem, and temperature scaling can't fix it. Two: two defects are missed with high confidence. I can't see a defect in either image, and even the model trained on them still calls them normal. The cause is unknown; it could be a label problem, that's a hypothesis. Three: noise, blur and brighter images degrade it." |
| **1:55-2:30** | Live API demo | terminal, four commands | "Live. Ready. A defective image: class, probability, and the deployed threshold, 0.776. Now the same image, blurred: the prediction is unchanged, but the input-quality guard warns the image is outside the training range. And a corrupt file: a clean 400 with a request ID, no stack trace." |
| **2:30-2:45** | Docker, CI and latency | `ci.yml` / Actions page (only if green), `latency.md` table, `quantization.json` | "The Docker image is built and smoke-tested in CI; I have no Docker locally. ONNX Runtime is about twice as fast as PyTorch: 27 versus 56 milliseconds at batch one. *(cut if short on time: INT8 was rejected: it flipped validation decisions at the fixed threshold.)*" |
| **2:45-2:55** | Limitations and next steps | README "Known limitations" | "Limits: stand-in data, only 19 test defects, calibration that moves between training runs. Next: real data, hard-negative mining, an anomaly-detection comparison, drift monitoring and a threshold re-validation procedure." |

## Demo notes (segment 1:55-2:30)

1. `ready` should print `{"status":"ready", ...}`. Say nothing while it prints.
2. In the defective response, point at `predicted_class`, `defect_probability`, `threshold_used` (0.776) and `input_quality.ok: true`.
3. In the blurred response, point at `input_quality.ok: false` and the warning ending "input outside the training range; prediction may be less reliable"; note the class is still `defective`
   (the guard never changes the prediction).
4. The corrupt-file call sends `README.md` declared as `image/jpeg`: expected `400` with code `invalid_image`.

## Facts to have at hand (all from files in the repo)

| fact | source |
|---|---|
| 51 duplicate groups, 103 images (1.7%); pHash at Hamming <= 4 flags 4,166 of 6,000 | `reports/dataset_inspection.txt` |
| split 490 / 105 / 105; 18.0 / 18.1 / 18.1% defective | `reports/dataset_summary.json` |
| test recall 19/19, exact CI [0.82, 1.00]; OOF recall 0.981, precision 0.938 at 0.776 | `reports/model_comparison.md`, `reports/error_analysis.md` |
| fold 2: 25 false positives at 0.5, normals' median logit -1.60 vs -4.25 | `reports/error_analysis.md` |
| ONNX Runtime 27.3 ms vs PyTorch 55.9 ms (2 threads, batch 1) | `reports/latency.md` |
| INT8: 2 validation decision flips at the fixed threshold | `reports/quantization.json` |
