# Interview prep: 15 likely questions

Answers are drawn from this project, with the file that backs each number. **[Hypothesis]** marks an explanation I did not test; **[Not tested]** marks something I would do
but did not do here. Remember the framing: every result comes from a **stand-in dataset** (Kaggle casting, 700 images, 18% defective).

---

### 1. Why a group-aware split?
Near-duplicate images (the same part photographed twice, re-saved or augmented copies) can land on both sides of a random split, so the test score partly measures
memory. I give every image a `duplicate_group_id` and split with `StratifiedGroupKFold` so a group is never shared between train, validation, test or CV folds. On this
subsample it is effectively a no-op: only 11 of the 700 images belong to a source duplicate group and none has its partner inside the subsample
(`reports/dataset_summary.json`). I built it because real data will not be that clean, and I unit-tested it with synthetic groups (`tests/test_splits.py`). I will not claim it
changed a number here.

### 2. How did you find duplicates, and why not just use perceptual hashing?
I tried pHash first and it failed on this data: every image shows the same part in the same pose, so hashes of *different* parts are close. At Hamming distance <= 4 it flagged 4,166 of
6,000 images (69%) and chained them into a 2,280-image group; 30 groups mixed both classes (`reports/dataset_inspection.txt`). The fix is two-stage: pHash proposes candidate pairs
(distance <= 8) and a pixel RMSE on 64x64 thumbnails (<= 3.0) confirms them, which gives 51 groups covering 103 images (1.7%), none mixing classes. Be honest that 3.0 is a
judgement call: pixel distances form a continuum, and RMSE <= 5 would give 183 groups and 2 mixed ones. I also did not check rotated or flipped copies.

### 3. Why not report accuracy?
With 18% defective, a model that always says "normal" is about 82% accurate (86 of 105 on the test set) and finds nothing. The cost of errors is asymmetric (a missed
defect ships a bad part; a false alarm costs inspector time), so I report recall and precision on the defective class with exact confidence intervals, plus PR-AUC and ROC-AUC. Precision also depends on
prevalence: at the deployed operating point (recall 0.981, false-positive rate 7/488) the implied precision is 0.94 at 18% defective but only 0.78 at 5%, 0.58 at 2% and 0.41 at 1%
(an arithmetic projection in `reports/error_analysis.md`, not a measurement).

### 4. How would you handle a 50:1 imbalance?
**[Not tested]** Here the imbalance is only 4.6:1 and the comparison of no handling, weighted loss and a weighted sampler was **inconclusive**: all three reach validation PR-AUC 1.000 and
recall 1.000 on 19 validation defects (`reports/training_runs.md`). What I would do at 50:1 is a plan, not a result: first note that the number of positives matters more than the ratio (700 images at 50:1
is about 14 defects, too few to evaluate), so I would collect more positives; keep PR-AUC and recall at a fixed precision as the metrics; compare weighting, sampling and plain training across several seeds
with confidence intervals rather than one run; calibrate and pick the threshold on a validation set with enough positives; and compare against an anomaly-detection model trained on normals only. One thing I did
observe: the sampler gave a badly calibrated model (test precision 0.655 at threshold 0.5 vs 1.000 for no handling).

### 5. Why ResNet-18 transfer learning?
Only 490 training images, so pretrained ImageNet features are the sensible start; ResNet-18 (11.2M parameters) fine-tunes on a CPU and exports cleanly to ONNX. I fine-tune in two stages (head
first, then all layers) with early stopping on validation PR-AUC. I did **not** show it is better than the alternatives: ResNet-18 at 300 px and EfficientNet-B0 tie with it on every metric on validation and test
(`reports/model_comparison.md`), and a logistic regression on 64x64 pixels already reaches 0.972 test PR-AUC, so this stand-in is easy. I picked ResNet-18 @224 on a log-loss tie-break and because it is cheaper than the 300 px model
(54 ms vs 91 ms in the evaluation's PyTorch timing).

### 6. The test score is perfect. Do you believe it?
Not as a precise estimate. 19/19 defects found with 0 false alarms, but with 19 defects the exact Clopper-Pearson 95% interval for recall is [0.82, 1.00], and bootstrap intervals for F1 and the AUCs
collapse to zero width, so I mark them degenerate and do not quote them. Cross-validation on 595 images is more informative: recall 0.981 [0.934, 0.998], precision 0.938 [0.875, 0.975] at the deployed threshold (the deployed threshold applied to predictions from five differently calibrated fold models, not the deployed model's own precision), and it disagrees with the test
set on precision at 0.5 (0.761 vs 1.000). On leakage: the test set was read only after the model and threshold were chosen on validation, I ran a nearest-neighbour probe (closest test-to-train/val pixel RMSE is 4.51; none within the duplicate
distance of 3.0), and the robustness check reads test read-only with nothing re-tuned.

### 7. Why threshold 0.776, and when must it be re-validated?
It is the validation-set threshold that reaches recall >= 0.95 with the highest precision. With 19 validation defects, 18/19 = 0.947 is below 0.95, so it effectively means catching all 19. Many thresholds tie at precision 1.0, and the
largest one sits exactly on the hardest validation defect (0.993), so I take the logit-space midpoint of the tied range instead (0.776). Re-validate whenever the model is retrained, the camera, lighting or process changes, a drift alert fires, or a new defect type appears:
the fold analysis showed the normal-class score offset moving by about 2.6 logits between training runs, which alone produced 25 false positives in one fold. The threshold is stored in `model_meta.json`, so changing it means a new model version.

### 8. What did fold 2 teach you?
That PR-AUC does not see calibration. Fold 2 has PR-AUC 1.000 (every defect scores above every normal, a gap of 2.44 logits) yet 25 false positives at 0.5, because its normals score a median -1.60 logits against -4.25 for the other folds. The offset is a
property of that fold's *model*: it also scores 95 of its own 390 training normals above 0.5, and the fold-2 images are not unusual on any image statistic I measured. Lessons: report calibration per training run, never pool a threshold across different models, and
re-check the threshold after every retrain. **[Hypothesis]** The cause is that each CV fold stops at epoch 5 of a truncated cosine schedule with no checkpoint selection, so the bias varies by seed; I did not test that (it needs repeated seeds and a schedule sweep).

### 9. Two defects are missed with high confidence. What do you do about them?
They are `cast_def_0_6988` (p = 0.024) and `cast_def_0_2949` (p = 0.118). I could not see a defect in either by eye and Grad-CAM does not land on anything identifiable; catching both would need a threshold of about 0.024, which flags 245 of 488 normals. The deployed model, which trained on them, still calls
both normal (p about 0.01, `examples/predictions.md`). **[Hypothesis]** Either the defect is too subtle at 224 px, or the labels are wrong, or the images share an acquisition condition (both have low sharpness and a bright background, n = 2). The action is to have an inspector re-label them, view them at full resolution, and
if they are real defects, collect more examples of that type; I would not tune the model to fit two images.

### 10. Why PIL and not cv2 for resizing?
Training uses torchvision `Resize` on PIL images, which calls `PIL.Image.resize(BILINEAR)`, an antialiased filter. `cv2.resize(INTER_LINEAR)` does not antialias when shrinking, so it produces different pixels. I measured the effect on the real model
(`reports/preprocess_equivalence.json`): cv2 INTER_LINEAR changes probabilities by up to 0.19, INTER_CUBIC by up to 0.43, and PIL NEAREST flips 40 of 105 decisions. With the PIL path the inference tensors are bit-identical to the training transform (max difference 0.0), enforced by a test.
cv2 changed no decision at 0.776 on those 105 images only because their scores are far from the threshold; with a fixed threshold that is luck, not safety.

### 11. How do you know the ONNX model behaves like the PyTorch one?
A parity script and a pytest compare them on all 105 test images and fail loudly on a mismatch: max logit difference 3.2e-05 (tolerance 1e-3), max probability difference 1.7e-06, and identical decisions at 0.776 for 105/105, with the nearest score 0.213 from the threshold
(`reports/export_parity.json`). It also checks the dynamic batch axis (batch vs single inference: 0.0 difference) and that the model file matches the sha256 recorded in `model_meta.json`. The API test additionally checks the HTTP result equals the offline ONNX result.

### 12. Why did you reject INT8 quantisation?
Dynamic quantization could not run: ONNX Runtime's CPU provider has no `ConvInteger` kernel, and ResNet-18 is mostly convolutions. Static QDQ quantization (calibrated on 100 training images) is 4x smaller (11.2 vs 44.7 MB) and about 1.5x faster, but logits move by up to 4.08, probabilities by up to 0.85, and 2 of 105
validation decisions flip at the fixed threshold (precision 1.000 to 0.905) (`reports/quantization.json`). Given that the threshold is already fragile to score shifts, I would need to re-tune and re-validate it for a 1.5x gain on a service that already answers in about 38 ms, so I did not adopt it.

### 13. What does the input-quality guard do, and what does it not do?
It compares mean brightness and Laplacian-variance sharpness with the 1st-99th percentile range of the 490 training images and returns `ok: false` with a cautious warning ("input outside the training range; prediction may be less reliable"). It never changes the prediction. It is a coarse flag, not a detector of wrong answers:
it also warns on 3.9% of training, 4.8% of validation and 7.6% of test images that are normal and correctly classified, and it does not flag the two known missed defects (`reports/quality_check_false_alarm_rate.json`). It was motivated by the robustness results (blur sigma 2, noise sigma 25 and brightness x1.3 hurt), but I did not
measure how well the warning predicts errors.

### 14. How would you detect drift in production?
**[Not implemented; plan]** The failure mode I actually observed was a shift of the *normal-class score distribution* (fold 2, and the noise/blur perturbations raised the mean normal score from 0.009 to 0.31-0.66) while ranking mostly survived. So I would monitor the distribution of `defect_probability` (mean, quantiles, fraction above the
threshold) and the image statistics the guard already computes (brightness, sharpness, fraction with warnings, input resolution), against a baseline taken from validation, alongside inspector override rates and, when labels arrive, rolling recall/precision with exact intervals. A two-sample test or population-stability index on the
scores is a reasonable alert rule, but the alert thresholds would need tuning on real traffic; I have none.

### 15. What would you do with real data?
First re-run the whole protocol on it, because the stand-in is much easier than most real data (a pixel baseline scores 0.972). Then: get defect-type labels and masks (to score Grad-CAM and report per-type recall), more defects overall, and group by part, batch or capture session when splitting;
re-derive the duplicate threshold on that data; add hard-negative mining from the false positives and re-label review of confident misses; try higher-resolution or rim crops, since the defects are small and rim-local; compare against an anomaly-detection model (for unseen defect types and very low prevalence); run several seeds
and report calibration spread; set the threshold with an exact lower bound on recall; use a time-separated test set; and build the drift-monitoring and retraining loop above. **[Not tested]** None of these were run here.
