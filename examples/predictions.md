# Example predictions

Model `20261007-004422_resnet18_none_224-0f824ffa`, threshold 0.7760. Stand-in dataset images. 'split' is where the image sits in the project split; the deployed model was trained on the **train** split, so predictions on train images are not held-out evidence.

| image | role | true | split | predicted | p(defect) | confidence | quality | warnings |
|---|---|---|---|---|---|---|---|---|
| `01_defect_correct_cast_def_0_7896.jpeg` | correct defect (random) | defective | test | **defective** | 1.0000 | 1.0000 | ok | - |
| `02_defect_correct_cast_def_0_5311.jpeg` | correct defect (random) | defective | test | **defective** | 0.9964 | 0.9964 | ok | - |
| `03_defect_correct_cast_def_0_3650.jpeg` | correct defect (random) | defective | test | **defective** | 1.0000 | 1.0000 | ok | - |
| `04_defect_correct_cast_def_0_9119.jpeg` | correct defect (hardest in test: lowest score) | defective | test | **defective** | 0.9890 | 0.9890 | ok | - |
| `05_normal_correct_cast_ok_0_3479.jpeg` | correct normal (random) | normal | test | **normal** | 0.0001 | 0.9999 | ok | - |
| `06_normal_correct_cast_ok_0_239.jpeg` | correct normal (random) | normal | test | **normal** | 0.0046 | 0.9954 | ok | - |
| `07_normal_correct_cast_ok_0_198.jpeg` | correct normal (random) | normal | test | **normal** | 0.0117 | 0.9883 | ok | - |
| `08_normal_correct_cast_ok_0_7299.jpeg` | correct normal (hardest in test: highest score) | normal | test | **normal** | 0.1705 | 0.8295 | ok | - |
| `09_oof_false_negative_cast_def_0_6988.jpeg` | OOF false negative (Step 3) | defective | train | **normal** | 0.0069 | 0.9931 | ok | - |
| `10_oof_false_negative_cast_def_0_2949.jpeg` | OOF false negative (Step 3) | defective | train | **normal** | 0.0134 | 0.9866 | ok | - |
| `11_oof_false_positive_cast_ok_0_4994.jpeg` | OOF false positive (Step 3) | normal | train | **normal** | 0.0111 | 0.9889 | ok | - |
| `12_oof_false_positive_cast_ok_0_7597.jpeg` | OOF false positive (Step 3) | normal | train | **normal** | 0.0002 | 0.9998 | ok | - |
| `13_oof_false_positive_cast_ok_0_5314.jpeg` | OOF false positive (Step 3) | normal | train | **normal** | 0.0043 | 0.9957 | ok | - |
| `14_perturbed_blur_sigma2_cast_def_0_7896.png` | perturbed (blur_sigma2) copy of cast_def_0_7896.jpeg | defective | test | **defective** | 0.9877 | 0.9877 | WARN | sharpness (Laplacian variance) 7.3 is below the training range [71.0, 354.1] (image may be blurred) |
| `15_perturbed_noise_sigma25_cast_def_0_7896.png` | perturbed (noise_sigma25) copy of cast_def_0_7896.jpeg | defective | test | **defective** | 0.9831 | 0.9831 | WARN | sharpness (Laplacian variance) 12014.3 is above the training range [71.0, 354.1] (image may be noisy or over-sharpened) |
| `16_perturbed_brightness_x1.3_cast_def_0_7896.png` | perturbed (brightness_x1.3) copy of cast_def_0_7896.jpeg | defective | test | **defective** | 0.9560 | 0.9560 | WARN | mean brightness 181.3 is above the training range [129.5, 158.3] |
