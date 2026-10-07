# API reference

The short version is in the [README](../README.md). This page has the full endpoint list, the response fields, the error codes and the settings.

## Endpoints and example commands

```bash
curl -s http://localhost:8000/health        # liveness
curl -s http://localhost:8000/ready         # 200 only when the model unit is verified and a warm-up inference passed
curl -s http://localhost:8000/model-info    # contents of model_meta.json
curl -s -X POST http://localhost:8000/predict -F "file=@examples/images/01_defect_correct_cast_def_0_7896.jpeg;type=image/jpeg"
curl -s -X POST http://localhost:8000/predict -F "file=@examples/images/14_perturbed_blur_sigma2_cast_def_0_7896.png;type=image/png"   # quality warning
curl -s -X POST http://localhost:8000/predict -F "file=@README.md;type=image/jpeg"                                                      # 400
```
More commands: [examples/curl_commands.md](../examples/curl_commands.md). `POST /predict/batch` (field `files`, up to 8 images) returns one result or error per image.

## Response fields

| field | meaning |
|---|---|
| `predicted_class` | `defective` if `defect_probability >= threshold_used`, else `normal` |
| `defect_probability` | softmax probability of the `defective` class. Use this one for ranking and thresholds |
| `confidence` | softmax probability of the **predicted** class (`defect_probability` if defective, otherwise `1 - defect_probability`) |
| `threshold_used` | the deployed threshold, 0.776 (from `model_meta.json`), not 0.5 |
| `model_version` | `<training run id>-<sha256 prefix of model.onnx>` |
| `latency_ms` | server time for decode + preprocess + quality check + inference (not network or upload) |
| `request_id` | also returned as the `X-Request-ID` header; a safe client-supplied one is honoured |
| `input_quality` | `ok`, `warnings`, and the measured brightness/sharpness. Informational only, see [Production considerations](production.md) |

## `confidence` and `defect_probability`

**Because the threshold is 0.776, `confidence` can be below 0.5.** Worked example of the rule (arithmetic, not a measured case): an image with
`defect_probability` 0.60 is below 0.776, so it is `normal`, and its `confidence` is 1 - 0.60 = 0.40. A `normal` answer with a low
`confidence` therefore means "close to the line", not "probably defective".

## Errors

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

## Settings

**Settings** are environment variables (prefix `DEFECT_`, defaults in [api/settings.py](../api/settings.py)):

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

`docker-compose.yml` sets the ONNX Runtime threads to intra-op 2 and inter-op 1, the values the container log showed ([reports/docker_local_check.md](../reports/docker_local_check.md)), matching the 2 physical cores benchmarked in [reports/latency.md](../reports/latency.md). The bare-process default is 0 / 0 (ONNX Runtime chooses).
