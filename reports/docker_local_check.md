# Local Docker verification (2026-10-07)

> **Provenance.** These are the author's own terminal outputs from running the image locally with Docker 29.8.2 on a Windows host
> (Microsoft Windows 11 Pro, version 10.0.22621, as reported by `Get-ComputerInfo` on the same machine). They were transcribed into this file from what the author
> reported; the raw terminal logs are not stored in the repository, and the file is not machine-generated. Nothing here was re-measured by the assistant that wrote it.

The image is built from `Dockerfile` (base `python:3.11-slim`, inference dependencies only, non-root user, `HEALTHCHECK` on `/ready`) and started with
`docker-compose.yml`. The model unit inside the image is the committed `artifacts/model.onnx` + `artifacts/model_meta.json`.

## Build

| item | value |
|---|---|
| command | `docker compose up --build` (cold build, no layer cache) |
| total time | **284.9 s** |
| build context | 44.81 MB (the committed `model.onnx` is 44.7 MB of that) |
| `pip install` step | 110.0 s |

## Image size (`docker images`)

| image | DISK USAGE | CONTENT SIZE |
|---|---|---|
| `defect-detection-api:latest` | **837 MB** | **219 MB** |

Two numbers because `docker images` reports two sizes: *disk usage* is the unpacked image on disk (all layers extracted), *content size* is the compressed image content (roughly what a
registry pull transfers). The two figures are not interchangeable; the README quotes both and labels them.

## Running container

| check | result |
|---|---|
| `docker compose ps` | `Up 4 minutes (healthy)`, port 8000 |
| `docker compose exec defect-api whoami` | `appuser` (non-root) |

## Startup log

| event | time (UTC, as logged) |
|---|---|
| `startup` | 00:09:13.969 |
| `ready` | 00:09:14.898 |

The two events are about **0.93 s** apart (14.898 - 13.969 = 0.929 s). That is the application's model load, session creation and warm-up *inside the already-running container*.
It is **not** the total container cold start (container creation, process start and Python imports are not in it), and total container cold start was not measured.

The `model_loaded` event: `model_version` `20261007-004422_resnet18_none_224-0f824ffa`, `threshold` 0.7760080151863646, ONNX Runtime intra-op threads 2 and inter-op threads 1,
`onnxruntime` 1.20.1. These match `artifacts/model_meta.json` (model version, threshold) and `docker-compose.yml` (`DEFECT_ORT_INTRA_OP_THREADS: "2"`, `DEFECT_ORT_INTER_OP_THREADS: "1"`).

## curl smoke tests against the running container

| request | result |
|---|---|
| `GET /health` | `{"status":"ok"}` |
| `POST /predict` with `examples/images/01_defect_correct_cast_def_0_7896.jpeg` | `defective`, `defect_probability` 0.99997, `input_quality.ok` true, `latency_ms` 60.66 |
| `POST /predict` with `README.md` (curl sent `application/octet-stream`) | **415** `unsupported_media_type` |
| `POST /predict` with `examples/images/14_perturbed_blur_sigma2_cast_def_0_7896.png` | `defective`, `defect_probability` 0.9877, `input_quality.ok` false, sharpness 7.3 against the training range [71.0, 354.1] |
| `POST /predict` with a text file labelled `image/jpeg` | **400** `invalid_image`; the `request_id` in the body matched the `x-request-id` response header and the structured log line (status 400, latency 39.88 ms, `error_code` `invalid_image`); the file content did not appear in the log |

The predictions agree with the examples recorded earlier from a non-Docker server (`examples/predictions.md`: example 01 `defect_probability` 0.9999652, blurred copy 0.9877 with the same sharpness warning).

## What this does and does not show

- `latency_ms` values (60.66 ms and the 39.88 ms in the log line) are **single requests, not a benchmark**. The benchmark is `reports/latency.md`, measured on a bare process, not in a container.
- The Python 3.11 Linux image was exercised **only by these manual calls**. No pytest run happened inside the container. (The torch-free test subset was run separately in a Windows Python 3.11
  virtualenv: 50 passed, 2 skipped; that is not the Linux image.)
- Not measured: total container cold start, memory use, behaviour under load in the container.
