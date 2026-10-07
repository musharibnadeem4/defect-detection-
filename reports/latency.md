# Latency and serving trade-offs

> Stand-in model (ResNet-18 @224). Numbers are specific to this machine; re-measure on the target hardware.

## Hardware / software

- CPU: Intel(R) Core(TM) i5-7200U CPU @ 2.50GHz (2 physical / 4 logical cores), 15.9 GB RAM, no GPU
- OS: Microsoft Windows 11 Pro (version 10.0.22621)
- Client/benchmark Python 3.12.0 (torch 2.6.0+cpu); API server Python 3.11.2, onnxruntime 1.20.1
- Client and server run on the same machine and share the same cores, so API figures include that contention.

## 1. Model only, batch size 1 (median / p95 ms)

Forward pass of the network on a preprocessed 1x3x224x224 tensor (real images; decode and resize excluded); 200 timed runs after 20 warm-up runs.

| threads | PyTorch (eager) | ONNX Runtime | ORT speed-up (median) |
|---|---|---|---|
| 1 | 79.9 / 87.7 | 43.7 / 59.1 | 1.83x |
| 2 | 55.9 / 63.9 | 27.3 / 34.5 | 2.05x |
| 4 | 55.3 / 64.8 | 26.2 / 36.5 | 2.11x |

## 2. End to end through the HTTP service

Single uvicorn worker, `POST /predict` with a ~10 KB JPEG, keep-alive connection. *client* = wall clock around the HTTP call (connection, multipart upload, decode, preprocess, quality check, inference, JSON); *server* = the `latency_ms` field (decode + preprocess + quality check + inference only). 200 sequential requests.

| ORT intra-op threads | cold start (spawn -> /ready) | client median / p95 (ms) | server median / p95 (ms) | HTTP+upload overhead (median) |
|---|---|---|---|---|
| 1 | 2.3 s | 51.6 / 54.7 | 48.3 / 51.4 | 3.3 ms |
| 2 | 2.5 s | 37.7 / 44.9 | 33.9 / 40.4 | 3.8 ms |
| 4 | 2.5 s | 36.3 / 87.7 | 31.9 / 83.1 | 4.4 ms |

Throughput with concurrent clients (160 requests per level; requests/s, and client p95 latency in ms):

| ORT intra-op threads | 1 client(s) | 2 client(s) | 4 client(s) | 8 client(s) |
|---|---|---|---|---|
| 1 | 18.2 rps (p95 69) | 31.9 rps (p95 71) | 37.4 rps (p95 116) | 37.7 rps (p95 256) |
| 2 | 25.7 rps (p95 43) | 30.9 rps (p95 70) | 35.3 rps (p95 142) | 35.8 rps (p95 291) |
| 4 | 26.1 rps (p95 46) | 27.0 rps (p95 109) | 24.5 rps (p95 255) | 30.0 rps (p95 403) |

Cold start here is the process start of the server as a bare process in a Python 3.11 environment (Python import, session creation, 2 warm-up inferences); every number in this file comes from bare processes, none from inside a container.

**Container (separate local check, see [docker_local_check.md](docker_local_check.md)).** Inside the running container the application logged its `startup` and `ready` events about 0.93 s apart (model load and warm-up only). That is **not** the total container cold start, which was not measured. Image size (`docker images`, Docker 29.8.2): 837 MB disk usage (unpacked), 219 MB content size (compressed); a cold uncached build took 284.9 s. The `latency_ms` values from the container smoke tests are single requests, not a benchmark.

## 3. INT8 quantisation experiment (optional; not deployed)

Evaluated on the validation split (105 images) with the deployed threshold fixed at 0.7760 (not re-tuned). FP32 reference: 44.7 MB, PR-AUC 1.000, recall 1.000 / precision 1.000 at the deployed threshold.

| method | result |
|---|---|
| dynamic | not usable: NotImplemented: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for ConvInteger(10) node with name '/conv1/Conv_quant' |
| static_qdq | 11.2 MB; batch-1 speed-up 1.53x / 1.56x / 1.50x (1/2/4 threads); max logit diff 4.08, max probability diff 0.85; **2 decision flips** at the deployed threshold (8 at 0.5); PR-AUC 0.997; at the deployed threshold recall 1.000, precision 0.905. **Verdict: do NOT adopt: 2 decision flip(s) at the deployed threshold; max probability drift 0.853 (limit 0.01)** |

Given that Step 3 showed the decision threshold is fragile to score shifts (a ~2.6-logit offset was enough to produce 25 false positives in one fold), a quantised model whose logits move by this much would need its threshold re-tuned and re-validated; the 1.5x speed-up does not justify that here, and FP32 latency is already well under typical inspection-line budgets of hundreds of milliseconds (an assumption, not a requirement given by the brief).
