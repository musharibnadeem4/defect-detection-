"""CPU latency benchmark -> reports/latency.md (+ reports/latency_results.json).

1. Model only, batch 1: PyTorch vs ONNX Runtime at 1/2/4 threads (real validation images; decode/preprocess excluded).
2. End to end through the real HTTP service (uvicorn subprocess): cold start (process spawn -> /ready), sequential
   request latency (client wall clock vs the server-reported latency_ms), and throughput with 1/2/4/8 concurrent
   clients. Client and server run on the SAME machine and compete for the same cores.
3. INT8 quantisation experiment (read from reports/quantization.json if scripts/quantize_int8.py was run).

Usage: python scripts/benchmark_latency.py [--server-python PATH]   (PATH: interpreter that has the API deps; use the
torch-free 3.11 venv to measure the inference-only environment).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from PIL import Image  # noqa: E402

from defect_detection import preprocess as pp  # noqa: E402
from defect_detection.data import load_splits  # noqa: E402
from defect_detection.utils import ROOT, load_config, load_json, resolve, save_json  # noqa: E402

THREADS = (1, 2, 4)
CONC = (1, 2, 4, 8)


def pct(v, q):
    return float(np.percentile(v, q))


def hardware(server_python: str) -> dict:
    ps = subprocess.run(["powershell", "-NoProfile", "-Command",
                         "$c=Get-CimInstance Win32_Processor; $m=(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB; "
                         "\"$($c.Name)|$($c.NumberOfCores)|$($c.NumberOfLogicalProcessors)|$([math]::Round($m,1))\""],
                        capture_output=True, text=True).stdout.strip().split("|")
    srv = subprocess.run([server_python, "-c", "import sys, onnxruntime as o; print(sys.version.split()[0], o.__version__)"],
                         capture_output=True, text=True).stdout.split()
    try:
        import torch
        tv = torch.__version__
    except Exception:  # noqa: BLE001
        tv = "n/a"
    return {"cpu": ps[0] if len(ps) == 4 else platform.processor(), "physical_cores": ps[1] if len(ps) == 4 else None,
            "logical_cores": ps[2] if len(ps) == 4 else os.cpu_count(), "ram_gb": ps[3] if len(ps) == 4 else None,
            "os": platform.platform(), "client_python": platform.python_version(), "server_python": srv[0] if srv else "?",
            "onnxruntime": ort.__version__, "torch": tv}


def model_only(cfg: dict, meta: dict, x: np.ndarray, n_warm=20, n_run=200) -> dict:
    import torch
    from defect_detection.model import build_model
    art = resolve(cfg, "artifacts_dir")
    ck = torch.load(art / "best.pt", map_location="cpu")
    model = build_model(ck["model"], len(cfg["classes"]), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    name, out = meta["onnx"]["input_name"], {}
    for t in THREADS:
        torch.set_num_threads(t)
        xt = [torch.from_numpy(x[i][None]) for i in range(len(x))]
        times = []
        with torch.inference_mode():
            for i in range(n_warm):
                model(xt[i % len(xt)])
            for i in range(n_run):
                t0 = time.perf_counter()
                model(xt[i % len(xt)])
                times.append((time.perf_counter() - t0) * 1000)
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = t, 1
        s = ort.InferenceSession(str(resolve(cfg, "model_onnx")), so, providers=["CPUExecutionProvider"])
        for i in range(n_warm):
            s.run(None, {name: x[i % len(x)][None]})
        ot = []
        for i in range(n_run):
            t0 = time.perf_counter()
            s.run(None, {name: x[i % len(x)][None]})
            ot.append((time.perf_counter() - t0) * 1000)
        out[t] = {"torch": {"median": pct(times, 50), "p95": pct(times, 95)}, "onnx": {"median": pct(ot, 50), "p95": pct(ot, 95)}}
        print(f"model-only threads={t}: torch {out[t]['torch']['median']:.1f}/{out[t]['torch']['p95']:.1f} ms, "
              f"onnx {out[t]['onnx']['median']:.1f}/{out[t]['onnx']['p95']:.1f} ms", flush=True)
    return out


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_api(server_python: str, intra: int, payloads: list[tuple[str, bytes]], n_seq=200, n_conc=160) -> dict:
    port = free_port()
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)]),
           "DEFECT_ORT_INTRA_OP_THREADS": str(intra), "DEFECT_ORT_INTER_OP_THREADS": "1"}
    log = open(ROOT / f"bench_server_{intra}.log", "w")
    t_spawn = time.perf_counter()
    proc = subprocess.Popen([server_python, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", str(port),
                             "--no-access-log"], cwd=ROOT, env=env, stdout=log, stderr=log)
    base = f"http://127.0.0.1:{port}"
    try:
        cold = None
        while time.perf_counter() - t_spawn < 120:
            try:
                if httpx.get(f"{base}/ready", timeout=1).status_code == 200:
                    cold = time.perf_counter() - t_spawn
                    break
            except httpx.HTTPError:
                time.sleep(0.05)
        if cold is None:
            raise RuntimeError("server did not become ready in 120 s")

        def one(client, i):
            name, data = payloads[i % len(payloads)]
            t0 = time.perf_counter()
            r = client.post("/predict", files={"file": (name, data, "image/jpeg")})
            return (time.perf_counter() - t0) * 1000, r.status_code, r.json().get("latency_ms")

        with httpx.Client(base_url=base, timeout=60) as c:
            for i in range(10):
                one(c, i)
            res = [one(c, i) for i in range(n_seq)]
        assert all(s == 200 for _, s, _ in res), "non-200 response during benchmark"
        wall, srv = [r[0] for r in res], [r[2] for r in res]
        out = {"cold_start_s": cold, "sequential": {"client_median": pct(wall, 50), "client_p95": pct(wall, 95),
                                                    "server_median": pct(srv, 50), "server_p95": pct(srv, 95)}, "concurrent": {}}
        for conc in CONC:
            clients = [httpx.Client(base_url=base, timeout=60) for _ in range(conc)]
            t0 = time.perf_counter()
            with ThreadPoolExecutor(conc) as ex:
                rs = list(ex.map(lambda i: one(clients[i % conc], i), range(n_conc)))
            el = time.perf_counter() - t0
            for c in clients:
                c.close()
            assert all(s == 200 for _, s, _ in rs)
            lat = [r[0] for r in rs]
            out["concurrent"][conc] = {"throughput_rps": n_conc / el, "client_median": pct(lat, 50), "client_p95": pct(lat, 95)}
        print(f"API intra={intra}: cold {cold:.1f}s, seq client {out['sequential']['client_median']:.1f}/"
              f"{out['sequential']['client_p95']:.1f} ms, server {out['sequential']['server_median']:.1f} ms; "
              f"rps " + ", ".join(f"{k}:{v['throughput_rps']:.1f}" for k, v in out["concurrent"].items()), flush=True)
        return out
    finally:
        # the venv's python.exe is a launcher that spawns the real interpreter: kill the whole process tree
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        else:
            proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        try:
            (ROOT / f"bench_server_{intra}.log").unlink(missing_ok=True)
        except PermissionError:
            pass


def write_report(cfg, hw, mo, api, quant) -> None:
    L = ["# Latency and serving trade-offs", "",
         "> Stand-in model (ResNet-18 @224). Numbers are specific to this machine; re-measure on the target hardware.", "",
         "## Hardware / software", "",
         f"- CPU: {hw['cpu']} ({hw['physical_cores']} physical / {hw['logical_cores']} logical cores), {hw['ram_gb']} GB RAM, no GPU",
         f"- OS: {hw['os']}",
         f"- Client/benchmark Python {hw['client_python']} (torch {hw['torch']}); API server Python {hw['server_python']}, "
         f"onnxruntime {hw['onnxruntime']}",
         "- Client and server run on the same machine and share the same cores, so API figures include that contention.", "",
         "## 1. Model only, batch size 1 (median / p95 ms)", "",
         "Forward pass of the network on a preprocessed 1x3x224x224 tensor (real images; decode and resize excluded); "
         "200 timed runs after 20 warm-up runs.", "",
         "| threads | PyTorch (eager) | ONNX Runtime | ORT speed-up (median) |", "|---|---|---|---|"]
    for t, r in mo.items():
        L.append(f"| {t} | {r['torch']['median']:.1f} / {r['torch']['p95']:.1f} | {r['onnx']['median']:.1f} / {r['onnx']['p95']:.1f} "
                 f"| {r['torch']['median'] / r['onnx']['median']:.2f}x |")
    L += ["", "## 2. End to end through the HTTP service", "",
          "Single uvicorn worker, `POST /predict` with a ~10 KB JPEG, keep-alive connection. *client* = wall clock around the "
          "HTTP call (connection, multipart upload, decode, preprocess, quality check, inference, JSON); *server* = the "
          "`latency_ms` field (decode + preprocess + quality check + inference only). 200 sequential requests.", "",
          "| ORT intra-op threads | cold start (spawn -> /ready) | client median / p95 (ms) | server median / p95 (ms) | HTTP+upload overhead (median) |",
          "|---|---|---|---|---|"]
    for t, r in api.items():
        s = r["sequential"]
        L.append(f"| {t} | {r['cold_start_s']:.1f} s | {s['client_median']:.1f} / {s['client_p95']:.1f} | "
                 f"{s['server_median']:.1f} / {s['server_p95']:.1f} | {s['client_median'] - s['server_median']:.1f} ms |")
    L += ["", "Throughput with concurrent clients (160 requests per level; requests/s, and client p95 latency in ms):", "",
          "| ORT intra-op threads | " + " | ".join(f"{c} client(s)" for c in CONC) + " |", "|---|" + "---|" * len(CONC)]
    for t, r in api.items():
        L.append(f"| {t} | " + " | ".join(f"{r['concurrent'][c]['throughput_rps']:.1f} rps (p95 {r['concurrent'][c]['client_p95']:.0f})" for c in CONC) + " |")
    L += ["", "Cold start is the process start of the server in this environment (Python import, session creation, 2 warm-up "
          "inferences); a container adds image pull and start-up time that I could **not** measure (Docker was not available).", ""]
    if quant:
        fp = quant["fp32"]
        L += ["## 3. INT8 quantisation experiment (optional; not deployed)", "",
              f"Evaluated on the validation split ({quant['n']} images) with the deployed threshold fixed at "
              f"{quant['threshold_fixed']:.4f} (not re-tuned). FP32 reference: {fp['size_mb']:.1f} MB, PR-AUC {fp['pr_auc']:.3f}, "
              f"recall {fp['recall@deployed']:.3f} / precision {fp['precision@deployed']:.3f} at the deployed threshold.", "",
              "| method | result |", "|---|---|"]
        for name, m in quant["methods"].items():
            if m["status"] != "ok":
                L.append(f"| {name} | {m['status']} |")
            else:
                s = m["speedup_median"]
                L.append(f"| {name} | {m['size_mb']:.1f} MB; batch-1 speed-up {s['1']:.2f}x / {s['2']:.2f}x / {s['4']:.2f}x (1/2/4 threads); "
                         f"max logit diff {m['max_abs_logit_diff']:.2f}, max probability diff {m['max_abs_prob_diff']:.2f}; "
                         f"**{m['decision_flips_at_deployed_threshold']} decision flips** at the deployed threshold "
                         f"({m['decision_flips_at_0_5']} at 0.5); PR-AUC {m['pr_auc']:.3f}; at the deployed threshold recall "
                         f"{m['recall_at_deployed']:.3f}, precision {m['precision_at_deployed']:.3f}. **Verdict: {m['verdict']}** |")
        L += ["", "Given that Step 3 showed the decision threshold is fragile to score shifts (a ~2.6-logit offset was enough to "
              "produce 25 false positives in one fold), a quantised model whose logits move by this much would need its "
              "threshold re-tuned and re-validated; the 1.5x speed-up does not justify that here, and FP32 latency is already "
              "well under typical inspection-line budgets of hundreds of milliseconds (an assumption, not a requirement given by the brief).", ""]
    (resolve(cfg, "reports_dir") / "latency.md").write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-python", default=sys.executable)
    args = ap.parse_args()
    cfg = load_config()
    meta = load_json(resolve(cfg, "model_meta"))
    sp = load_splits(cfg)
    val = sp[sp.split == "val"].head(20)              # timing inputs only; validation images, nothing is scored
    raw = resolve(cfg, "raw_dir")
    imgs = [Image.open(raw / r.label / r.filename) for r in val.itertuples()]
    x = pp.preprocess_batch(imgs, meta["input"]["size"], meta["normalization"]["mean"], meta["normalization"]["std"])
    payloads = [(r.filename, (raw / r.label / r.filename).read_bytes()) for r in val.itertuples()]
    hw = hardware(args.server_python)
    print(json.dumps(hw), flush=True)
    mo = model_only(cfg, meta, x)
    api = {t: run_api(args.server_python, t, payloads) for t in THREADS}
    qp = resolve(cfg, "reports_dir") / "quantization.json"
    quant = load_json(qp) if qp.exists() else None
    save_json({"hardware": hw, "model_only": mo, "api": api}, resolve(cfg, "reports_dir") / "latency_results.json")
    write_report(cfg, hw, mo, api, quant)
    print((resolve(cfg, "reports_dir") / "latency.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
