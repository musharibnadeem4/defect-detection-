"""Task runner (cross-platform; the Makefile just delegates here).  Usage: python tasks.py <task> [<task> ...]

  setup         pip install -r requirements.txt
  data          build the stand-in dataset (data/raw + manifest) and the group-aware splits + normalisation stats
  train         baseline + imbalance sweep + extra architectures (scripts/train.py --phase all)
  evaluate      test-set evaluation, comparison table, 5-fold CV + OOF predictions
  analysis      Step 3: fold models, error analysis, Grad-CAM, robustness
  export        best.pt -> artifacts/model.onnx + model_meta.json, then the PyTorch/ONNX parity check (fails loudly)
  test          pytest (unit, parity and API tests)
  serve         run the API locally (uvicorn); HOST / PORT env vars, default 127.0.0.1:8000
  docker-build  docker build -t defect-detection-api .
  docker-run    docker run -p 8000:8000 defect-detection-api
  examples      call the RUNNING API on curated images -> examples/predictions.{json,md}
  all           data, train, evaluate, export, test
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
IMAGE = os.environ.get("IMAGE", "defect-detection-api")


def run(*cmd: str, env: dict | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(list(cmd), cwd=ROOT, check=True, env={**os.environ, **(env or {})})


def setup():
    run(PY, "-m", "pip", "install", "-r", "requirements.txt")


def data():
    run(PY, "scripts/prepare_standin_dataset.py", "--force")
    run(PY, "scripts/make_splits.py")


def train():
    run(PY, "scripts/train.py", "--phase", "all")


def evaluate():
    run(PY, "scripts/evaluate.py")


def analysis():
    for s in ("cv_save_folds", "error_analysis", "gradcam", "robustness"):
        run(PY, f"scripts/{s}.py")


def export():
    run(PY, "scripts/export_onnx.py")


def test():
    run(PY, "-m", "pytest", "-q")


def serve():
    env = {"PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)])}
    run(PY, "-m", "uvicorn", "api.main:app", "--host", os.environ.get("HOST", "127.0.0.1"),
        "--port", os.environ.get("PORT", "8000"), env=env)


def docker_build():
    run("docker", "build", "-t", IMAGE, ".")


def docker_run():
    run("docker", "run", "--rm", "-p", f"{os.environ.get('PORT', '8000')}:8000", IMAGE)


def examples():
    run(PY, "scripts/predict_examples.py", "--url", f"http://{os.environ.get('HOST', '127.0.0.1')}:{os.environ.get('PORT', '8000')}")


def all_():
    for t in (data, train, evaluate, export, test):
        t()


TASKS = {"setup": setup, "data": data, "train": train, "evaluate": evaluate, "analysis": analysis, "export": export,
         "test": test, "serve": serve, "docker-build": docker_build, "docker-run": docker_run, "examples": examples,
         "all": all_}

if __name__ == "__main__":
    names = sys.argv[1:]
    if not names or any(n not in TASKS for n in names):
        print(__doc__)
        sys.exit(0 if not names else 2)
    try:
        for n in names:
            TASKS[n]()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"task failed: {e}")
        sys.exit(1)
