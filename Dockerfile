# Inference image (serves the ONNX model via FastAPI). Finalised in a later step.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY api ./api
COPY configs ./configs
ENV PYTHONPATH=/app/src
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
