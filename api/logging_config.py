"""Structured JSON logging. Image bytes are never logged: log calls pass only the small `fields` dicts built in
main.py (request id, path, status, latency, predicted class, probability, quality warnings)."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

LOGGER_NAME = "defect.api"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {"ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
               "level": record.levelname, "logger": record.name, "event": record.getMessage()}
        out.update(getattr(record, "fields", {}))
        if record.exc_info:  # server-side only; never returned to clients
            out["exception"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def setup_logging(level: str = "INFO") -> logging.Logger:
    log = logging.getLogger(LOGGER_NAME)
    log.setLevel(level.upper())
    if not any(getattr(h, "_defect_json", False) for h in log.handlers):
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(JsonFormatter())
        h._defect_json = True  # type: ignore[attr-defined]
        log.addHandler(h)
        log.propagate = False
    return log
