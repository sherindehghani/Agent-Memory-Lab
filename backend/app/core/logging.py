from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

CONTEXT_FIELDS = ("run_id", "experiment_id", "agent_id", "event_type")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in CONTEXT_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        data = getattr(record, "data", None)
        if data:
            payload["data"] = data
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter() if json_logs else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())


def log_event(
    logger: logging.Logger,
    event_type: str,
    message: str,
    *,
    run_id: str | None = None,
    experiment_id: str | None = None,
    agent_id: str | None = None,
    level: int = logging.INFO,
    **data: Any,
) -> None:
    logger.log(
        level,
        message,
        extra={
            "event_type": event_type,
            "run_id": run_id,
            "experiment_id": experiment_id,
            "agent_id": agent_id,
            "data": data or None,
        },
    )
