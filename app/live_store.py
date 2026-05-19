from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class LiveTraceStore:
    ttl_seconds: int = 1800
    _records: dict[str, dict[str, Any]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def _cutoff(self) -> datetime:
        return utcnow() - timedelta(seconds=self.ttl_seconds)

    def _prune_locked(self) -> None:
        cutoff = self._cutoff()
        expired = [
            trace_id
            for trace_id, record in self._records.items()
            if self._parse_created_at(record.get("created_at")) < cutoff
        ]

        for trace_id in expired:
            self._records.pop(trace_id, None)

    def _parse_created_at(self, created_at: Any) -> datetime:
        if isinstance(created_at, datetime):
            return created_at.astimezone(timezone.utc)

        if isinstance(created_at, str):
            try:
                normalized = created_at.replace("Z", "+00:00")
                return datetime.fromisoformat(normalized).astimezone(timezone.utc)
            except ValueError:
                return utcnow()

        return utcnow()

    def save(self, trace_record: dict[str, Any]) -> None:
        trace_id = trace_record.get("trace_id")
        if not trace_id:
            return

        with self._lock:
            self._prune_locked()
            self._records[trace_id] = deepcopy(trace_record)

    def get(self, trace_id: str) -> dict[str, Any] | None:
        with self._lock:
            self._prune_locked()
            record = self._records.get(trace_id)
            return deepcopy(record) if record else None

    def update(self, trace_id: str, patch: dict[str, Any]) -> dict[str, Any] | None:
        with self._lock:
            self._prune_locked()
            record = self._records.get(trace_id)
            if not record:
                return None

            record.update(deepcopy(patch))
            return deepcopy(record)

    def recent(self) -> list[dict[str, Any]]:
        with self._lock:
            self._prune_locked()
            return [
                deepcopy(record)
                for record in sorted(
                    self._records.values(),
                    key=lambda item: item.get("created_at", ""),
                    reverse=True,
                )
            ]


LIVE_TRACE_TTL_SECONDS = int(os.getenv("LIVE_TRACE_TTL_SECONDS", "1800"))
live_trace_store = LiveTraceStore(ttl_seconds=LIVE_TRACE_TTL_SECONDS)


def compact_trace_summary(trace_record: dict[str, Any]) -> dict[str, Any]:
    request = trace_record.get("request", {})
    response = trace_record.get("response", {})
    llm = trace_record.get("llm", {})

    return {
        "trace_id": trace_record.get("trace_id"),
        "created_at": trace_record.get("created_at"),
        "conversation_id": trace_record.get("conversation_id"),
        "request": {
            "message": request.get("message"),
            "tickers": request.get("tickers", []),
            "analysis_mode": request.get("analysis_mode"),
            "primary_tool": request.get("primary_tool"),
            "tool_sequence": request.get("tool_sequence", []),
        },
        "prompt": {
            "template_id": trace_record.get("prompt", {}).get("template_id"),
            "version": trace_record.get("prompt", {}).get("version"),
        },
        "llm": {
            "model": llm.get("model"),
            "usage": llm.get("usage", {}),
        },
        "cost_summary": trace_record.get("cost_summary", {}),
        "tool_trace": trace_record.get("tool_trace", []),
        "response": {
            "outcome": response.get("outcome"),
            "warnings": response.get("warnings", []),
            "answer_preview": (response.get("answer") or "")[:600],
        },
        "promotion": trace_record.get("promotion", {}),
        "has_raw_payload": True,
    }
