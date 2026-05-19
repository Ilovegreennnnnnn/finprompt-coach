from __future__ import annotations

import os
from datetime import datetime, timezone
from threading import Event, Thread
from time import sleep
from typing import Any

from app.explorer_service import run_scheduled_watchlists


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


class PhoenixDailyScheduler:
    def __init__(self) -> None:
        self.enabled = _env_flag("ENABLE_PHOENIX_MARKET_LOOP", "false")
        self.hour_utc = int(os.getenv("PHOENIX_MARKET_LOOP_HOUR_UTC", "13"))
        self.minute_utc = int(os.getenv("PHOENIX_MARKET_LOOP_MINUTE_UTC", "0"))
        self.poll_seconds = max(15, int(os.getenv("PHOENIX_MARKET_LOOP_POLL_SECONDS", "60")))
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._last_run_date: str | None = None
        self._last_result: dict[str, Any] = {
            "status": "idle",
            "last_run_date": None,
            "last_error": None,
            "runs_executed": 0,
        }

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return

        self._thread = Thread(target=self._loop, name="phoenix-daily-scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._thread = None
        self._stop_event.clear()

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "hour_utc": self.hour_utc,
            "minute_utc": self.minute_utc,
            "poll_seconds": self.poll_seconds,
            **self._last_result,
        }

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            now = datetime.now(timezone.utc)
            today = now.date().isoformat()
            should_run = (
                self._last_run_date != today
                and (now.hour > self.hour_utc or (now.hour == self.hour_utc and now.minute >= self.minute_utc))
            )

            if should_run:
                if now.weekday() >= 5:
                    self._last_run_date = today
                    self._last_result = {
                        "status": "skipped_weekend",
                        "last_run_date": today,
                        "last_error": None,
                        "runs_executed": 0,
                    }
                else:
                    try:
                        results = run_scheduled_watchlists()
                        self._last_run_date = today
                        self._last_result = {
                            "status": "completed",
                            "last_run_date": today,
                            "last_error": None,
                            "runs_executed": len(results),
                        }
                    except Exception as exc:
                        self._last_run_date = today
                        self._last_result = {
                            "status": "error",
                            "last_run_date": today,
                            "last_error": str(exc),
                            "runs_executed": 0,
                        }

            sleep(self.poll_seconds)


phoenix_daily_scheduler = PhoenixDailyScheduler()
