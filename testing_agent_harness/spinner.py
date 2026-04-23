"""Threaded, TTY-only wave spinner.

Writes a single line of animated block characters plus a status string to
``stderr`` via ``\\r`` + ``\\x1b[2K``. Falls back to a no-op when output is not
a TTY (e.g. piped to a file) so it never pollutes captured logs.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import TextIO


WAVE = "▁▂▃▄▅▆▇█▇▆▅▄▃▂"
_HIDE = "\x1b[?25l"
_SHOW = "\x1b[?25h"
_CLEAR_LINE = "\r\x1b[2K"


class WaveSpinner:
    """Animated status line that can be started/stopped from the main thread."""

    def __init__(
        self,
        stream: TextIO | None = None,
        status: str = "thinking",
        width: int = 14,
        interval: float = 0.08,
    ) -> None:
        self._stream = stream or sys.stderr
        self._status = status
        self._width = width
        self._interval = interval
        self._tick = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._enabled = self._stream.isatty() and os.environ.get("TERM") != "dumb"

    def start(self, status: str | None = None) -> None:
        if not self._enabled or self._running:
            if status:
                self._status = status
            return
        if status:
            self._status = status
        self._running = True
        try:
            self._stream.write(_HIDE)
            self._stream.flush()
        except Exception:  # noqa: BLE001
            pass
        self._thread = threading.Thread(target=self._loop, name="wave-spinner", daemon=True)
        self._thread.start()

    def set_status(self, status: str) -> None:
        with self._lock:
            self._status = status

    def stop(self) -> None:
        if not self._enabled:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None
        try:
            self._stream.write(_CLEAR_LINE + _SHOW)
            self._stream.flush()
        except Exception:  # noqa: BLE001
            pass

    def _loop(self) -> None:
        while self._running:
            with self._lock:
                status = self._status
                offset = self._tick
            frame = "".join(WAVE[(offset + i) % len(WAVE)] for i in range(self._width))
            # cyan wave + dim status
            payload = f"{_CLEAR_LINE}\x1b[36m{frame}\x1b[0m  \x1b[2m{status}...\x1b[0m"
            try:
                self._stream.write(payload)
                self._stream.flush()
            except Exception:  # noqa: BLE001
                return
            self._tick += 1
            time.sleep(self._interval)

    def __enter__(self) -> "WaveSpinner":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
