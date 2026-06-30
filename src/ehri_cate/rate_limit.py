"""Thread-safe sliding-window rate limiter.

Paces LLM4SSH calls to a configurable requests-per-minute budget.
When used, one instance should be shared across all models in a run,
because such a cap applies to the API key as a whole, not per model.
"""

from __future__ import annotations

import threading
import time
from collections import deque


class RateLimiter:
    """Allow up to `max_calls` acquisitions in any rolling `period_s` window."""

    def __init__(self, max_calls: int, period_s: float = 60.0):
        if max_calls <= 0:
            raise ValueError("max_calls must be positive")
        self.max_calls = max_calls
        self.period_s = period_s
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

    def acquire(self) -> None:
        """Block until a slot is available, then reserve it."""
        with self._cv:
            while True:
                now = time.monotonic()
                # Evict timestamps that fell out of the window.
                cutoff = now - self.period_s
                while self._calls and self._calls[0] <= cutoff:
                    self._calls.popleft()

                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return

                # Sleep until the oldest in-window call ages out.
                wait_s = self._calls[0] + self.period_s - now
                # cv.wait so other threads can also re-check after evictions.
                self._cv.wait(timeout=max(wait_s, 0.01))
                # Wake the next waiter when a slot frees up.
                self._cv.notify_all()


class NullRateLimiter:
    """A no-op limiter for tests or runs with no enforced cap."""

    def acquire(self) -> None:
        return None
