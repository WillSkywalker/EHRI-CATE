from __future__ import annotations

import threading
import time

from ehri_cate.rate_limit import RateLimiter


def test_under_limit_returns_immediately():
    rl = RateLimiter(max_calls=5, period_s=10.0)
    t0 = time.monotonic()
    for _ in range(5):
        rl.acquire()
    assert time.monotonic() - t0 < 0.05


def test_blocks_when_limit_reached():
    # 3 calls allowed per 0.3s window. 4th call must wait ~0.3s for the first to age out.
    rl = RateLimiter(max_calls=3, period_s=0.3)
    rl.acquire()
    rl.acquire()
    rl.acquire()
    t0 = time.monotonic()
    rl.acquire()
    elapsed = time.monotonic() - t0
    assert 0.25 < elapsed < 0.6, f"expected ~0.3s wait, got {elapsed:.2f}s"


def test_threaded_throughput_is_capped():
    rl = RateLimiter(max_calls=4, period_s=0.5)

    def worker():
        rl.acquire()

    threads = [threading.Thread(target=worker) for _ in range(8)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - t0
    # 8 calls at 4/0.5s → second batch must wait for first to age out.
    assert 0.4 < elapsed < 1.0, f"expected ~0.5s for 8 calls at 4/0.5s, got {elapsed:.2f}s"
