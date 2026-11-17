"""Gunicorn settings, auto-loaded from the working directory.

LLM calls take seconds to tens of seconds (GLM-5.1 ~25s with thinking), so sync
workers would hit gunicorn's 30s timeout and cap concurrency at the worker count.
Threaded workers keep the heartbeat alive during long calls and serve several
requests per process. The worker count comes from the command line (ansible).
"""

worker_class = "gthread"
threads = 8
