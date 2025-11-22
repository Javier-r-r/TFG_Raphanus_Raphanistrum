"""Background worker utilities.

This module provides two helpers used by the segmentation app:
- a shared ThreadPoolExecutor to run callables without creating raw threads
- a helper to run subprocesses while streaming stdout/stderr into a Queue

Both helpers optionally post status messages into an ``out_q`` queue
using tuples of the form ``(type, payload)`` where ``type`` is one of
``"log"``, ``"done"`` or ``"error"``.
"""

import threading
import concurrent.futures
import subprocess
import queue
import os

from typing import Callable, Tuple, Dict


_CPU_COUNT = os.cpu_count() or 2
_MAX_WORKERS = max(1, _CPU_COUNT - 1)
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="bgworker")


def run_callable_in_thread(fn: Callable, args: Tuple = (), kwargs: Dict = None, out_q: queue.Queue = None):
    """Run a callable in the shared ThreadPoolExecutor.

    The callable is executed as ``fn(*args, **kwargs)``. Progress and
    completion are reported via ``out_q`` if provided. The queue receives
    a ``("log", "Task started")`` when the task begins, ``("done", result)``
    when it finishes successfully, or ``("error", str(e))`` if an
    exception is raised.

    Args:
        fn: Callable to execute.
        args: Positional arguments tuple for ``fn``.
        kwargs: Keyword arguments for ``fn``.
        out_q: Optional ``queue.Queue`` to receive log/done/error messages.

    Returns:
        concurrent.futures.Future: Future for the submitted task.
    """
    if kwargs is None:
        kwargs = {}

    def _target():
        try:
            if out_q:
                out_q.put(("log", "Task started"))
            result = fn(*args, **kwargs)
            if out_q:
                out_q.put(("done", result))
        except Exception as e:
            if out_q:
                out_q.put(("error", str(e)))

    future = _executor.submit(_target)
    return future


def run_subprocess(cmd: list, out_q: queue.Queue = None, cwd: str = None, env: dict = None):
    """Start a subprocess and stream its output to ``out_q`` if provided.

    The function launches the subprocess with merged stdout/stderr and
    starts a daemon thread that reads lines from the process' stdout and
    posts them as ``("log", line)``. When the process exits a final
    ``("done", {"returncode": rc})`` message is posted. If starting or
    reading the process fails, an ``("error", str(e))`` message is sent.

    Args:
        cmd: List with the command and arguments to execute.
        out_q: Optional ``queue.Queue`` to receive log/done/error messages.
        cwd: Working directory for the subprocess.
        env: Environment mapping for the subprocess.

    Returns:
        tuple(process, thread) or None: The subprocess.Popen instance and
        the reader thread, or None if process launch failed.
    """
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            env=env or os.environ,
            universal_newlines=True,
            bufsize=1,
        )
    except Exception as e:
        if out_q:
            out_q.put(("error", f"Failed to start subprocess: {e}"))
        return None

    def _reader():
        try:
            for line in proc.stdout:
                if out_q:
                    out_q.put(("log", line.rstrip()))
            ret = proc.wait()
            if out_q:
                out_q.put(("done", {"returncode": ret}))
        except Exception as e:
            if out_q:
                out_q.put(("error", str(e)))

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    return proc, t