import threading
import concurrent.futures
import os
import subprocess
import queue
import os
from typing import Callable, Any, Tuple, Dict



# --- ThreadPoolExecutor global para limitar hilos ---
_CPU_COUNT = os.cpu_count() or 2
_MAX_WORKERS = max(1, _CPU_COUNT - 1)
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="bgworker")

def run_callable_in_thread(fn: Callable, args: Tuple = (), kwargs: Dict = None, out_q: queue.Queue = None):
    """Ejecuta fn(*args, **kwargs) en un ThreadPoolExecutor y publica mensajes en out_q."""
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

    # Usar el pool en vez de crear hilos sueltos
    future = _executor.submit(_target)
    return future


def run_subprocess(cmd: list, out_q: queue.Queue = None, cwd: str = None, env: dict = None):
    """Lanza un subprocess y publica stdout/exit en out_q. Útil para entrenamientos/procesos pesados."""
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