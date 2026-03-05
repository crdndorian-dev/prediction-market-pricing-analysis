from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


RUNTIME_FILE_NAME = ".job_runtime.json"


@dataclass
class ManagedProcessHandle:
    process: Optional[subprocess.Popen[str]]
    pid: int
    pgid: Optional[int]
    job_id: Optional[str]
    service: Optional[str]
    command: Optional[list[str]]
    run_dir: Optional[Path] = None


@dataclass
class TerminationResult:
    ok: bool
    reason: str
    pid: Optional[int]
    pgid: Optional[int]
    was_alive: bool
    term_sent: bool
    kill_sent: bool
    alive_after_term: bool
    alive_after_kill: bool


def _runtime_file_path(run_dir: Path) -> Path:
    return run_dir / RUNTIME_FILE_NAME


def _safe_get_pgid(pid: int) -> Optional[int]:
    if pid <= 0:
        return None
    if os.name == "nt":
        return None
    try:
        return os.getpgid(pid)
    except Exception:
        return None


def _wait_for_exit(
    *,
    process: Optional[subprocess.Popen[str]],
    pid: int,
    timeout_s: float,
) -> bool:
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            return True
        if not is_process_alive(pid):
            return True
        time.sleep(0.05)
    if process is not None and process.poll() is not None:
        return True
    return not is_process_alive(pid)


def _signal_process(pid: int, pgid: Optional[int], sig: int) -> bool:
    if pid <= 0:
        return False
    try:
        if pgid and pgid > 0 and os.name != "nt":
            try:
                os.killpg(pgid, sig)
            except PermissionError:
                os.kill(pid, sig)
        else:
            os.kill(pid, sig)
        return True
    except ProcessLookupError:
        return False
    except Exception:
        return False


def runtime_payload_for_handle(
    handle: ManagedProcessHandle,
    *,
    status: str = "running",
) -> Dict[str, Any]:
    return {
        "pid": int(handle.pid),
        "pgid": int(handle.pgid) if handle.pgid else None,
        "job_id": handle.job_id,
        "service": handle.service,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": list(handle.command or []),
        "status": status,
    }


def spawn_managed_process(
    cmd: Iterable[str],
    *,
    job_id: str,
    service: str,
    run_dir: Optional[Path] = None,
    cwd: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    stdout: Any = subprocess.PIPE,
    stderr: Any = subprocess.PIPE,
    text: bool = True,
    bufsize: int = 1,
) -> ManagedProcessHandle:
    popen = subprocess.Popen(
        list(cmd),
        stdout=stdout,
        stderr=stderr,
        text=text,
        bufsize=bufsize,
        cwd=str(cwd) if cwd is not None else None,
        env=dict(env) if env is not None else None,
        start_new_session=True,
    )
    handle = ManagedProcessHandle(
        process=popen,
        pid=int(popen.pid),
        pgid=_safe_get_pgid(int(popen.pid)),
        job_id=job_id,
        service=service,
        command=list(cmd),
        run_dir=run_dir,
    )
    if run_dir is not None:
        write_runtime_file(run_dir, runtime_payload_for_handle(handle))
    return handle


def terminate_managed_process(
    handle: ManagedProcessHandle,
    *,
    term_timeout_s: float = 5.0,
    kill_timeout_s: float = 5.0,
) -> TerminationResult:
    pid = int(handle.pid)
    pgid = int(handle.pgid) if handle.pgid else _safe_get_pgid(pid)
    process = handle.process
    was_alive = is_process_alive(pid)
    if not was_alive:
        return TerminationResult(
            ok=True,
            reason="already_exited",
            pid=pid,
            pgid=pgid,
            was_alive=False,
            term_sent=False,
            kill_sent=False,
            alive_after_term=False,
            alive_after_kill=False,
        )

    term_sent = _signal_process(pid, pgid, signal.SIGTERM)
    alive_after_term = not _wait_for_exit(
        process=process,
        pid=pid,
        timeout_s=term_timeout_s,
    )

    kill_sent = False
    alive_after_kill = alive_after_term
    if alive_after_term:
        kill_sent = _signal_process(pid, pgid, signal.SIGKILL)
        alive_after_kill = not _wait_for_exit(
            process=process,
            pid=pid,
            timeout_s=kill_timeout_s,
        )

    if alive_after_kill:
        return TerminationResult(
            ok=False,
            reason="kill_timeout",
            pid=pid,
            pgid=pgid,
            was_alive=True,
            term_sent=term_sent,
            kill_sent=kill_sent,
            alive_after_term=alive_after_term,
            alive_after_kill=True,
        )

    return TerminationResult(
        ok=True,
        reason="terminated" if alive_after_term else "term_graceful",
        pid=pid,
        pgid=pgid,
        was_alive=True,
        term_sent=term_sent,
        kill_sent=kill_sent,
        alive_after_term=alive_after_term,
        alive_after_kill=False,
    )


def is_process_alive(pid: Optional[int]) -> bool:
    if pid is None:
        return False
    pid_int = int(pid)
    if pid_int <= 0:
        return False
    try:
        os.kill(pid_int, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False
    if os.name != "nt":
        try:
            state_proc = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(pid_int)],
                capture_output=True,
                text=True,
                check=False,
            )
            state = (state_proc.stdout or "").strip().upper()
            if not state:
                return False
            if state.startswith("Z"):
                return False
        except Exception:
            pass
    return True


def write_runtime_file(run_dir: Path, payload: Dict[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = _runtime_file_path(run_dir)
    temp_path = runtime_path.with_name(f".{runtime_path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2))
    temp_path.replace(runtime_path)
    return runtime_path


def read_runtime_file(run_dir: Path) -> Optional[Dict[str, Any]]:
    runtime_path = _runtime_file_path(run_dir)
    if not runtime_path.exists():
        return None
    try:
        payload = json.loads(runtime_path.read_text())
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def clear_runtime_file(run_dir: Path) -> None:
    runtime_path = _runtime_file_path(run_dir)
    try:
        runtime_path.unlink(missing_ok=True)
    except Exception:
        return


def managed_handle_from_runtime_payload(
    run_dir: Path,
    payload: Dict[str, Any],
) -> Optional[ManagedProcessHandle]:
    pid_raw = payload.get("pid")
    try:
        pid = int(pid_raw)
    except Exception:
        return None

    pgid_raw = payload.get("pgid")
    pgid: Optional[int]
    try:
        pgid = int(pgid_raw) if pgid_raw is not None else None
    except Exception:
        pgid = None

    command_raw = payload.get("command")
    command: Optional[list[str]] = None
    if isinstance(command_raw, list):
        command = [str(item) for item in command_raw]

    job_id = payload.get("job_id")
    service = payload.get("service")

    return ManagedProcessHandle(
        process=None,
        pid=pid,
        pgid=pgid,
        job_id=str(job_id) if job_id is not None else None,
        service=str(service) if service is not None else None,
        command=command,
        run_dir=run_dir,
    )
