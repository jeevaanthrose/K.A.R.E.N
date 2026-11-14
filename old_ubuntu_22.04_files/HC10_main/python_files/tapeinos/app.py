"""
Tapeinos - ROS2 Web Control Dashboard
======================================
(same docstring as before — omitted for brevity)
"""

import atexit
import os
import shlex
import signal
import subprocess
import threading
import time
from collections import deque
from flask import Flask, Response, jsonify, render_template, stream_with_context

# ---------------------------------------------------------------------------
# App Initialization
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ── Register sensor blueprint ───────────────────────────────────────────────
from sensors.routes import sensors_bp, init_sensors, shutdown_all_sensors
app.register_blueprint(sensors_bp)

# ---------------------------------------------------------------------------
# ROS2 Source File
# ---------------------------------------------------------------------------
BASH_SOURCE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bash.source")
HC10_LAUNCH_PACKAGE = os.environ.get("TAPEINOS_HC10_LAUNCH_PACKAGE", "hc10_moveit_config")
HC10_HW_LAUNCH      = os.environ.get("TAPEINOS_HC10_HW_LAUNCH", "yaskawa_hw.launch.py")
HC10_MOVEIT_LAUNCH  = os.environ.get("TAPEINOS_HC10_MOVEIT_LAUNCH", "xy_start.launch.py")
SERVO_PACKAGE       = os.environ.get("TAPEINOS_SERVO_PACKAGE", "cpp_pubsub")
SERVO_ENABLE_EXEC   = os.environ.get("TAPEINOS_SERVO_ENABLE_EXEC", "enable_client")
SERVO_DISABLE_EXEC  = os.environ.get("TAPEINOS_SERVO_DISABLE_EXEC", "disable_client")


def _ros2_source_prefix() -> str:
    if not os.path.exists(BASH_SOURCE_FILE):
        return ""
    lines = []
    with open(BASH_SOURCE_FILE, "r") as f:
        for raw in f:
            line = raw.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return " && ".join(lines)


def _wrap_cmd(cmd: list[str]) -> list[str]:
    prefix  = _ros2_source_prefix()
    cmd_str = shlex.join(cmd)
    if prefix:
        return ["bash", "--login", "-c", f"{prefix} && exec {cmd_str}"]
    return cmd


def _build_env() -> dict:
    env = {
        **os.environ,
        "PYTHONUNBUFFERED":                "1",
        "DISPLAY":                         os.environ.get("DISPLAY", ":0"),
        "XDG_RUNTIME_DIR":                 os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"),
        "QT_QPA_PLATFORM":                 os.environ.get("QT_QPA_PLATFORM", "xcb"),
        "LIBGL_ALWAYS_SOFTWARE":           os.environ.get("LIBGL_ALWAYS_SOFTWARE", ""),
        "RCUTILS_LOGGING_BUFFERED_STREAM": "0",
        "RCL_LOG_LEVEL":                   os.environ.get("RCL_LOG_LEVEL", "INFO"),
    }
    if "XAUTHORITY" in os.environ:
        env["XAUTHORITY"] = os.environ["XAUTHORITY"]
    return env


# ---------------------------------------------------------------------------
# Process Registry & Log Buffers
# ---------------------------------------------------------------------------
PANELS      = ("microros", "servo", "moveit", "jog")
LOG_HISTORY = 200

processes: dict[str, subprocess.Popen] = {}

log_buffers: dict[str, deque] = {p: deque(maxlen=LOG_HISTORY) for p in PANELS}
log_locks:   dict[str, threading.Lock] = {p: threading.Lock()  for p in PANELS}


# ---------------------------------------------------------------------------
# JogRunner singleton — resolved once at startup, reused everywhere
# ---------------------------------------------------------------------------
# Resolved lazily on the first /start_jog call and cached here.
# sensors/routes.py calls jog_once.get_runner() directly and will reach the
# same instance because jog_once caches it as a module-level singleton.
_jog_runner = None
_jog_runner_lock = threading.Lock()


def _get_jog_runner():
    """
    Return the cached JogRunner, initialising it at most once.
    Thread-safe; logs to the 'jog' panel on failure.
    """
    global _jog_runner
    if _jog_runner is not None:
        return _jog_runner
    with _jog_runner_lock:
        if _jog_runner is not None:          # double-checked inside lock
            return _jog_runner
        try:
            import jog_once
            _jog_runner = jog_once.get_runner()
        except Exception as exc:
            _append_log("jog", f"[error] JogRunner init failed: {exc}")
            return None
    return _jog_runner


# ---------------------------------------------------------------------------
# Graceful shutdown  (called at most once)
# ---------------------------------------------------------------------------
_shutdown_called = False
_shutdown_lock   = threading.Lock()


def _shutdown_all() -> None:
    global _shutdown_called
    with _shutdown_lock:
        if _shutdown_called:
            return
        _shutdown_called = True

    running = [p for p in PANELS if processes.get(p) and processes[p].poll() is None]
    if running:
        print(f"\n[tapeinos] shutdown — stopping: {', '.join(running)}", flush=True)
        for panel in running:
            _stop_process(panel)
        print("[tapeinos] all processes stopped.", flush=True)

    # Shut down the jog_once singleton
    try:
        import jog_once
        jog_once.shutdown_runner()
    except Exception:
        pass

    shutdown_all_sensors()


atexit.register(_shutdown_all)


def _signal_handler(signum, frame):
    _shutdown_all()
    raise SystemExit(0)


# Register signal handlers only in the process that actually serves requests.
# Under Werkzeug's reloader the parent process just watches files; the child
# (WERKZEUG_RUN_MAIN=true) is the real server.  Outside debug mode there is
# no reloader so we always register.
if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
    signal.signal(signal.SIGINT,  _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _append_log(panel: str, line: str) -> None:
    with log_locks[panel]:
        log_buffers[panel].append(line)


def _stream_output(panel: str, proc: subprocess.Popen) -> None:
    try:
        for raw in proc.stdout:
            if proc.poll() is not None and not raw:
                break
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            _append_log(panel, line)
    except Exception as exc:
        _append_log(panel, f"[reader error] {exc}")
    finally:
        _append_log(panel, "[process exited]")


def _process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _wait_for_group_exit(pgid: int, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not _process_group_alive(pgid):
            return True
        time.sleep(0.1)
    return not _process_group_alive(pgid)


def _start_process(panel: str, cmd: list[str]) -> dict:
    if panel in processes and processes[panel].poll() is None:
        return {"status": "already_running", "panel": panel}

    try:
        prefix = _ros2_source_prefix()
        if prefix:
            _append_log(panel, "[env] bash.source loaded ✓")
        else:
            _append_log(panel, f"[warn] bash.source not found at {BASH_SOURCE_FILE}")

        wrapped = _wrap_cmd(cmd)
        proc = subprocess.Popen(
            wrapped,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=_build_env(),
            start_new_session=True,
        )
        processes[panel] = proc

        with log_locks[panel]:
            log_buffers[panel].clear()
        _append_log(panel, f"[started] PID {proc.pid} — {' '.join(cmd)}")

        t = threading.Thread(
            target=_stream_output,
            args=(panel, proc),
            daemon=True,
            name=f"reader-{panel}",
        )
        t.start()
        return {"status": "started", "panel": panel, "pid": proc.pid}

    except FileNotFoundError:
        msg = f"[error] Command not found: {cmd[0]}"
        _append_log(panel, msg)
        return {"status": "error", "message": msg}
    except Exception as exc:
        msg = f"[error] {exc}"
        _append_log(panel, msg)
        return {"status": "error", "message": msg}


def _stop_process(panel: str) -> dict:
    proc = processes.get(panel)
    if proc is None or proc.poll() is not None:
        _append_log(panel, "[stopped] (was not running)")
        return {"status": "not_running", "panel": panel}

    try:
        pgid = os.getpgid(proc.pid)
        _append_log(panel, f"[stopping] SIGINT → pgid {pgid} …")
        os.killpg(pgid, signal.SIGINT)

        if not _wait_for_group_exit(pgid, timeout_s=5.0):
            _append_log(panel, f"[stopping] SIGTERM → pgid {pgid} …")
            os.killpg(pgid, signal.SIGTERM)
            if not _wait_for_group_exit(pgid, timeout_s=2.0):
                _append_log(panel, f"[stopping] SIGKILL → pgid {pgid} …")
                os.killpg(pgid, signal.SIGKILL)
                _wait_for_group_exit(pgid, timeout_s=1.0)

        try:
            proc.wait(timeout=0.2)
        except Exception:
            pass

        processes.pop(panel, None)
        if _process_group_alive(pgid):
            msg = f"[warn] process group {pgid} still alive after shutdown"
            _append_log(panel, msg)
            return {"status": "warning", "panel": panel, "message": msg}

        _append_log(panel, "[stopped]")
        return {"status": "stopped", "panel": panel}

    except ProcessLookupError:
        processes.pop(panel, None)
        _append_log(panel, "[stopped] (process group already gone)")
        return {"status": "stopped", "panel": panel}
    except Exception as exc:
        msg = f"[error stopping] {exc}"
        _append_log(panel, msg)
        return {"status": "error", "message": msg}


# ---------------------------------------------------------------------------
# SSE log stream
# ---------------------------------------------------------------------------

def _sse_log_generator(panel: str):
    with log_locks[panel]:
        history = list(log_buffers[panel])
    for line in history:
        yield f"data: {line}\n\n"

    last_len = len(history)
    while True:
        time.sleep(0.25)
        with log_locks[panel]:
            buf = list(log_buffers[panel])
        current_len = len(buf)
        if current_len > last_len:
            for line in buf[last_len:]:
                yield f"data: {line}\n\n"
            last_len = current_len
        else:
            yield ": ping\n\n"


# ---------------------------------------------------------------------------
# Routes – Pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("dashboard/index.html")


@app.route("/sensors")
def sensors():
    return render_template("sensors/index.html")


# ---------------------------------------------------------------------------
# Routes – Log streams (SSE)
# ---------------------------------------------------------------------------

@app.route("/logs/<panel>")
def log_stream(panel: str):
    if panel not in log_buffers:
        return jsonify({"error": "unknown panel"}), 404
    return Response(
        stream_with_context(_sse_log_generator(panel)),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Routes – MicroROS
# ---------------------------------------------------------------------------

@app.route("/start_microros", methods=["POST"])
def start_microros():
    cmd = ["ros2", "launch", HC10_LAUNCH_PACKAGE, HC10_HW_LAUNCH]
    return jsonify(_start_process("microros", cmd))


@app.route("/stop_microros", methods=["POST"])
def stop_microros():
    return jsonify(_stop_process("microros"))


# ---------------------------------------------------------------------------
# Routes – Servo
# ---------------------------------------------------------------------------

@app.route("/start_servo", methods=["POST"])
def start_servo():
    cmd = ["ros2", "run", SERVO_PACKAGE, SERVO_ENABLE_EXEC]
    return jsonify(_start_process("servo", cmd))


@app.route("/stop_servo", methods=["POST"])
def stop_servo():
    DISABLE_CMD     = ["ros2", "run", SERVO_PACKAGE, SERVO_DISABLE_EXEC]
    DISABLE_TIMEOUT = 8
    _append_log("servo", "[stopping] sending disable_client command …")
    disable_ok = False
    try:
        result = subprocess.run(
            _wrap_cmd(DISABLE_CMD),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, timeout=DISABLE_TIMEOUT, env=_build_env(),
        )
        output = result.stdout.decode("utf-8", errors="replace").strip()
        for line in output.splitlines():
            _append_log("servo", f"  {line}")
        if result.returncode == 0:
            _append_log("servo", "[stopping] disable_client exited cleanly ✓")
            disable_ok = True
        else:
            _append_log("servo", f"[warn] disable_client exited {result.returncode}")
    except subprocess.TimeoutExpired:
        _append_log("servo", f"[warn] disable_client timed out after {DISABLE_TIMEOUT}s")
    except FileNotFoundError:
        _append_log("servo", "[error] disable_client not found")
    except Exception as exc:
        _append_log("servo", f"[error] disable_client: {exc}")

    result = _stop_process("servo")
    result["disable_ok"] = disable_ok
    return jsonify(result)


# ---------------------------------------------------------------------------
# Routes – MoveIt
# ---------------------------------------------------------------------------

@app.route("/start_moveit", methods=["POST"])
def start_moveit():
    cmd = ["ros2", "launch", HC10_LAUNCH_PACKAGE, HC10_MOVEIT_LAUNCH]
    return jsonify(_start_process("moveit", cmd))


@app.route("/stop_moveit", methods=["POST"])
def stop_moveit():
    return jsonify(_stop_process("moveit"))


# ---------------------------------------------------------------------------
# Routes – Jog
# ---------------------------------------------------------------------------

@app.route("/start_jog", methods=["POST"])
def start_jog():
    try:
        os.remove("/tmp/jog_state.json")
    except FileNotFoundError:
        pass

    # Eagerly initialise the singleton so it's ready before the first command.
    runner = _get_jog_runner()
    if runner is None:
        _append_log("jog", "[error] JogRunner could not be initialised")
        return jsonify({"status": "error", "message": "JogRunner unavailable"}), 503

    _append_log("jog", "[jog] ready — press a button to move")
    try:
        import jog_once
        _append_log("jog", f"[hc10] joints: {', '.join(jog_once.JOINT_NAMES)}")
        _append_log("jog", f"[hc10] joint topics: {', '.join(jog_once.JOINT_STATE_TOPICS)}")
        _append_log("jog", f"[hc10] status topic: {jog_once.ROBOT_STATUS_TOPIC}")
    except Exception:
        pass
    _append_log("jog", "[jog] default mode: cartesian  (press J to switch to joint)")
    return jsonify({"status": "started", "panel": "jog", "pid": None})


@app.route("/stop_jog", methods=["POST"])
def stop_jog():
    # The JogRunner singleton stays alive so stop-topic subscriptions from
    # ultrasonic sensors remain registered.  We just mark the UI as stopped.
    _append_log("jog", "[jog] stopped")
    return jsonify({"status": "stopped", "panel": "jog"})


# Semaphore to limit concurrent in-flight jog commands to 1.
# The runner's internal _lock already serialises execution, but without this
# semaphore rapid button clicks silently pile up an unbounded thread queue
# that drains long after the user has stopped pressing keys.
_jog_semaphore = threading.Semaphore(1)


@app.route("/jog_cmd/<cmd>", methods=["POST"])
def jog_cmd(cmd: str):
    ALLOWED = set("wsadrfjcop123456!@#$%^")
    if not cmd or cmd[0] not in ALLOWED:
        return jsonify({"status": "error", "message": f"disallowed command: {cmd!r}"}), 400

    key    = cmd[0]
    runner = _get_jog_runner()
    if runner is None:
        msg = "[error] JogRunner not available — is ROS2 installed?"
        _append_log("jog", msg)
        return jsonify({"status": "error", "message": msg}), 503

    # If a jog command is already running, drop this one rather than queuing.
    if not _jog_semaphore.acquire(blocking=False):
        msg = f"[drop] key='{key}' — previous command still executing"
        _append_log("jog", msg)
        return jsonify({"status": "busy", "key": key, "message": msg}), 429

    def _run():
        try:
            ok, message = runner.execute(key)
            if ok:
                _append_log("jog", f"[cmd] key='{key}' → {message}")
            else:
                _append_log("jog", f"[warn] key='{key}' → {message}")
        except Exception as exc:
            _append_log("jog", f"[error] jog_cmd '{key}': {exc}")
        finally:
            _jog_semaphore.release()

    t = threading.Thread(target=_run, daemon=True, name=f"jog-cmd-{key}")
    t.start()
    return jsonify({"status": "sent", "key": key})


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@app.route("/status")
def status():
    result = {}
    for panel in PANELS:
        proc    = processes.get(panel)
        running = proc is not None and proc.poll() is None
        with log_locks[panel]:
            logs = list(log_buffers[panel])
        result[panel] = {
            "state": "running" if running else "stopped",
            "pid":   proc.pid if running else None,
            "logs":  logs,
        }
    # Reflect jog runner liveness without touching private module internals.
    runner_alive = _jog_runner is not None
    if runner_alive:
        result["jog"]["state"] = "running"
    return jsonify(result)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Tapeinos – ROS2 Control Dashboard")
    print("  http://localhost:5000")
    prefix = _ros2_source_prefix()
    if prefix:
        print(f"  bash.source : {BASH_SOURCE_FILE} ✓")
    else:
        print(f"  bash.source : NOT FOUND at {BASH_SOURCE_FILE} !")
    print("=" * 60)

    # Start sensor auto-reconnect after app is fully configured.
    # Called here (not at module level) so it only runs in the serving process,
    # not in the Werkzeug reloader's file-watcher parent process.
    init_sensors(app)

    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True, use_reloader=False)
