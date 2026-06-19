"""Docker container lifecycle for OllamAGI."""
import docker
import io
import os
import queue
import tarfile
import threading
import time
from pathlib import Path
from core.config import (
    CONTAINER_IMAGES, WORKSPACE_DIR, SSH_KEY, SSH_HOST, SSH_USER, DOCKER_SOCKET
)

# Bind-mount the host home directory so agents can read/write persistent files
HOME_DIR = os.environ.get("OLLAMAGI_USER_HOME", str(Path.home()))
_client = None


def _docker():
    global _client
    if _client is None:
        _client = docker.from_env()
    return _client


def create_container(flow_id: str, task_id: str, container_type: str = "python"):
    image = CONTAINER_IMAGES.get(container_type, CONTAINER_IMAGES["generic"])
    work_dir = WORKSPACE_DIR / flow_id
    work_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale container with same name if any
    name = f"ollamagi-{flow_id}-{task_id}"
    try:
        old = _docker().containers.get(name)
        old.remove(force=True)
    except docker.errors.NotFound:
        pass

    container = _docker().containers.run(
        image=image,
        name=name,
        command="tail -f /dev/null",
        detach=True,
        remove=False,
        volumes={
            str(work_dir): {"bind": "/work", "mode": "rw"},
            HOME_DIR:       {"bind": HOME_DIR, "mode": "rw"},   # full home access
            DOCKER_SOCKET:  {"bind": "/var/run/docker.sock", "mode": "rw"},
        },
        network_mode="bridge",
        cap_add=["NET_ADMIN", "NET_RAW"],
    )
    time.sleep(1)
    _inject_ssh(container)
    _bootstrap_python(container, container_type)
    return container


# Common packages pre-installed so agents don't need to pip-install basics every run
_COMMON_PACKAGES = (
    "uv requests httpx aiohttp python-dotenv rich colorama "
    "beautifulsoup4 lxml pyyaml toml psutil loguru selenium webdriver-manager"
)

def _bootstrap_python(container, container_type: str):
    if container_type == "pentest":
        return  # Kali has its own package management
    # Bootstrap must finish before generated code starts. The old detached setup
    # raced the task and regularly produced "python3: command not found".
    if container_type == "generic":
        command = (
            "if ! command -v python3 >/dev/null 2>&1; then "
            "export DEBIAN_FRONTEND=noninteractive; "
            "apt-get update -qq && "
            "apt-get install -y -qq python3 python3-pip ca-certificates curl; "
            "fi; command -v python3"
        )
    else:
        command = (
            "command -v python3 && "
            f"python3 -m pip install -q --disable-pip-version-check --prefer-binary {_COMMON_PACKAGES}"
        )

    exit_code, output = _exec_with_timeout(
        container, ["bash", "-c", command], timeout=240, user="root"
    )
    if exit_code != 0:
        raise RuntimeError(f"container bootstrap failed (exit {exit_code}): {output[-1200:]}")


def _inject_ssh(container) -> bool:
    try:
        container.exec_run("mkdir -p /root/.ssh", user="root")
        _docker().api.put_archive(container.id, "/root/.ssh", _tar_file(SSH_KEY))
        container.exec_run("chmod 600 /root/.ssh/id_ed25519", user="root")
        # SSH config
        cfg = (f"Host {SSH_HOST}\n"
               f"  StrictHostKeyChecking no\n"
               f"  User {SSH_USER}\n"
               f"  IdentityFile /root/.ssh/id_ed25519\n")
        container.exec_run(
            ["bash", "-c", f"printf '{cfg}' > /root/.ssh/config && chmod 600 /root/.ssh/config"],
            user="root"
        )
        result = container.exec_run(['ssh', SSH_HOST, 'echo ok'], user="root")
        ok = result.output and b"ok" in result.output
        return ok
    except Exception as e:
        print(f"[ssh-inject] {e}")
        return False


def _tar_file(path: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        tar.add(str(path), arcname=path.name)
    buf.seek(0)
    return buf.read()


def _script_tar(script: str) -> bytes:
    buf = io.BytesIO()
    content = script.encode()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name="ollamagi_task.sh")
        info.size = len(content)
        info.mode = 0o755
        tar.addfile(info, io.BytesIO(content))
    buf.seek(0)
    return buf.read()


def _python_tar(code: str) -> bytes:
    buf = io.BytesIO()
    content = code.encode("utf-8")
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name="ollamagi_task.py")
        info.size = len(content)
        info.mode = 0o755
        tar.addfile(info, io.BytesIO(content))
    buf.seek(0)
    return buf.read()


def _exec_with_timeout(container, command: list[str], timeout: int, user: str | None = None) -> tuple[int, str]:
    """Run a Docker exec with a real wall-clock timeout."""
    result_q: queue.Queue = queue.Queue(maxsize=1)

    def worker():
        try:
            kwargs = {"stream": False, "demux": False}
            if user:
                kwargs["user"] = user
            result_q.put(("ok", container.exec_run(command, **kwargs)))
        except BaseException as exc:
            result_q.put(("error", exc))

    threading.Thread(target=worker, daemon=True).start()
    try:
        kind, value = result_q.get(timeout=max(1, timeout))
    except queue.Empty:
        try:
            container.kill()
        except Exception:
            pass
        return 124, f"Command timed out after {timeout} seconds"

    if kind == "error":
        raise value
    output = value.output.decode("utf-8", errors="replace") if value.output else ""
    return value.exit_code if value.exit_code is not None else -1, output


def _strict_bash(script: str) -> str:
    strict = "set -euo pipefail"
    if strict in script:
        return script
    lines = script.splitlines()
    if lines and lines[0].startswith("#!"):
        return "\n".join([lines[0], strict, *lines[1:]])
    return f"{strict}\n{script}"


def exec_script(container, script: str, timeout: int = 300) -> tuple[int, str]:
    script = _strict_bash(script)
    _docker().api.put_archive(container.id, "/tmp", _script_tar(script))
    return _exec_with_timeout(container, ["bash", "/tmp/ollamagi_task.sh"], timeout)


def exec_python(container, code: str, timeout: int = 300) -> tuple[int, str]:
    """Write Python code directly (no bash wrapper) and execute with python3."""
    _docker().api.put_archive(container.id, "/tmp", _python_tar(code))
    return _exec_with_timeout(container, ["python3", "/tmp/ollamagi_task.py"], timeout)


def exec_command(container, cmd: str) -> tuple[int, str]:
    result = container.exec_run(["bash", "-c", cmd], stream=False, demux=False)
    output = result.output.decode("utf-8", errors="replace") if result.output else ""
    return result.exit_code or 0, output


def stop_container(container):
    try:
        # Agent processes run as root for package installation. Return generated
        # artifacts to the host service user so later flows and manual edits work.
        container.exec_run(
            ["chown", "-R", f"{os.getuid()}:{os.getgid()}", "/work"],
            user="root",
        )
    except Exception:
        pass
    try:
        container.stop(timeout=5)
        container.remove(force=True)
    except Exception:
        pass


def sync_workspace(flow_id: str) -> list[Path]:
    work_dir = WORKSPACE_DIR / flow_id
    return list(work_dir.rglob("*")) if work_dir.exists() else []
