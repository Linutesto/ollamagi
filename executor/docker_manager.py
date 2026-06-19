"""Docker container lifecycle for OllamAGI."""
import docker
import io
import os
import tarfile
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
    "beautifulsoup4 lxml pyyaml toml psutil"
)

def _bootstrap_python(container, container_type: str):
    if container_type == "pentest":
        return  # Kali has its own package management
    # Generic/debian containers don't have python3 by default — install it first.
    # python:latest already has it, so this is a no-op there.
    pre = "apt-get install -y -qq python3 python3-pip curl 2>/dev/null || true && "
    container.exec_run(
        ["bash", "-c",
         pre +
         "python3 -m pip install -q --upgrade pip 2>/dev/null && "
         f"python3 -m pip install -q --prefer-binary {_COMMON_PACKAGES} 2>/dev/null && "
         "curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null && "
         "ln -sf /root/.local/bin/uv /usr/local/bin/uv 2>/dev/null || true"],
        user="root",
        detach=True,
    )


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


def exec_script(container, script: str, timeout: int = 300) -> tuple[int, str]:
    _docker().api.put_archive(container.id, "/tmp", _script_tar(script))
    result = container.exec_run(["bash", "/tmp/ollamagi_task.sh"], stream=False, demux=False)
    output = result.output.decode("utf-8", errors="replace") if result.output else ""
    return result.exit_code or 0, output


def exec_python(container, code: str, timeout: int = 300) -> tuple[int, str]:
    """Write Python code directly (no bash wrapper) and execute with python3."""
    _docker().api.put_archive(container.id, "/tmp", _python_tar(code))
    result = container.exec_run(["python3", "/tmp/ollamagi_task.py"], stream=False, demux=False)
    output = result.output.decode("utf-8", errors="replace") if result.output else ""
    return result.exit_code or 0, output


def exec_command(container, cmd: str) -> tuple[int, str]:
    result = container.exec_run(["bash", "-c", cmd], stream=False, demux=False)
    output = result.output.decode("utf-8", errors="replace") if result.output else ""
    return result.exit_code or 0, output


def stop_container(container):
    try:
        container.stop(timeout=5)
        container.remove(force=True)
    except Exception:
        pass


def sync_workspace(flow_id: str) -> list[Path]:
    work_dir = WORKSPACE_DIR / flow_id
    return list(work_dir.rglob("*")) if work_dir.exists() else []
