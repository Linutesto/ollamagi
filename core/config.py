from pathlib import Path
import os

# All paths and secrets are read from environment variables.
# Copy .env.example → .env and adjust for your machine.
# Defaults make the app work out-of-the-box on a typical Linux system.

_home = Path(os.environ.get("OLLAMAGI_USER_HOME", str(Path.home())))
BASE_DIR = Path(os.environ.get("OLLAMAGI_DIR", str(Path(__file__).parent.parent)))
WORKSPACE_DIR = BASE_DIR / "workspace"

HERMES_DB = Path(os.environ.get("HERMES_DB", str(_home / ".hermes/cognitive_memory.sqlite")))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DOCKER_SOCKET = os.environ.get("DOCKER_SOCKET", "/var/run/docker.sock")

# SSH key injected into agent containers so they can reach the host
SSH_KEY = Path(os.environ.get("SSH_KEY", str(_home / ".ssh/ollamagi_agent")))
# Default Docker bridge gateway — host is reachable at this IP from containers
SSH_HOST = os.environ.get("SSH_HOST", "172.17.0.1")
SSH_USER = os.environ.get("SSH_USER", os.environ.get("USER", "root"))

MODELS = {
    "orchestrator": os.environ.get("MODEL_ORCHESTRATOR", "qwen2.5:32b"),
    "coder":        os.environ.get("MODEL_CODER",        "qwen2.5-coder:32b"),
    "tools":        os.environ.get("MODEL_TOOLS",        "qwen2.5:32b"),
    "analysis":     os.environ.get("MODEL_ANALYSIS",     "qwen2.5:32b"),
    "fast":         os.environ.get("MODEL_FAST",         "qwen2.5:7b"),
    "embeddings":   os.environ.get("MODEL_EMBEDDINGS",   "mxbai-embed-large:latest"),
}

CONTAINER_IMAGES = {
    "pentest": os.environ.get("IMAGE_PENTEST", "vxcontrol/kali-linux:latest"),
    "python":  os.environ.get("IMAGE_PYTHON",  "python:latest"),
    "generic": os.environ.get("IMAGE_GENERIC", "debian:latest"),
}

OLLAMA_CTX = int(os.environ.get("OLLAMA_CTX", "65536"))
MAX_TASK_TIMEOUT = int(os.environ.get("MAX_TASK_TIMEOUT", "300"))

# Hardware info displayed in the dashboard System tab (cosmetic only)
HW_CPU = os.environ.get("HW_CPU", "")
HW_RAM = os.environ.get("HW_RAM", "")
HW_GPU = os.environ.get("HW_GPU", "")

WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
