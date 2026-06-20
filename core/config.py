from pathlib import Path
import os

# Load repository-local configuration before evaluating any settings.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

# All paths and secrets are read from environment variables.
# Copy .env.example → .env and adjust for your machine.
# Defaults make the app work out-of-the-box on a typical Linux system.

_home = Path(os.environ.get("OLLAMAGI_USER_HOME", str(Path.home())))
BASE_DIR = Path(os.environ.get("OLLAMAGI_DIR", str(Path(__file__).parent.parent)))
WORKSPACE_DIR = BASE_DIR / "workspace"

MEMORY_DB = Path(os.environ.get("MEMORY_DB", str(_home / ".ollamagi/cognitive_memory.sqlite"))).expanduser()
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DOCKER_SOCKET = os.environ.get("DOCKER_SOCKET", "/var/run/docker.sock")

# SSH key injected into agent containers so they can reach the host
SSH_KEY = Path(os.environ.get("SSH_KEY", str(_home / ".ssh/ollamagi_agent"))).expanduser()
# Default Docker bridge gateway — host is reachable at this IP from containers
SSH_HOST = os.environ.get("SSH_HOST", "172.17.0.1")
SSH_USER = os.environ.get("SSH_USER", os.environ.get("USER", "root"))

SINGLE_MODEL = os.environ.get(
    "MODEL_SINGLE",
    os.environ.get("MODEL_ORCHESTRATOR", "vaultbox/qwen3.5-uncensored:27b"),
)

# All reasoning, planning, coding, tool, and memory calls use one model. This
# avoids repeated VRAM eviction/reload cycles and makes runtime behavior stable.
MODELS = {
    "orchestrator": SINGLE_MODEL,
    "coder":        SINGLE_MODEL,
    "tools":        SINGLE_MODEL,
    "analysis":     SINGLE_MODEL,
    "fast":         SINGLE_MODEL,
    "embeddings":   os.environ.get("MODEL_EMBEDDINGS", "mxbai-embed-large:latest"),
}

CONTAINER_IMAGES = {
    "pentest": os.environ.get("IMAGE_PENTEST", "vxcontrol/kali-linux:latest"),
    "python":  os.environ.get("IMAGE_PYTHON",  "python:3.11-slim-bookworm"),
    "generic": os.environ.get("IMAGE_GENERIC", "debian:bookworm-slim"),
}

OLLAMA_CTX = int(os.environ.get("OLLAMA_CTX", "65536"))
MAX_TASK_TIMEOUT = int(os.environ.get("MAX_TASK_TIMEOUT", "300"))

# Hardware info displayed in the dashboard System tab (cosmetic only)
HW_CPU = os.environ.get("HW_CPU", "")
HW_RAM = os.environ.get("HW_RAM", "")
HW_GPU = os.environ.get("HW_GPU", "")

WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
