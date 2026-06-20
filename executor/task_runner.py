"""Execute tasks and stream output."""
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable
from executor.docker_manager import create_container, exec_script, stop_container
from core.model_router import generate_task_script
from core.fractal_memory import context_for_task


@dataclass
class TaskResult:
    task_id: int
    title: str
    status: str  # "success" | "failed" | "timeout"
    exit_code: int
    output: str
    duration: float
    artifacts: list[str] = field(default_factory=list)


def run_task(
    flow_id: str,
    task: dict,
    on_log: Callable[[str], None] | None = None,
) -> TaskResult:
    log = on_log or print
    t0 = time.time()

    log(f"[ollamagi] task {task['id']}: {task['title']}")
    log(f"[ollamagi] generating script via {task.get('type', 'analysis')} model...")

    mem_ctx = context_for_task(task["description"])
    if mem_ctx:
        log(f"[ollamagi] injecting {len(mem_ctx.splitlines())} memory context lines")

    script = generate_task_script(task, context=mem_ctx)
    log(f"[ollamagi] script generated ({len(script)} chars), spinning container...")

    container = None
    try:
        container = create_container(flow_id, task["id"], task.get("container", "python"))
        log(f"[ollamagi] container {container.name} ready, executing...")

        timeout = task.get("timeout", 300)
        exit_code, output = exec_script(container, script, timeout=timeout)

        duration = time.time() - t0
        status = "success" if exit_code == 0 else "failed"
        log(f"[ollamagi] task done: {status} in {duration:.1f}s")

        return TaskResult(
            task_id=task["id"],
            title=task["title"],
            status=status,
            exit_code=exit_code,
            output=output[-8000:],  # last 8k chars
            duration=duration,
        )
    except Exception as e:
        duration = time.time() - t0
        log(f"[ollamagi] task error: {e}")
        return TaskResult(
            task_id=task["id"],
            title=task["title"],
            status="failed",
            exit_code=-1,
            output=str(e),
            duration=duration,
        )
    finally:
        if container:
            stop_container(container)
