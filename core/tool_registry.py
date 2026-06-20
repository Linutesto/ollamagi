"""
Tool Registry — standardized tool interface for OllamAGI agents.

Usage:
    from core.tool_registry import tool, registry

    @tool("web_search", "Search the web", {
        "query": {"type": "string"},
        "max_results": {"type": "integer", "default": 10},
    })
    def web_search(query: str, max_results: int = 10) -> list[dict]:
        ...

    # Discover plugins from a directory
    registry.load_plugins(Path("plugins"))

    # Get Ollama-compatible schemas
    schemas = registry.ollama_schemas()
"""
import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any, Callable


class ToolDef:
    def __init__(self, name: str, description: str,
                 parameters: dict, fn: Callable):
        self.name = name
        self.description = description
        self.parameters = parameters  # JSON Schema object properties
        self.fn = fn

    def ollama_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": [
                        k for k, v in self.parameters.items()
                        if not isinstance(v.get("default"), type(None))
                        and "default" not in v
                    ],
                },
            },
        }

    def call(self, **kwargs) -> Any:
        return self.fn(**kwargs)


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolDef] = {}

    def register(self, name: str, description: str,
                 parameters: dict, fn: Callable) -> "ToolRegistry":
        self._tools[name] = ToolDef(name, description, parameters, fn)
        return self

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def all(self) -> list[ToolDef]:
        return list(self._tools.values())

    def ollama_schemas(self) -> list[dict]:
        return [t.ollama_schema() for t in self._tools.values()]

    def load_plugins(self, directory: Path):
        """Auto-import all .py files in directory; functions decorated with @tool are registered."""
        if not directory.exists():
            return
        for path in sorted(directory.glob("*.py")):
            if path.name.startswith("_"):
                continue
            spec = importlib.util.spec_from_file_location(path.stem, path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                try:
                    # Inject registry into the module namespace before executing
                    mod.__dict__["_registry"] = self
                    spec.loader.exec_module(mod)
                except Exception as e:
                    print(f"[tool_registry] failed to load plugin {path.name}: {e}")

    def call(self, name: str, arguments: dict | str) -> Any:
        """Execute a registered tool by name. arguments can be a JSON string or dict."""
        t = self._tools.get(name)
        if not t:
            raise KeyError(f"No tool named '{name}'")
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        return t.call(**arguments)

    def summary(self) -> str:
        lines = []
        for t in self._tools.values():
            params = ", ".join(
                f"{k}: {v.get('type', 'any')}" for k, v in t.parameters.items()
            )
            lines.append(f"  • {t.name}({params}) — {t.description}")
        return "\n".join(lines)


# Module-level decorator + singleton
registry = ToolRegistry()


def tool(name: str, description: str, parameters: dict | None = None):
    """Decorator to register a function as a tool."""
    def decorator(fn: Callable) -> Callable:
        # Auto-derive parameters from type hints if not provided
        params = parameters or {}
        if not params:
            sig = inspect.signature(fn)
            hints = fn.__annotations__
            for pname, param in sig.parameters.items():
                if pname == "return":
                    continue
                ptype = hints.get(pname, str)
                type_map = {str: "string", int: "integer", float: "number",
                            bool: "boolean", list: "array", dict: "object"}
                entry: dict = {"type": type_map.get(ptype, "string")}
                if param.default is not inspect.Parameter.empty:
                    entry["default"] = param.default
                doc = (fn.__doc__ or "").strip()
                if doc and pname in doc:
                    # Crude: extract one-liner after param name in docstring
                    pass
                params[pname] = entry
        registry.register(name, description, params, fn)
        fn._tool_name = name
        return fn
    return decorator
