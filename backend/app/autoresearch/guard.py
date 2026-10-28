from __future__ import annotations

import ast

MAX_BYTES = 20_000
ALLOWED_MODULES = {"numpy", "scipy", "sklearn", "math", "functools", "itertools", "collections", "typing",
                   "dataclasses", "__future__", "numbers", "statistics"}
BLOCKED_MODULE_PREFIXES = ("sklearn.datasets", "numpy.ctypeslib", "numpy.f2py", "numpy.distutils", "scipy.io",
                           "numpy.lib.npyio")
BLOCKED_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr", "setattr",
                 "delattr", "input", "breakpoint", "exit", "quit", "memoryview", "help", "print"}
BLOCKED_ATTRS = {"datasets", "load", "loadtxt", "fromfile", "save", "savez", "savez_compressed", "savetxt",
                 "tofile", "memmap", "genfromtxt", "ctypeslib", "f2py", "npyio", "system", "popen", "dump",
                 "dumps", "io", "DataSource"}


def check_source(source: str) -> list[str]:
    if len(source.encode()) > MAX_BYTES:
        return [f"train.py is larger than {MAX_BYTES} bytes"]
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error on line {exc.lineno}: {exc.msg}"]
    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                errors += _check_module(alias.name, node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                errors.append(f"line {node.lineno}: relative imports are not allowed")
                continue
            errors += _check_module(node.module or "", node.lineno)
            for alias in node.names:
                if alias.name == "*":
                    errors.append(f"line {node.lineno}: star imports are not allowed")
                errors += _check_module(f"{node.module}.{alias.name}", node.lineno, root_checked=True)
        elif isinstance(node, ast.Name) and (node.id in BLOCKED_NAMES or _dunder(node.id)):
            errors.append(f"line {node.lineno}: use of {node.id!r} is not allowed")
        elif isinstance(node, ast.Attribute) and (node.attr in BLOCKED_ATTRS or _dunder(node.attr)):
            errors.append(f"line {node.lineno}: attribute {node.attr!r} is not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            errors.append(f"line {node.lineno}: global/nonlocal statements are not allowed")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "build_model" for n in tree.body):
        errors.append("train.py must define a top-level build_model(seed) function")
    return sorted(set(errors))


def _dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__") and name != "__init__"


def _check_module(name: str, lineno: int, root_checked: bool = False) -> list[str]:
    if not root_checked and name.split(".")[0] not in ALLOWED_MODULES:
        return [f"line {lineno}: import of {name!r} is not allowed"]
    if any(name == p or name.startswith(p + ".") for p in BLOCKED_MODULE_PREFIXES):
        return [f"line {lineno}: import of {name!r} is not allowed"]
    return []


def count_code_lines(source: str) -> int:
    return sum(1 for line in source.splitlines() if line.strip() and not line.strip().startswith("#"))
