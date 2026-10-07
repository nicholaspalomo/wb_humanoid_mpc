# Copyright (c) 2026, Nicholas Palomo. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Python imports as the Google Python Style Guide writes them (2.2, 2.3): modules, absolute, unaliased, no sys.path.

- `py-import-modules`: `from x import y` imports a module y, never a class or a function (2.2.4): write
  `from remote_control import operator_bus` and then `operator_bus.TopicPublisher`. `typing`, `collections.abc` and
  `typing_extensions` are exempt. A first-party name is a module when `x/y.py` or `x/y/` exists below the repository
  root or a Bazel import root (tools/hooks/lint_files.py), or when it ends in `_pb2`; a standard-library name when
  importlib finds `x.y`; a third-party name when it is in THIRD_PARTY_MODULES, since the lint environments do not install
  those packages.
- `py-relative-import`: no relative imports (`from . import x`, `from .x import y`).
- `py-import-alias`: `import x as y` only with a standard abbreviation (IMPORT_ALIASES: np, tk, rr, ...).
  `from x import y as z` stays allowed for the guide's reasons: a clash, a long or a too generic name.
- `py-sys-path`: no `sys.path` changes outside tests; run a script as a module (`python3 -m tools.hooks.lint_code`).

The checks read the syntax tree (`ast`), so they run under Python 3.11 and 3.12 alike.
"""

import ast
import functools
import importlib.util
import os
import sys

from tools.hooks import check_types
from tools.hooks import lint_files

IMPORT_MODULES = "py-import-modules"
RELATIVE_IMPORT = "py-relative-import"
IMPORT_ALIAS = "py-import-alias"
SYS_PATH = "py-sys-path"

PYTHON = frozenset({check_types.Language.PYTHON})
EXEMPT_MODULES = frozenset(
    {"typing", "collections.abc", "typing_extensions", "__future__"}
)
# The standard abbreviations an import may be aliased to.
IMPORT_ALIASES = frozenset(
    {"np", "jnp", "tk", "rr", "rrb", "pa", "pc", "pd", "plt", "pin"}
)
# Modules of third-party packages that the lint environments do not install, so importlib cannot tell them from names.
THIRD_PARTY_MODULES = frozenset(
    {
        "PIL.Image",
        "brax.envs.base",
        "brax.training.agents.ppo.networks",
        "brax.training.agents.ppo.train",
        "flax.linen",
        "google.protobuf.descriptor",
        "google.protobuf.descriptor_pb2",
        "google.protobuf.descriptor_pool",
        "google.protobuf.internal.api_implementation",
        "google.protobuf.json_format",
        "google.protobuf.message",
        "google.protobuf.message_factory",
        "google.protobuf.text_format",
        "google.protobuf.timestamp_pb2",
        "jax.numpy",
        "mujoco.mjx",
        "mujoco.viewer",
        "rerun.blueprint",
        "rerun.chunk",
        "scipy.spatial",
        "scipy.linalg",
        "tkinter.ttk",
        "tkinter.messagebox",
        "tkinter.filedialog",
        "tkinter.font",
        "unittest.mock",
    }
)

# The repository root, against which first-party modules are resolved; the tests point it elsewhere.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@functools.lru_cache(maxsize=4)
def _roots(root: str) -> tuple[str, ...]:
    try:
        imports = lint_files.python_import_roots(root)
    except OSError:
        imports = []
    return (root,) + tuple(os.path.join(root, r) for r in imports)


def _first_party_package(module: str, here: str) -> str | None:
    """The directory of first-party package `module` (below a root, or next to the importing file), or None."""
    parts = module.split(".")
    for base in (here,) + _roots(ROOT):
        candidate = os.path.join(base, *parts)
        if os.path.isdir(candidate):
            return candidate
    return None


def _is_first_party_module(module: str, here: str) -> bool:
    parts = module.split(".")
    for base in (here,) + _roots(ROOT):
        candidate = os.path.join(base, *parts)
        if os.path.isfile(candidate + ".py") or os.path.isdir(candidate):
            return True
    return False


@functools.lru_cache(maxsize=1024)
def _is_standard_module(name: str) -> bool | None:
    """Whether the standard-library `name` is a module; None when its top-level package is not in the standard library."""
    if name.split(".")[0] not in sys.stdlib_module_names:
        return None
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def _is_module(package: str, name: str, here: str) -> bool:
    """True when `from package import name` imports a module."""
    qualified = f"{package}.{name}"
    if name.endswith(("_pb2", "_pb2_grpc")) or qualified in THIRD_PARTY_MODULES:
        return True
    directory = _first_party_package(package, here)
    if directory is not None:
        return os.path.isfile(os.path.join(directory, name + ".py")) or os.path.isdir(
            os.path.join(directory, name)
        )
    if _is_first_party_module(package, here):
        return False  # a name inside a first-party module
    standard = _is_standard_module(qualified)
    if standard is not None:
        return standard
    return False


def _parse(source: str) -> ast.Module | None:
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def check_import_modules(source: str, path: str) -> list[check_types.Finding]:
    """`from x import y` where y is not a module."""
    tree = _parse(source)
    if tree is None:
        return []
    here = os.path.dirname(os.path.join(ROOT, path))
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
            continue
        if node.module in EXEMPT_MODULES:
            continue
        for alias in node.names:
            if alias.name == "*" or not _is_module(node.module, alias.name, here):
                findings.append(
                    check_types.Finding(
                        path,
                        node.lineno,
                        node.col_offset + 1,
                        IMPORT_MODULES,
                        f"`from {node.module} import {alias.name}` imports a name, not a module: import the module "
                        f"and write `{node.module.rsplit('.', 1)[-1]}.{alias.name}` (Python style 2.2.4).",
                    )
                )
    return findings


def check_relative_import(source: str, path: str) -> list[check_types.Finding]:
    """Relative imports."""
    tree = _parse(source)
    if tree is None:
        return []
    return [
        check_types.Finding(
            path,
            node.lineno,
            node.col_offset + 1,
            RELATIVE_IMPORT,
            "relative import: import by the module's full path (Python style 2.2.4).",
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level
    ]


def check_import_alias(source: str, path: str) -> list[check_types.Finding]:
    """Aliases other than the standard abbreviations."""
    tree = _parse(source)
    if tree is None:
        return []
    findings = []
    for node in ast.walk(tree):
        # `from x import y as z` has its own uses (a clash, a long or generic name); `import y as z` only abbreviates.
        if not isinstance(node, ast.Import):
            continue
        for alias in node.names:
            if (
                alias.asname
                and alias.asname not in IMPORT_ALIASES
                and alias.asname != alias.name
            ):
                findings.append(
                    check_types.Finding(
                        path,
                        node.lineno,
                        node.col_offset + 1,
                        IMPORT_ALIAS,
                        f"`as {alias.asname}`: alias an import only to a standard abbreviation "
                        f"({', '.join(sorted(IMPORT_ALIASES))}) or to avoid a clash (Python style 2.2.4).",
                    )
                )
    return findings


def check_sys_path(source: str, path: str) -> list[check_types.Finding]:
    """`sys.path.insert` / `sys.path.append` / assignments to `sys.path`."""
    tree = _parse(source)
    if tree is None:
        return []
    findings = []
    for node in ast.walk(tree):
        target: ast.expr | None = None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("insert", "append", "extend"):
                target = node.func.value
        elif isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            target = next(
                (t.value if isinstance(t, ast.Subscript) else t for t in targets), None
            )
        if not (
            isinstance(node, (ast.expr, ast.stmt))
            and isinstance(target, ast.Attribute)
            and target.attr == "path"
            and isinstance(target.value, ast.Name)
            and target.value.id == "sys"
        ):
            continue
        findings.append(
            check_types.Finding(
                path,
                node.lineno,
                node.col_offset + 1,
                SYS_PATH,
                "`sys.path` is changed: import first-party modules by their path below a Bazel import root or the "
                "repository root, and run scripts as modules (`python3 -m tools.hooks.lint_code`) (Python style "
                "2.3).",
            )
        )
    return findings


CHECKS = [
    check_types.Check(
        name=IMPORT_MODULES,
        languages=PYTHON,
        scope=lint_files.Scope.FIRST_PARTY,
        check_source=check_import_modules,
        description="import modules, not classes or functions (Python style 2.2.4; decision D7).",
    ),
    check_types.Check(
        name=RELATIVE_IMPORT,
        languages=PYTHON,
        scope=lint_files.Scope.FIRST_PARTY,
        check_source=check_relative_import,
        description="no relative imports (Python style 2.2.4).",
    ),
    check_types.Check(
        name=IMPORT_ALIAS,
        languages=PYTHON,
        scope=lint_files.Scope.FIRST_PARTY,
        check_source=check_import_alias,
        description="import aliases only for the standard abbreviations (Python style 2.2.4).",
    ),
    check_types.Check(
        name=SYS_PATH,
        languages=PYTHON,
        scope=lint_files.Scope.FIRST_PARTY_NON_TEST,
        check_source=check_sys_path,
        description="no sys.path changes outside tests (Python style 2.3).",
    ),
]
