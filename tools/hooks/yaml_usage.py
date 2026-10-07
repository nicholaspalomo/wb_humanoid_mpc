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


"""Keeps YAML out of the MPC's configuration: no YAML file where the configuration lives, and no yaml-cpp.

The MPC's hyperparameter files are typed textprotos (humanoid_nmpc/humanoid_mpc_config/README.md), read strictly into
the structs nproto generates from their schemas. Two checks keep the YAML that they replaced from coming back:

  config-yaml  a `.yaml` or `.yml` file under robot_models/ or humanoid_nmpc/, where the configuration files live. A
               YAML file there would be read by nothing, or by a loader that is not the typed one. Git-ignored files
               (the stale `<file>.live.yaml` copies a working tree may still hold) are not files of the repository,
               so the linter never sees them.
  yaml-cpp     an #include (or #include_next) of a header under yaml-cpp/, in quotes or angle brackets, in C++; and a
               label of the yaml-cpp repository (@yaml_cpp, @@yaml_cpp) in a Bazel file. The repository is no longer
               registered (bazel/system_libs.bzl), so a label would fail anyway; the check says why.

    python3 -m tools.hooks.lint_code --only config-yaml,yaml-cpp

`python3 -m tools.hooks.lint_code` (make lint) and the pre-commit hook run both as registry checks.
"""

import re

from tools.hooks import check_types
from tools.hooks import cpp_source
from tools.hooks import lint_files
from tools.hooks import nolint

CONFIG_YAML = "config-yaml"
YAML_CPP = "yaml-cpp"

# The directories that hold the MPC's configuration files.
CONFIG_DIRS = ("robot_models/", "humanoid_nmpc/")
YAML_EXTENSIONS = (".yaml", ".yml")

_INCLUDE = re.compile(
    r'^[ \t]*#[ \t]*include(?:_next)?[ \t]*([<"])(yaml-cpp/[^>"\n]*)[>"]', re.MULTILINE
)
_LABEL = re.compile(r"""["']@{1,2}yaml_cpp(?:\b|//)[^"'\n]*["']""")


def is_config_yaml(path: str) -> bool:
    """True for a YAML file in one of the configuration directories (`path` relative to the repository root)."""
    path = lint_files.normalize(path)
    return path.startswith(CONFIG_DIRS) and path.lower().endswith(YAML_EXTENSIONS)


def _first_content_line(source: str) -> int:
    """The 1-based number of the first line of `source` that is neither blank nor a `#` comment, or 1 when none is."""
    for number, line in enumerate(source.split("\n"), start=1):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return number
    return 1


def _config_yaml_findings(source: str, path: str) -> list[check_types.Finding]:
    if not is_config_yaml(path):
        return []
    # The file is refused whatever it holds; the finding stands on its first line of content, where a NOLINT marker
    # of the file (on that line, or alone on the line above it) applies.
    return [
        check_types.Finding(
            path,
            _first_content_line(source),
            1,
            CONFIG_YAML,
            "a YAML file where the MPC's configuration lives; its files are typed textprotos "
            "(humanoid_nmpc/humanoid_mpc_config/README.md), and nothing reads YAML.",
        )
    ]


def _line_column(source: str, offset: int) -> tuple[int, int]:
    line = source.count("\n", 0, offset) + 1
    return line, offset - (source.rfind("\n", 0, offset) + 1) + 1


def _cpp_findings(source: str, path: str) -> list[check_types.Finding]:
    """The yaml-cpp headers the #include directives of the C++ `source` name, outside comments."""
    without_comments, _ = cpp_source.mask(source)
    findings = []
    for match in _INCLUDE.finditer(without_comments):
        line, column = _line_column(source, match.start(2) - 1)
        findings.append(
            check_types.Finding(
                path,
                line,
                column,
                YAML_CPP,
                f"`{match.group(2)}` is a yaml-cpp header.",
            )
        )
    return findings


def _starlark_findings(source: str, path: str) -> list[check_types.Finding]:
    """The labels of the yaml-cpp repository in the Bazel file `source`, outside `#` comments."""
    findings = []
    for number, line in enumerate(source.split("\n"), start=1):
        start = nolint.hash_comment_start(line)
        code = line if start is None else line[:start]
        for match in _LABEL.finditer(code):
            findings.append(
                check_types.Finding(
                    path,
                    number,
                    match.start() + 1,
                    YAML_CPP,
                    f"{match.group(0)} names the yaml-cpp repository.",
                )
            )
    return findings


def _yaml_cpp_findings(source: str, path: str) -> list[check_types.Finding]:
    language = check_types.language_of(path)
    if language == check_types.Language.CPP:
        return _cpp_findings(source, path)
    if language == check_types.Language.STARLARK:
        return _starlark_findings(source, path)
    return []


CHECKS = [
    check_types.Check(
        name=CONFIG_YAML,
        languages=frozenset({check_types.Language.TEXT}),
        scope=lint_files.Scope.FIRST_PARTY,
        check_source=_config_yaml_findings,
        description="no .yaml or .yml file under robot_models/ or humanoid_nmpc/: the MPC's configuration files are "
        "typed textprotos (humanoid_nmpc/humanoid_mpc_config).",
        hint="Write the configuration as a textproto of its schema (humanoid_nmpc/humanoid_mpc_config/README.md).",
    ),
    check_types.Check(
        name=YAML_CPP,
        languages=frozenset({check_types.Language.CPP, check_types.Language.STARLARK}),
        scope=lint_files.Scope.NOT_THIRDPARTY,
        check_source=_yaml_cpp_findings,
        description="no yaml-cpp: no #include of a yaml-cpp/ header and no @yaml_cpp label; the configuration is read "
        "by nproto's strict textproto parser.",
        hint="Read configuration through its typed textproto (humanoid_common_mpc/config/ConfigFiles.h, "
        "nproto::LoadTextprotoFile()).",
    ),
]
