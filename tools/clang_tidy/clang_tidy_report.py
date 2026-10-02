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

"""Collects the clang-tidy aspect's reports of a build and prints its findings (tools/clang_tidy/README.md).

    bazel build --config=clang-tidy --keep_going --build_event_json_file=.bazel/clang_tidy_bep.json //...
    python3 -m tools.clang_tidy.clang_tidy_report .bazel/clang_tidy_bep.json [--summary] [--paths DIR ...]

The report files are the `clang_tidy` output group of the build's targets, read from the build event protocol file:
globbing bazel-bin would also pick up the stale reports of deleted files. Paths are made repository-relative, and a
finding that several actions report (a header's, by the header's own action and by every .cpp that includes it) is
printed once. `--paths` keeps the findings located in the given directories, and `--summary` adds counts per check and
per Bazel package. The exit status is 1 when any finding remains.
"""

import argparse
import collections
import dataclasses
import json
import re
import sys
import urllib.parse

from tools.hooks import lint_files

OUTPUT_GROUP = "clang_tidy"

# `path:line:column: warning|error: message [check,-warnings-as-errors]`, or a diagnostic without a location
# (`error: unable to handle compilation [clang-diagnostic-error]`).
_DIAGNOSTIC = re.compile(
    r"^(?:(?P<path>[^\s:][^:]*):(?P<line>\d+):(?P<column>\d+): )?(?P<severity>warning|error): "
    r"(?P<message>.*?)(?: \[(?P<checks>[^\]\s]+)\])?$"
)
# Everything up to the execution root of a Bazel action, in an absolute path.
_EXECROOT = re.compile(r"^.*?/execroot/_main/")


@dataclasses.dataclass(frozen=True)
class Finding:
    """One clang-tidy diagnostic with the lines that follow it (notes, the source line and its caret)."""

    path: str
    line: int
    column: int
    check: str
    message: str
    severity: str = "error"
    details: tuple[str, ...] = ()

    @property
    def key(self) -> tuple[str, int, int, str]:
        """What identifies the finding across actions: its location and check."""
        return (self.path, self.line, self.column, self.check)

    def format(self) -> str:
        """The finding as clang-tidy prints it, with a repository-relative path."""
        location = f"{self.path}:{self.line}:{self.column}: " if self.path else ""
        return "\n".join(
            [f"{location}{self.severity}: {self.message} [{self.check}]", *self.details]
        )


def repository_path(path: str) -> str:
    """`path` relative to the repository root: without an action's execution root, `./` or `file://`.

    Args:
      path: A path as clang-tidy or the build event protocol prints it.

    Returns:
      The repository-relative path; generated files keep their `bazel-out/` prefix and external ones `external/`.
    """
    if path.startswith("file://"):
        path = urllib.parse.unquote(path[len("file://") :])
    path = _EXECROOT.sub("", path)
    while path.startswith("./"):
        path = path[2:]
    return path


def _check_name(checks: str | None, severity: str) -> str:
    if not checks:
        return (
            "clang-diagnostic-error"
            if severity == "error"
            else "clang-diagnostic-warning"
        )
    names = [name for name in checks.split(",") if name and not name.startswith("-")]
    return names[0] if names else checks


def parse_report(text: str) -> list[Finding]:
    """The findings of one report file, each with the lines that follow it up to the next finding.

    Args:
      text: The report, as the aspect's wrapper wrote clang-tidy's output.

    Returns:
      The findings in order of appearance; notes stay with the finding they belong to.
    """
    findings: list[Finding] = []
    for line in text.splitlines():
        match = _DIAGNOSTIC.match(line)
        if match is None:
            if findings:
                findings[-1] = dataclasses.replace(
                    findings[-1], details=findings[-1].details + (line,)
                )
            continue
        findings.append(
            Finding(
                path=repository_path(match.group("path") or ""),
                line=int(match.group("line") or 0),
                column=int(match.group("column") or 0),
                check=_check_name(match.group("checks"), match.group("severity")),
                message=match.group("message"),
                severity=match.group("severity"),
            )
        )
    return findings


def deduplicate(findings: list[Finding]) -> list[Finding]:
    """`findings` sorted, each (path, line, column, check) once.

    Args:
      findings: The findings of every report.

    Returns:
      The first finding of each key, sorted by key.
    """
    unique: dict[tuple[str, int, int, str], Finding] = {}
    for finding in findings:
        unique.setdefault(finding.key, finding)
    return [unique[key] for key in sorted(unique)]


def output_group_files(bep_path: str, output_group: str) -> list[str]:
    """The files of an output group of every target a build completed, from its build event protocol JSON file.

    Args:
      bep_path: The file of `--build_event_json_file`.
      output_group: The output group's name, e.g. OUTPUT_GROUP.

    Returns:
      The absolute paths of the files, sorted.
    """
    named_sets: dict[str, dict] = {}
    roots: list[str] = []
    with open(bep_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            event = json.loads(line)
            event_id = event.get("id", {})
            if "namedSet" in event_id:
                named_sets[event_id["namedSet"]["id"]] = event.get(
                    "namedSetOfFiles", {}
                )
            elif "targetCompleted" in event_id:
                for group in event.get("completed", {}).get("outputGroup", []):
                    if group.get("name") == output_group:
                        roots += [
                            file_set["id"] for file_set in group.get("fileSets", [])
                        ]
    files: set[str] = set()
    seen: set[str] = set()
    pending = list(roots)
    while pending:
        set_id = pending.pop()
        if set_id in seen:
            continue
        seen.add(set_id)
        named_set = named_sets.get(set_id, {})
        for file in named_set.get("files", []):
            uri = file.get("uri", "")
            if uri.startswith("file://"):
                files.add(urllib.parse.unquote(uri[len("file://") :]))
        pending += [file_set["id"] for file_set in named_set.get("fileSets", [])]
    return sorted(files)


def in_paths(path: str, paths: list[str]) -> bool:
    """True when `path` is one of `paths` or lies in one of them (all repository-relative); always for no `paths`.

    Args:
      path: A repository-relative path.
      paths: Repository-relative files or directories.

    Returns:
      Whether the path is selected.
    """
    if not paths:
        return True
    for selected in paths:
        selected = repository_path(selected).rstrip("/")
        if path == selected or path.startswith(selected + "/"):
            return True
    return False


def collect(
    report_files: list[str], paths: list[str]
) -> tuple[list[Finding], list[str]]:
    """The deduplicated findings of `report_files` that lie in `paths`, and the report files that are missing.

    Args:
      report_files: Absolute paths of report files.
      paths: The directories to keep findings from; all for none.

    Returns:
      The findings, and the report files that could not be read (an action that failed).
    """
    findings: list[Finding] = []
    missing: list[str] = []
    for report in report_files:
        try:
            with open(report, encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            missing.append(report)
            continue
        findings += parse_report(text)
    selected = [
        finding for finding in deduplicate(findings) if in_paths(finding.path, paths)
    ]
    return selected, missing


def summary(findings: list[Finding], root: str) -> list[str]:
    """Counts per check and per Bazel package, most first.

    Args:
      findings: Deduplicated findings.
      root: The repository root, where the Bazel packages are looked up.

    Returns:
      The lines to print.
    """
    by_check = collections.Counter(finding.check for finding in findings)
    by_package = collections.Counter(
        lint_files.bazel_package(finding.path, root) if finding.path else "(no file)"
        for finding in findings
    )
    lines = ["Findings per check:"]
    lines += [
        f"  {count:6d}  {name}"
        for name, count in sorted(
            by_check.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    lines.append("Findings per package:")
    lines += [
        f"  {count:6d}  {name}"
        for name, count in sorted(
            by_package.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    return lines


def main(argv: list[str] | None = None) -> int:
    """Prints the findings of a clang-tidy build and returns 1 when there are any."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "bep", help="the --build_event_json_file of the clang-tidy build"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="also print counts per check and per package",
    )
    parser.add_argument(
        "--paths", nargs="*", default=[], help="keep only findings in these directories"
    )
    parser.add_argument(
        "--root",
        default=".",
        help="the repository root (default: the working directory)",
    )
    arguments = parser.parse_args(argv)

    findings, missing = collect(
        output_group_files(arguments.bep, OUTPUT_GROUP), arguments.paths
    )
    for finding in findings:
        print(finding.format())
    for report in missing:
        print(
            f"clang-tidy: no report at {report}: its action failed (see the build output above).",
            file=sys.stderr,
        )
    if arguments.summary and findings:
        print("\n".join(summary(findings, arguments.root)))
    if findings:
        files = len({finding.path for finding in findings})
        print(f"❌ clang-tidy: {len(findings)} findings in {files} files.")
        return 1
    print("✅ clang-tidy: no findings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
