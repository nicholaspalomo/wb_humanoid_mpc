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

"""The clang-tidy aspect's self-test: the action of //tools/clang_tidy:selftest (tools/clang_tidy/clang_tidy.bzl).

It fails, and with it `make lint-tidy`, unless:

- clang-tidy recognizes every check and option of the configurations (`--verify-config`, which compiles nothing);
- the findings of each fixture in tools/clang_tidy/testdata, deduplicated as clang_tidy_report.py deduplicates them, are
  exactly its `<name>.expected` file: one `path:line:column: check` per line;
- the fix-its of the rename fixture, applied all or nothing by clang_tidy_apply.py to a copy of its files, give the
  golden copy, file for file.

Usage: python3 -m tools.clang_tidy.clang_tidy_selftest <manifest.json> <stamp>
"""

import difflib
import json
import os
import subprocess
import sys
import tempfile

from tools.clang_tidy import clang_tidy_apply
from tools.clang_tidy import clang_tidy_report

_EXPECTED_SUFFIX = ".expected"


def expected_findings(text: str) -> list[str]:
    """The `path:line:column: check` lines of an expected file, without blank lines and `#` comments.

    Args:
      text: The expected file.

    Returns:
      The lines, stripped.
    """
    lines = [line.strip() for line in text.splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def actual_findings(report_files: list[str]) -> list[str]:
    """The deduplicated findings of `report_files` as `path:line:column: check` lines.

    Args:
      report_files: The fixture's report files.

    Returns:
      The lines, sorted.
    """
    findings, missing = clang_tidy_report.collect(report_files, [])
    lines = [
        f"{finding.path}:{finding.line}:{finding.column}: {finding.check}"
        for finding in findings
    ]
    return lines + [f"(missing report {report})" for report in missing]


def _diff(name: str, expected: list[str], actual: list[str]) -> str:
    return "\n".join(
        difflib.unified_diff(
            expected, actual, f"{name}.expected", f"{name} (clang-tidy)", lineterm=""
        )
    )


def check_configs(wrapper: str, configs: list[str]) -> list[str]:
    """Runs `--verify-config` on each configuration.

    Args:
      wrapper: tools/clang_tidy/run_clang_tidy.sh.
      configs: The configuration files.

    Returns:
      A failure message per configuration clang-tidy rejects.
    """
    failures = []
    for config in configs:
        result = subprocess.run(
            [wrapper, "--verify-config", config],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            failures.append(
                f"{config}: clang-tidy --verify-config failed:\n{result.stdout}{result.stderr}"
            )
    return failures


def check_fixtures(fixtures: list[dict], expected_files: list[str]) -> list[str]:
    """Compares each fixture's findings with its expected file.

    Args:
      fixtures: The manifest's fixtures, each with its `name` and `reports`.
      expected_files: The `<name>.expected` files.

    Returns:
      A failure message per fixture whose findings differ, or that has no expected file, and per expected file that
      names no fixture.
    """
    expected_by_name = {
        os.path.basename(path)[: -len(_EXPECTED_SUFFIX)]: path
        for path in expected_files
    }
    failures = []
    for fixture in fixtures:
        name = fixture["name"]
        if name not in expected_by_name:
            failures.append(f"fixture {name} has no {name}{_EXPECTED_SUFFIX} file")
            continue
        with open(expected_by_name.pop(name), encoding="utf-8") as f:
            expected = expected_findings(f.read())
        actual = actual_findings(fixture["reports"])
        if actual != expected:
            failures.append(
                f"fixture {name}: the findings differ from its expected file:\n{_diff(name, expected, actual)}"
            )
    failures += [
        f"{path} names no fixture of the self-test"
        for path in sorted(expected_by_name.values())
    ]
    return failures


def check_rename(
    fixtures: list[dict],
    renamed_prefix: str,
    golden_prefix: str,
    golden_files: list[str],
) -> list[str]:
    """Applies the fix-its of the files under `renamed_prefix` to a copy of them and compares it with the golden copy.

    Args:
      fixtures: The manifest's fixtures, with their `sources` and `fixes`.
      renamed_prefix: The directory of the fixture whose fix-its are applied, e.g. tools/clang_tidy/testdata/rename/.
      golden_prefix: The directory of its golden copy, e.g. tools/clang_tidy/testdata/renamed/.
      golden_files: The golden copy's files.

    Returns:
      Failure messages; none when the copy and the golden copy agree.
    """
    sources: set[str] = set()
    fixes: list[str] = []
    for fixture in fixtures:
        sources.update(
            source for source in fixture["sources"] if source.startswith(renamed_prefix)
        )
        fixes += fixture["fixes"]
    if not sources:
        return [f"no fixture has files under {renamed_prefix}"]
    failures = []
    with tempfile.TemporaryDirectory() as root:
        for source in sorted(sources):
            os.makedirs(os.path.dirname(os.path.join(root, source)), exist_ok=True)
            with open(source, "rb") as original, open(
                os.path.join(root, source), "wb"
            ) as copy:
                copy.write(original.read())
        result = clang_tidy_apply.plan(
            clang_tidy_apply.load_diagnostics(fixes), ["*"], [renamed_prefix], root
        )
        if result.applied == 0:
            failures.append(f"no fix-it of {renamed_prefix} was applied")
        for (check, path, offset, message), reason in result.refused:
            failures.append(
                f"left alone: {path} (byte {offset}): {message} [{check}]: {reason}"
            )
        clang_tidy_apply.apply(result, root)
        golden_by_path = {
            path[len(golden_prefix) :]: path
            for path in golden_files
            if path.startswith(golden_prefix)
        }
        for source in sorted(sources):
            relative = source[len(renamed_prefix) :]
            if relative not in golden_by_path:
                failures.append(
                    f"{source} has no golden copy {golden_prefix}{relative}"
                )
                continue
            with open(os.path.join(root, source), encoding="utf-8") as f:
                fixed = f.read().splitlines()
            with open(golden_by_path.pop(relative), encoding="utf-8") as f:
                golden = f.read().splitlines()
            if fixed != golden:
                diff = "\n".join(
                    difflib.unified_diff(
                        golden,
                        fixed,
                        golden_prefix + relative,
                        source + " (fixed)",
                        lineterm="",
                    )
                )
                failures.append(
                    f"{source}: the applied fix-its differ from the golden copy:\n{diff}"
                )
        failures += [
            f"{path} is no file of {renamed_prefix}"
            for path in sorted(golden_by_path.values())
        ]
    return failures


def main(argv: list[str]) -> int:
    """Runs the self-test of the manifest `argv[0]` and writes the empty stamp `argv[1]` when it passes."""
    if len(argv) != 2:
        print("usage: clang_tidy_selftest <manifest.json> <stamp>", file=sys.stderr)
        return 2
    manifest_path, stamp = argv
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    failures = check_configs(manifest["wrapper"], manifest["configs"])
    failures += check_fixtures(manifest["fixtures"], manifest["expected"])
    failures += check_rename(
        manifest["fixtures"],
        manifest["renamed_prefix"],
        manifest["golden_prefix"],
        manifest["golden"],
    )
    if failures:
        print(
            "\n\n".join(
                [
                    "❌ The clang-tidy aspect's self-test failed (tools/clang_tidy/README.md):",
                    *failures,
                ]
            ),
            file=sys.stderr,
        )
        return 1
    with open(stamp, "w", encoding="utf-8"):
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
