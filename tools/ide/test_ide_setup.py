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

"""Pins what keeps the editor's view of the build current (tools/ide/README.md).

compile_commands.json is refreshed by every Makefile target that builds, by the launch and deployment builds and by the
dev container's start, and left alone in CI; pyrightconfig.json names the Bazel import roots of the BUILD files. A new
target that calls Bazel directly, or a new `imports`, fails here instead of leaving the editor stale.
"""

import json
import os
import re
import unittest

from tools.hooks import lint_files

# LINT.IfChange(bazel_indexed)
_RUNNER = "python3 -m tools.ide.compile_commands --after bazel"
# LINT.ThenChange(//Makefile:bazel_indexed)
# A Bazel command that builds, spelled out. A recipe writes `$(bazel_indexed) build` instead, which this does not match.
_BUILDING_COMMAND = re.compile(r"(?<![\w/.-])bazel (build|test|run)\b")


def _root() -> str:
    """Returns the checkout, through MODULE.bazel's runfiles link under Bazel and the tree this file is in otherwise."""
    candidates = []
    if "TEST_SRCDIR" in os.environ:
        candidates.append(
            os.path.join(os.environ["TEST_SRCDIR"], "_main", "MODULE.bazel")
        )
    candidates.append(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "MODULE.bazel"
        )
    )
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.dirname(os.path.realpath(candidate))
    raise FileNotFoundError("MODULE.bazel")


def _read(relative_path: str) -> str:
    """Returns the content of a file of the checkout."""
    with open(os.path.join(_root(), relative_path), encoding="utf-8") as f:
        return f.read()


def _makefile_recipe_lines() -> list[tuple[int, str]]:
    """Returns the Makefile's recipe lines, with their line numbers, continuation lines included."""
    lines = []
    in_recipe = False
    for number, line in enumerate(_read("Makefile").splitlines(), start=1):
        if line.startswith("\t"):
            in_recipe = True
            lines.append((number, line))
        elif in_recipe and lines and lines[-1][1].endswith("\\"):
            lines.append((number, line))
        else:
            in_recipe = False
    return lines


class CompileCommandsRefreshTest(unittest.TestCase):
    """Every build the repository's tooling runs refreshes the database afterwards."""

    def test_every_makefile_build_goes_through_the_runner(self) -> None:
        unwrapped = [
            f"Makefile:{number}: {line.strip()}"
            for number, line in _makefile_recipe_lines()
            if _BUILDING_COMMAND.search(line)
        ]
        self.assertEqual(
            unwrapped, [], "run these through $(bazel_indexed), see tools/ide/README.md"
        )

    def test_a_spelled_out_bazel_build_is_found(self) -> None:
        for line in (
            "\tbazel build //...",
            "\tcd x && bazel run //pkg:bin",
            "\t@$(dev_run) 'bazel test //x'",
        ):
            with self.subTest(line=line):
                self.assertIsNotNone(_BUILDING_COMMAND.search(line))
        for line in (
            "\t$(bazel_indexed) build //...",
            "\t$(bazel_indexed_in_repo) run //pkg:bin",
            "\tmybazel build",
        ):
            with self.subTest(line=line):
                self.assertIsNone(_BUILDING_COMMAND.search(line))

    def test_the_makefile_runners_are_the_generators_after_mode(self) -> None:
        makefile = _read("Makefile")
        self.assertIn(f"bazel_indexed_in_repo := {_RUNNER}\n", makefile)
        self.assertIn(
            "bazel_indexed := cd $(current_path) && $(bazel_indexed_in_repo)\n",
            makefile,
        )

    def test_make_compile_commands_builds_the_generated_headers(self) -> None:
        recipe = re.search(
            r"^compile-commands:\n((?:\t.*\n)+)", _read("Makefile"), re.MULTILINE
        )
        self.assertIsNotNone(recipe)
        assert recipe is not None
        self.assertIn(
            "python3 -m tools.ide.compile_commands --build_headers", recipe.group(1)
        )

    def test_the_launch_and_deployment_builds_go_through_the_runner(self) -> None:
        for path in ("tools/deploy/session.sh", "tools/deploy/deploy_robot.sh"):
            source = _read(path)
            with self.subTest(path=path):
                self.assertIn(_RUNNER + " build", source)
                for number, line in enumerate(source.splitlines(), start=1):
                    if line.lstrip().startswith("#"):
                        continue
                    for match in _BUILDING_COMMAND.finditer(line):
                        self.assertTrue(
                            line[: match.start()].endswith(_RUNNER[: -len("bazel")]),
                            f"{path}:{number}: {line.strip()}",
                        )

    def test_the_dev_container_refreshes_when_it_starts(self) -> None:
        compose = _read("docker-compose.yaml")
        self.assertRegex(
            compose,
            r"python3 -m tools\.ide\.compile_commands --build_headers --quiet[^;\n]*&",
        )


class RefreshSurroundingsTest(unittest.TestCase):
    """The refresh's query loads packages as the build does, and the local CI emulator leaves the database alone."""

    def test_the_autoload_flag_applies_to_query_too(self) -> None:
        # .bazelrc says why: a `bazel query` that loads packages otherwise than the build makes the next build-like
        # command fetch repositories again.
        commands = []
        for line in _read(".bazelrc").splitlines():
            words = line.split()
            if len(words) == 2 and words[1].startswith(
                "--incompatible_autoload_externally="
            ):
                commands.append(words[0])
        self.assertEqual(commands, ["common"])

    def test_make_ci_local_sets_ci_as_github_does(self) -> None:
        self.assertRegex(_read("tools/ci_local.sh"), r"\(-e CI=true\)")
        self.assertRegex(
            _read("tools/ci_local.sh"),
            r'docker run --rm [^\n]*"\$\{GITHUB_ENVIRONMENT\[@\]\}"',
        )


class PythonImportRootsTest(unittest.TestCase):
    """The language server resolves first-party imports against the directories Bazel puts on the path."""

    def test_pyright_extra_paths_are_the_bazel_import_roots(self) -> None:
        config = json.loads(_read("pyrightconfig.json"))
        self.assertEqual(config["extraPaths"], lint_files.python_import_roots(_root()))


if __name__ == "__main__":
    unittest.main()
