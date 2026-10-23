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

"""Tests the compilation database generator against canned `bazel aquery` output and a fake `bazel`."""

import contextlib
import dataclasses
import io
import json
import os
import signal
import stat
import sys
import tempfile
import textwrap
from typing import Any
import unittest
from unittest import mock

from tools.ide import compile_commands

_WORKSPACE = "/ws/checkout"
_OUTPUT_BASE = "/cache/output_base"
_EXECROOT = "/cache/output_base/execroot/_main"


def _arguments(source: str, *extra: str) -> list[str]:
    """Returns the command line of a CppCompile action as Bazel runs it."""
    return [
        "/usr/bin/gcc",
        "-MD",
        "-MF",
        "bazel-out/k8-opt/bin/pkg/_objs/lib/a.pic.d",
        "-frandom-seed=bazel-out/k8-opt/bin/pkg/_objs/lib/a.pic.o",
        "-iquote",
        ".",
        "-Ibazel-out/k8-opt/bin/pkg/include",
        "-isystem",
        "external/abseil-cpp+",
        *extra,
        "-c",
        source,
        "-o",
        "bazel-out/k8-opt/bin/pkg/_objs/lib/a.pic.o",
    ]


def _action(target_id: int, source: str, *extra: str) -> dict[str, Any]:
    """Returns a CppCompile action of the shape `bazel aquery --output=jsonproto` prints."""
    return {
        "targetId": target_id,
        "mnemonic": "CppCompile",
        "arguments": _arguments(source, *extra),
    }


class CompileArgumentsTest(unittest.TestCase):
    """The flags only Bazel's action uses are dropped, every other flag is kept in order."""

    def test_drops_the_dependency_object_and_seed_flags(self) -> None:
        arguments = compile_commands.compile_arguments(_arguments("pkg/a.cpp"))
        self.assertEqual(
            arguments,
            [
                "/usr/bin/gcc",
                "-iquote",
                ".",
                "-Ibazel-out/k8-opt/bin/pkg/include",
                "-isystem",
                "external/abseil-cpp+",
                "-c",
                "pkg/a.cpp",
            ],
        )

    def test_keeps_flags_that_only_start_like_a_dropped_one(self) -> None:
        self.assertEqual(
            compile_commands.compile_arguments(["gcc", "-MDX", "-output"]),
            ["gcc", "-MDX", "-output"],
        )


class SourceTest(unittest.TestCase):
    """The source is the value of `-c`, and moves to its path in the checkout."""

    def test_finds_the_source_after_dash_c(self) -> None:
        self.assertEqual(
            compile_commands.source_of(["gcc", "-c", "pkg/a.cpp", "-o", "a.o"]),
            "pkg/a.cpp",
        )

    def test_has_no_source_without_dash_c(self) -> None:
        self.assertIsNone(compile_commands.source_of(["gcc", "-E", "pkg/a.cpp"]))
        self.assertIsNone(compile_commands.source_of(["gcc", "-c"]))

    def test_first_party_sources_are_in_the_checkout(self) -> None:
        self.assertEqual(
            compile_commands.checkout_path(
                "pkg/a.cpp", _WORKSPACE, _EXECROOT, _OUTPUT_BASE
            ),
            "/ws/checkout/pkg/a.cpp",
        )

    def test_generated_sources_are_in_the_execution_root_and_external_ones_in_the_output_base(
        self,
    ) -> None:
        self.assertEqual(
            compile_commands.checkout_path(
                "bazel-out/k8-opt/bin/pkg/a.pb.cc", _WORKSPACE, _EXECROOT, _OUTPUT_BASE
            ),
            "/cache/output_base/execroot/_main/bazel-out/k8-opt/bin/pkg/a.pb.cc",
        )
        self.assertEqual(
            compile_commands.checkout_path(
                "external/zlib+/adler32.c", _WORKSPACE, _EXECROOT, _OUTPUT_BASE
            ),
            "/cache/output_base/external/zlib+/adler32.c",
        )
        self.assertEqual(
            compile_commands.checkout_path(
                "/abs/a.cpp", _WORKSPACE, _EXECROOT, _OUTPUT_BASE
            ),
            "/abs/a.cpp",
        )

    def test_with_source_replaces_only_the_value_of_dash_c(self) -> None:
        self.assertEqual(
            compile_commands.with_source(
                ["gcc", "-include", "pkg/a.cpp", "-c", "pkg/a.cpp"], "/ws/a.cpp"
            ),
            ["gcc", "-include", "pkg/a.cpp", "-c", "/ws/a.cpp"],
        )


class ExternalRepositoriesTest(unittest.TestCase):
    """Paths into the execution root's external/ move to the output base, in every spelling of an include flag."""

    def test_rewrites_the_paths_into_external(self) -> None:
        self.assertEqual(
            compile_commands.with_external_repositories(
                [
                    "gcc",
                    "-isystem",
                    "external/abseil-cpp+",
                    "-iquote",
                    "external/eigen",
                    "-Iexternal/zlib+/include",
                    "-isystemexternal/x",
                    "-Ibazel-out/k8-opt/bin/external/abseil-cpp+",
                    "-Ipkg/include",
                    "-DEXTERNAL=external/",
                ],
                _OUTPUT_BASE,
            ),
            [
                "gcc",
                "-isystem",
                "/cache/output_base/external/abseil-cpp+",
                "-iquote",
                "/cache/output_base/external/eigen",
                "-I/cache/output_base/external/zlib+/include",
                "-isystem/cache/output_base/external/x",
                "-Ibazel-out/k8-opt/bin/external/abseil-cpp+",
                "-Ipkg/include",
                "-DEXTERNAL=external/",
            ],
        )


class DatabaseEntriesTest(unittest.TestCase):
    """One entry per source, run from the execution root, deterministic in content and order."""

    def _aquery(self) -> dict[str, Any]:
        return {
            "targets": [
                {"id": 1, "label": "//pkg:z_test"},
                {"id": 2, "label": "//pkg:lib"},
            ],
            "actions": [
                _action(1, "pkg/b.cpp", "-DFROM_TEST"),
                _action(2, "pkg/b.cpp", "-DFROM_LIB"),
                _action(2, "pkg/a.cpp"),
                {
                    "targetId": 2,
                    "mnemonic": "CppCompile",
                    "arguments": ["gcc", "-E", "x.cpp"],
                },
            ],
        }

    def test_entries_are_sorted_by_file_and_run_from_the_execution_root(self) -> None:
        entries = compile_commands.database_entries(
            self._aquery(), _WORKSPACE, _EXECROOT, _OUTPUT_BASE
        )
        self.assertEqual(
            [entry["file"] for entry in entries],
            ["/ws/checkout/pkg/a.cpp", "/ws/checkout/pkg/b.cpp"],
        )
        for entry in entries:
            self.assertEqual(entry["directory"], _EXECROOT)
            self.assertEqual(entry["arguments"][-2:], ["-c", entry["file"]])

    def test_a_source_of_several_targets_keeps_the_first_label(self) -> None:
        entries = compile_commands.database_entries(
            self._aquery(), _WORKSPACE, _EXECROOT, _OUTPUT_BASE
        )
        self.assertIn("-DFROM_LIB", entries[1]["arguments"])
        self.assertNotIn("-DFROM_TEST", entries[1]["arguments"])

    def test_external_include_directories_move_to_the_output_base(self) -> None:
        # The execution root links only the external repositories of its last build; the output base has them all.
        entries = compile_commands.database_entries(
            self._aquery(), _WORKSPACE, _EXECROOT, _OUTPUT_BASE
        )
        self.assertIn("-Ibazel-out/k8-opt/bin/pkg/include", entries[0]["arguments"])
        self.assertIn(
            "/cache/output_base/external/abseil-cpp+", entries[0]["arguments"]
        )
        self.assertNotIn("external/abseil-cpp+", entries[0]["arguments"])

    def test_an_empty_query_gives_an_empty_database(self) -> None:
        self.assertEqual(
            compile_commands.database_entries({}, _WORKSPACE, _EXECROOT, _OUTPUT_BASE),
            [],
        )


_ENTRIES = [{"directory": "/d", "file": "/d/a.cpp", "arguments": ["gcc"]}]


class WriteDatabaseTest(unittest.TestCase):
    """The database is replaced at once, only when its content changes, with no temporary file left behind."""

    def test_writes_json_and_leaves_no_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, compile_commands.OUTPUT_NAME)
            with open(path, "w", encoding="utf-8") as f:
                f.write("[]\n")
            self.assertTrue(compile_commands.write_database(_ENTRIES, path))
            with open(path, encoding="utf-8") as f:
                self.assertEqual(json.load(f), _ENTRIES)
            self.assertEqual(os.listdir(directory), [compile_commands.OUTPUT_NAME])

    def test_leaves_a_file_with_the_same_content_alone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, compile_commands.OUTPUT_NAME)
            self.assertTrue(compile_commands.write_database(_ENTRIES, path))
            before = os.stat(path)
            self.assertFalse(compile_commands.write_database(_ENTRIES, path))
            after = os.stat(path)
            self.assertEqual(
                (before.st_ino, before.st_mtime_ns), (after.st_ino, after.st_mtime_ns)
            )
            self.assertTrue(compile_commands.write_database([], path))

    def test_a_failed_replace_keeps_the_old_file_and_removes_the_temporary_one(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, compile_commands.OUTPUT_NAME)
            with open(path, "w", encoding="utf-8") as f:
                f.write("old\n")
            with mock.patch.object(
                os, "replace", side_effect=OSError("read-only")
            ), self.assertRaises(OSError):
                compile_commands.write_database(_ENTRIES, path)
            self.assertEqual(os.listdir(directory), [compile_commands.OUTPUT_NAME])
            with open(path, encoding="utf-8") as f:
                self.assertEqual(f.read(), "old\n")


class MergedEntriesTest(unittest.TestCase):
    """A partial result keeps the previous database's entries for the files it lacks, and replaces the others."""

    def test_keeps_the_files_the_result_lacks_and_takes_the_results_entries(
        self,
    ) -> None:
        previous = [
            {"directory": "/old", "file": "/d/a.cpp", "arguments": ["old"]},
            {"directory": "/old", "file": "/d/c.cpp", "arguments": ["old"]},
        ]
        entries = [
            {"directory": "/new", "file": "/d/b.cpp", "arguments": ["new"]},
            {"directory": "/new", "file": "/d/a.cpp", "arguments": ["new"]},
        ]
        self.assertEqual(
            compile_commands.merged_entries(entries, previous),
            [
                {"directory": "/new", "file": "/d/a.cpp", "arguments": ["new"]},
                {"directory": "/new", "file": "/d/b.cpp", "arguments": ["new"]},
                {"directory": "/old", "file": "/d/c.cpp", "arguments": ["old"]},
            ],
        )

    def test_reads_only_a_compilation_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, compile_commands.OUTPUT_NAME)
            self.assertEqual(compile_commands.read_database(path), [])
            for content in ("old\n", "{}", '[1, {"directory": "/d"}]'):
                with self.subTest(content=content):
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(content)
                    self.assertEqual(compile_commands.read_database(path), [])
            with open(path, "w", encoding="utf-8") as f:
                json.dump(_ENTRIES, f)
            self.assertEqual(compile_commands.read_database(path), _ENTRIES)


class SkipReasonTest(unittest.TestCase):
    """The database is written in the dev container only: not in CI, not where the checkout is outside its mount."""

    _IN_THE_DEV_CONTAINER = "/wb_humanoid_mpc_ws/workspace/wb_humanoid_mpc"

    def test_the_dev_container_writes_it(self) -> None:
        self.assertIsNone(compile_commands.skip_reason({}, self._IN_THE_DEV_CONTAINER))
        self.assertIsNone(
            compile_commands.skip_reason({"CI": "false"}, self._IN_THE_DEV_CONTAINER)
        )

    def test_ci_does_not(self) -> None:
        self.assertEqual(
            compile_commands.skip_reason({"CI": "true"}, self._IN_THE_DEV_CONTAINER),
            "CI is set",
        )

    def test_a_container_that_opts_out_does_not(self) -> None:
        self.assertEqual(
            compile_commands.skip_reason(
                {"WB_COMPILE_COMMANDS": "off"}, self._IN_THE_DEV_CONTAINER
            ),
            "WB_COMPILE_COMMANDS=off",
        )
        for value in ("", "on", "OFF"):
            with self.subTest(value=value):
                self.assertIsNone(
                    compile_commands.skip_reason(
                        {"WB_COMPILE_COMMANDS": value}, self._IN_THE_DEV_CONTAINER
                    )
                )

    def test_main_says_an_opted_out_container_leaves_it_alone(self) -> None:
        stderr = io.StringIO()
        with mock.patch.dict(
            os.environ, {"WB_COMPILE_COMMANDS": "off"}
        ), mock.patch.object(
            compile_commands, "_bazel", side_effect=AssertionError("ran bazel")
        ), contextlib.redirect_stderr(
            stderr
        ):
            self.assertEqual(compile_commands.main(["--quiet"]), 1)
        self.assertEqual(
            stderr.getvalue(),
            "compile_commands: WB_COMPILE_COMMANDS=off: compile_commands.json is left alone\n",
        )

    def test_a_checkout_outside_the_mount_does_not(self) -> None:
        for root in (
            "/workspace",
            "/home/me/workspace/wb_humanoid_mpc",
            "/wb_humanoid_mpc_ws_other/x",
        ):
            with self.subTest(root=root):
                self.assertEqual(
                    compile_commands.skip_reason({}, root),
                    "the checkout is not below /wb_humanoid_mpc_ws",
                )

    def test_main_says_so_in_one_line_and_runs_no_bazel(self) -> None:
        stderr = io.StringIO()
        with mock.patch.object(
            compile_commands, "_skip_reason", return_value="CI is set"
        ), mock.patch.object(
            compile_commands, "_bazel", side_effect=AssertionError("ran bazel")
        ), contextlib.redirect_stderr(
            stderr
        ):
            self.assertEqual(compile_commands.main(["--quiet"]), 1)
        self.assertEqual(len(stderr.getvalue().splitlines()), 1)
        self.assertIn("not the dev container (CI is set)", stderr.getvalue())


# Logs its arguments and the files its --query_file and --target_pattern_file name; answers `query` ($FAKE_BAZEL_QUERY
# when it is set), `info` and `aquery` (the file $FAKE_BAZEL_AQUERY), and exits with $FAKE_BAZEL_STATUS_<command>, 0 by
# default.
_FAKE_BAZEL = textwrap.dedent(
    """\
    #!/bin/sh
    echo "$*" >> "${FAKE_BAZEL_LOG}"
    for argument in "$@"; do
        case "${argument}" in
            --query_file=*) cat "${argument#--query_file=}" > "${FAKE_BAZEL_LOG}.query" ;;
            --target_pattern_file=*) cat "${argument#--target_pattern_file=}" > "${FAKE_BAZEL_LOG}.targets" ;;
        esac
    done
    case "$1" in
        query) printf '%b' "${FAKE_BAZEL_QUERY-//pkg:lib\\n//pkg:a+b\\n}" ;;
        info) printf '%s\\n' "${FAKE_BAZEL_INFO:-execution_root: /cache/output_base/execroot/_main}" \\
            "output_base: /cache/output_base" ;;
        aquery) cat "${FAKE_BAZEL_AQUERY}" ;;
    esac
    eval "exit \\${FAKE_BAZEL_STATUS_$1:-0}"
    """
)


@dataclasses.dataclass
class _FakeBazelRun:
    """What a run of main() with the fake bazel left: its status, the Bazel calls, their files, the database, its output."""

    status: int
    calls: list[str]
    files: dict[str, str]
    written: str
    stderr: str
    stdout: str

    def call(self, command: str) -> list[str]:
        """Returns the arguments of the one Bazel call of `command`."""
        matching = [call.split() for call in self.calls if call.split()[0] == command]
        assert len(matching) == 1, self.calls
        return matching[0]


def _run_main(
    argv: list[str],
    aquery_output: str | None = None,
    statuses: dict[str, int] | None = None,
    info_output: str | None = None,
    query_output: str | None = None,
    old_database: str = "old\n",
) -> _FakeBazelRun:
    """Runs main() in the dev container with the fake bazel, which exits with the status `statuses` gives a command.

    Args:
      argv: main()'s arguments but --output, which names a database that holds `old_database` before the run.
      aquery_output: What the fake aquery prints; one action of //pkg:lib compiling pkg/a.cpp when None.
      statuses: The exit status of each Bazel command by name, 0 for the others.
      info_output: The execution_root line the fake info prints, when not None.
      query_output: What the fake query prints, when not None; two labels otherwise.
      old_database: The database's content before the run.

    Returns:
      What the run left.
    """
    with tempfile.TemporaryDirectory() as directory:
        bazel = os.path.join(directory, "bazel")
        with open(bazel, "w", encoding="utf-8") as f:
            f.write(_FAKE_BAZEL)
        os.chmod(bazel, os.stat(bazel).st_mode | stat.S_IXUSR)
        aquery = os.path.join(directory, "aquery.json")
        with open(aquery, "w", encoding="utf-8") as f:
            if aquery_output is None:
                json.dump(
                    {
                        "targets": [{"id": 1, "label": "//pkg:lib"}],
                        "actions": [_action(1, "pkg/a.cpp")],
                    },
                    f,
                )
            else:
                f.write(aquery_output)
        output = os.path.join(directory, "out.json")
        with open(output, "w", encoding="utf-8") as f:
            f.write(old_database)
        log = os.path.join(directory, "log")
        environment = {
            "PATH": f"{directory}{os.pathsep}{os.environ.get('PATH', '')}",
            "FAKE_BAZEL_LOG": log,
            "FAKE_BAZEL_AQUERY": aquery,
        }
        if info_output is not None:
            environment["FAKE_BAZEL_INFO"] = info_output
        if query_output is not None:
            environment["FAKE_BAZEL_QUERY"] = query_output
        for command, status in (statuses or {}).items():
            environment[f"FAKE_BAZEL_STATUS_{command}"] = str(status)
        stderr = io.StringIO()
        stdout = io.StringIO()
        with mock.patch.dict(os.environ, environment), mock.patch.object(
            compile_commands, "_skip_reason", return_value=None
        ), contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(stdout):
            status = compile_commands.main(["--output", output, *argv])
        with open(log, encoding="utf-8") as f:
            calls = f.read().splitlines()
        files = {}
        for kind in ("query", "targets"):
            if os.path.exists(f"{log}.{kind}"):
                with open(f"{log}.{kind}", encoding="utf-8") as f:
                    files[kind] = f.read()
        with open(output, encoding="utf-8") as f:
            written = f.read()
        return _FakeBazelRun(
            status, calls, files, written, stderr.getvalue(), stdout.getvalue()
        )


class MainTest(unittest.TestCase):
    """main() analyzes the C++ targets with the build's flags and leaves the database alone when Bazel fails."""

    def test_queries_the_cpp_targets_with_the_builds_flags(self) -> None:
        run = _run_main(["--quiet", "--", "--config=dbg"])
        self.assertEqual(run.status, 0, run.stderr)
        self.assertEqual(
            [call.split()[0] for call in run.calls], ["query", "info", "aquery"]
        )
        query = run.call("query")
        self.assertNotIn("--config=dbg", query)
        self.assertIn('kind("cc_.* rule", //...)', run.calls[0])
        self.assertIn("--config=dbg", run.call("info"))
        self.assertEqual(run.call("info")[-2:], ["execution_root", "output_base"])
        self.assertIn("--config=dbg", run.call("aquery"))
        self.assertEqual(
            run.files["query"], 'mnemonic("CppCompile", set("//pkg:a+b" "//pkg:lib"))'
        )
        self.assertNotIn("//...", run.calls[2])
        self.assertEqual(
            json.loads(run.written)[0]["file"],
            os.path.join(compile_commands._repository_root(), "pkg/a.cpp"),
        )

    def test_build_headers_builds_the_compilation_prerequisites_of_the_same_targets(
        self,
    ) -> None:
        run = _run_main(["--build_headers"])
        self.assertEqual(run.status, 0, run.stderr)
        self.assertEqual(
            [call.split()[0] for call in run.calls],
            ["query", "build", "info", "aquery"],
        )
        build = run.call("build")
        self.assertIn("--output_groups=compilation_prerequisites_INTERNAL_", build)
        self.assertIn("--keep_going", build)
        self.assertNotIn("//...", build)
        self.assertEqual(run.files["targets"], "//pkg:a+b\n//pkg:lib\n")

    def test_a_failed_header_build_warns_and_the_query_still_runs(self) -> None:
        run = _run_main(["--build_headers"], statuses={"build": 1})
        self.assertEqual(run.status, 0, run.stderr)
        self.assertIn("warning: the generated headers of some targets", run.stderr)
        self.assertEqual(len(json.loads(run.written)), 1)

    def test_a_partial_aquery_is_written_with_a_warning(self) -> None:
        run = _run_main([], statuses={"aquery": 1})
        self.assertEqual(run.status, 0, run.stderr)
        self.assertIn("warning: some targets failed to analyze", run.stderr)
        self.assertEqual(len(json.loads(run.written)), 1)

    def test_a_partial_target_query_is_used_with_a_warning(self) -> None:
        run = _run_main([], statuses={"query": 3})
        self.assertEqual(run.status, 0, run.stderr)
        self.assertIn("warning: some packages failed to load", run.stderr)

    def test_a_partial_aquery_keeps_the_entries_of_the_sources_it_lacks(self) -> None:
        # The old database has a.cpp and b.cpp; the aquery that some targets failed in compiles a.cpp alone.
        root = compile_commands._repository_root()
        old = [
            {"directory": "/old", "file": f"{root}/pkg/a.cpp", "arguments": ["gcc"]},
            {"directory": "/old", "file": f"{root}/pkg/b.cpp", "arguments": ["gcc"]},
        ]
        for status, expected_directories in (
            (1, [_EXECROOT, "/old"]),
            (0, [_EXECROOT]),
        ):
            with self.subTest(status=status):
                run = _run_main(
                    [], statuses={"aquery": status}, old_database=json.dumps(old)
                )
                self.assertEqual(run.status, 0, run.stderr)
                written = json.loads(run.written)
                self.assertEqual(
                    [entry["directory"] for entry in written], expected_directories
                )
                self.assertEqual(written[0]["file"], f"{root}/pkg/a.cpp")
        partial = _run_main([], statuses={"aquery": 1}, old_database=json.dumps(old))
        self.assertIn(
            "2 source files in", partial.stderr.splitlines()[-1], partial.stderr
        )
        self.assertIn("1 of them kept from the previous database", partial.stderr)

    def test_a_query_that_lists_no_target_leaves_the_database_unchanged(
        self,
    ) -> None:
        for status in (0, 3):
            with self.subTest(status=status):
                run = _run_main([], statuses={"query": status}, query_output="")
                self.assertEqual(run.status, 1)
                self.assertEqual(run.written, "old\n")
                self.assertIn("bazel query listed no C++ target", run.stderr)
                self.assertEqual(
                    [call.split()[0] for call in run.calls], ["query"], run.calls
                )

    def test_every_line_goes_to_stderr_and_a_quiet_refresh_announces_itself(
        self,
    ) -> None:
        quiet = _run_main(["--quiet"])
        self.assertEqual(quiet.status, 0, quiet.stderr)
        self.assertEqual(quiet.stdout, "")
        lines = quiet.stderr.splitlines()
        self.assertEqual(lines[0], "compile_commands: refreshing compile_commands.json")
        self.assertRegex(
            lines[-1], r"^compile_commands: 1 source files in .+out\.json$"
        )
        loud = _run_main([])
        self.assertEqual(loud.stdout, "")
        self.assertNotIn("refreshing", loud.stderr)

    def test_a_fatal_bazel_status_leaves_the_database_unchanged(self) -> None:
        for command in ("build", "aquery"):
            for status in (2, 8, 36, 37):
                with self.subTest(command=command, status=status):
                    run = _run_main(["--build_headers"], statuses={command: status})
                    self.assertEqual(run.status, 1)
                    self.assertEqual(run.written, "old\n")

    def test_a_failed_query_or_info_leaves_the_database_unchanged(self) -> None:
        for command in ("query", "info"):
            with self.subTest(command=command):
                run = _run_main([], statuses={command: 7})
                self.assertEqual(run.status, 1)
                self.assertEqual(run.written, "old\n")

    def test_a_failed_aquery_without_actions_leaves_the_database_unchanged(
        self,
    ) -> None:
        for output in ("", "{}", "not json", "[1]"):
            with self.subTest(output=output):
                run = _run_main([], aquery_output=output, statuses={"aquery": 1})
                self.assertEqual(run.status, 1)
                self.assertEqual(run.written, "old\n")

    def test_unparsable_info_output_leaves_the_database_unchanged(self) -> None:
        run = _run_main([], info_output="Starting local Bazel server")
        self.assertEqual(run.status, 1)
        self.assertEqual(run.written, "old\n")

    def test_unparsable_aquery_output_leaves_the_database_unchanged(self) -> None:
        run = _run_main([], aquery_output="not json")
        self.assertEqual(run.status, 1)
        self.assertEqual(run.written, "old\n")

    def test_a_failed_write_leaves_the_database_unchanged(self) -> None:
        with mock.patch.object(
            compile_commands, "write_database", side_effect=OSError("disk full")
        ):
            run = _run_main([])
        self.assertEqual(run.status, 1)
        self.assertIn("disk full", run.stderr)
        self.assertEqual(run.written, "old\n")


class ConfigurationFlagsTest(unittest.TestCase):
    """The refresh reads the configuration the command built in, and nothing else of its command line."""

    def test_splits_the_command_and_the_arguments_bazel_parses(self) -> None:
        self.assertEqual(
            compile_commands.command_parts(
                ["bazel", "run", "//pkg:bin", "--config=dbg", "--", "--flag"]
            ),
            ("run", ["//pkg:bin", "--config=dbg"]),
        )
        self.assertEqual(compile_commands.command_parts(["bazel"]), (None, []))

    def test_keeps_the_configuration_flags_in_both_spellings(self) -> None:
        self.assertEqual(
            compile_commands.configuration_flags(
                [
                    "//...",
                    "--config",
                    "dbg",
                    "-c",
                    "opt",
                    "--platforms=//p:arm64",
                    "--test_output=all",
                    "--keep_going",
                ]
            ),
            ["--config=dbg", "--compilation_mode=opt", "--platforms=//p:arm64"],
        )

    def test_every_flag_is_a_long_option_with_its_value(self) -> None:
        arguments = []
        for flag in sorted(compile_commands._CONFIGURATION_FLAGS):
            arguments += [flag, "value"]
        flags = compile_commands.configuration_flags(arguments)
        self.assertEqual(len(flags), len(compile_commands._CONFIGURATION_FLAGS))
        for flag in flags:
            self.assertRegex(flag, r"^--[a-z_]+=value$")

    def test_leaves_out_the_aspect_only_configs(self) -> None:
        self.assertEqual(
            compile_commands.configuration_flags(
                ["--config=clang-tidy", "--config", "clang-tidy-fix", "--config=dbg"]
            ),
            ["--config=dbg"],
        )

    def test_a_flag_without_its_value_is_ignored(self) -> None:
        self.assertEqual(
            compile_commands.configuration_flags(["//...", "--config"]), []
        )


class _ExecError(Exception):
    """Raised by the fake os.execvp in place of replacing the test's process."""


@dataclasses.dataclass
class _RunAfter:
    """What run_after() did with fake commands: its status (None after an exec), the runs, refreshes, execs, stderr."""

    status: int | None = None
    runs: list[list[str]] = dataclasses.field(default_factory=list)
    refreshes: list[list[str]] = dataclasses.field(default_factory=list)
    execs: list[list[str]] = dataclasses.field(default_factory=list)
    stderr: str = ""


def _run_after(
    command: list[str],
    statuses: list[int],
    refresh_error: type[BaseException] | None = None,
    skip_reason: str | None = None,
    write_script: bool = True,
) -> _RunAfter:
    """Runs run_after() with fake commands, which exit with `statuses` in turn and write the script --script_path names.

    Args:
      command: The Bazel command line.
      statuses: The exit status of each command run, in order.
      refresh_error: What the refresh raises, or None for a refresh that passes.
      skip_reason: Why this is not the dev container, or None in it.
      write_script: Whether a command writes the script its --script_path names.

    Returns:
      What run_after() did.
    """
    result = _RunAfter()

    def fake_run(arguments: list[str]) -> int:
        result.runs.append(list(arguments))
        for argument in arguments:
            if write_script and argument.startswith("--script_path="):
                with open(argument.split("=", 1)[1], "w", encoding="utf-8") as f:
                    f.write("exec program 'a b' \"$@\"\n")
        return statuses[len(result.runs) - 1]

    def fake_refresh(flags: list[str], quiet: bool) -> int:
        del quiet  # Unused.
        result.refreshes.append(list(flags))
        if refresh_error is not None:
            raise refresh_error()
        return 0

    def fake_exec(program: str, arguments: list[str]) -> None:
        del program  # Unused.
        result.execs.append(list(arguments))
        raise _ExecError()

    stderr = io.StringIO()
    with mock.patch.object(compile_commands, "_run", fake_run), mock.patch.object(
        compile_commands, "refresh", fake_refresh
    ), mock.patch.object(os, "execvp", fake_exec), mock.patch.object(
        compile_commands, "_skip_reason", return_value=skip_reason
    ), contextlib.redirect_stderr(
        stderr
    ):
        try:
            result.status = compile_commands.run_after(command)
        except _ExecError:
            result.status = None
    result.stderr = stderr.getvalue()
    return result


class RunAfterTest(unittest.TestCase):
    """--after keeps the command's exit status and refreshes after every build that ran."""

    def test_a_build_refreshes_with_its_configuration_and_keeps_its_status(
        self,
    ) -> None:
        result = _run_after(
            ["bazel", "build", "//...", "--config=dbg", "-c", "dbg"], [1]
        )
        self.assertEqual(result.status, 1)
        self.assertEqual(
            result.runs, [["bazel", "build", "//...", "--config=dbg", "-c", "dbg"]]
        )
        self.assertEqual(result.refreshes, [["--config=dbg", "--compilation_mode=dbg"]])

    def test_a_test_refreshes_after_failing_tests_too(self) -> None:
        result = _run_after(["bazel", "test", "//pkg:test", "--test_output=all"], [3])
        self.assertEqual(result.status, 3)
        self.assertEqual(result.refreshes, [[]])

    def test_a_command_that_built_nothing_does_not_refresh(self) -> None:
        # Interrupted (Ctrl-C), a command line error, not started, ended by SIGINT and by SIGKILL.
        for status in (8, 2, 127, 128 + signal.SIGINT, 128 + signal.SIGKILL):
            with self.subTest(status=status):
                result = _run_after(["bazel", "build", "//..."], [status])
                self.assertEqual(result.status, status)
                self.assertEqual(result.refreshes, [])

    def test_ctrl_c_during_the_refresh_keeps_the_commands_status(self) -> None:
        result = _run_after(
            ["bazel", "build", "//..."], [0], refresh_error=KeyboardInterrupt
        )
        self.assertEqual(result.status, 0)
        self.assertIn("was not refreshed: interrupted", result.stderr)

    def test_a_run_writes_a_script_refreshes_and_then_execs_the_script(self) -> None:
        command = ["bazel", "run", "//pkg:bin", "--config=dbg", "--", "--flag"]
        result = _run_after(command, [0])
        self.assertIsNone(result.status)
        self.assertEqual(len(result.runs), 1)
        run = result.runs[0]
        self.assertEqual(run[:2], ["bazel", "run"])
        self.assertRegex(run[2], r"^--script_path=/.+/run\.sh$")
        self.assertEqual(run[3:], command[2:])
        self.assertEqual(result.refreshes, [["--config=dbg"]])
        script_path = run[2].split("=", 1)[1]
        self.assertEqual(
            result.execs,
            [["bash", "-c", "exec program 'a b' \"$@\"\n", script_path]],
        )
        self.assertFalse(os.path.exists(os.path.dirname(script_path)))

    def test_a_run_passes_its_program_arguments_without_a_separator(self) -> None:
        result = _run_after(["bazel", "run", "//pkg:bin", "list", "--config=dbg"], [0])
        self.assertEqual(result.runs[0][3:], ["//pkg:bin", "list", "--config=dbg"])
        self.assertEqual(result.refreshes, [["--config=dbg"]])

    def test_a_run_whose_build_fails_does_not_start_the_program(self) -> None:
        result = _run_after(["bazel", "run", "//pkg:bin"], [1])
        self.assertEqual(result.status, 1)
        self.assertEqual(result.refreshes, [[]])
        self.assertEqual(result.execs, [])

    def test_a_run_whose_build_wrote_no_script_fails(self) -> None:
        result = _run_after(["bazel", "run", "//pkg:bin"], [0], write_script=False)
        self.assertEqual(result.status, 1)
        self.assertEqual(result.refreshes, [[]])
        self.assertEqual(result.execs, [])
        self.assertIn("bazel run wrote no script", result.stderr)

    def test_ctrl_c_during_the_refresh_of_a_run_does_not_start_the_program(
        self,
    ) -> None:
        result = _run_after(
            ["bazel", "run", "//pkg:bin"], [0], refresh_error=KeyboardInterrupt
        )
        self.assertEqual(result.status, 130)
        self.assertEqual(result.execs, [])

    def test_a_run_that_writes_a_script_itself_is_a_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = [
                "bazel",
                "run",
                f"--script_path={os.path.join(directory, 'x.sh')}",
                "//pkg:bin",
            ]
            result = _run_after(command, [0])
        self.assertEqual(result.status, 0)
        self.assertEqual(result.runs, [command])
        self.assertEqual(result.refreshes, [[]])
        self.assertEqual(result.execs, [])

    def test_other_commands_run_unchanged_without_a_refresh(self) -> None:
        for command in (["bazel", "query", "//..."], ["bazel", "coverage", "//..."]):
            with self.subTest(command=command):
                result = _run_after(command, [])
                self.assertIsNone(result.status)
                self.assertEqual(result.execs, [command])
                self.assertEqual(result.refreshes, [])

    def test_an_opted_out_container_runs_the_command_alone(self) -> None:
        result = _run_after(
            ["bazel", "build", "//..."], [], skip_reason="WB_COMPILE_COMMANDS=off"
        )
        self.assertIsNone(result.status)
        self.assertEqual(result.execs, [["bazel", "build", "//..."]])
        self.assertEqual(result.refreshes, [])
        self.assertEqual(
            result.stderr,
            "compile_commands: WB_COMPILE_COMMANDS=off: compile_commands.json is left alone\n",
        )

    def test_outside_the_dev_container_the_command_runs_alone(self) -> None:
        for command in (["bazel", "build", "//..."], ["bazel", "run", "//pkg:bin"]):
            with self.subTest(command=command):
                result = _run_after(command, [], skip_reason="CI is set")
                self.assertIsNone(result.status)
                self.assertEqual(result.execs, [command])
                self.assertEqual(result.refreshes, [])
                self.assertEqual(len(result.stderr.splitlines()), 1)
                self.assertIn("not the dev container (CI is set)", result.stderr)


class _RecordingStream(io.StringIO):
    """A stream that records its flushes in a shared list of events."""

    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    def flush(self) -> None:
        self.events.append("flush")
        super().flush()


class ExecTest(unittest.TestCase):
    """_exec() writes out this process's output before the program replaces it, and reports one that cannot start."""

    def test_flushes_stdout_and_stderr_before_the_exec(self) -> None:
        events: list[str] = []

        def fake_execvp(program: str, arguments: list[str]) -> None:
            del program, arguments  # Unused.
            events.append("exec")
            raise _ExecError()

        with mock.patch.object(
            sys, "stdout", _RecordingStream(events)
        ), mock.patch.object(
            sys, "stderr", _RecordingStream(events)
        ), mock.patch.object(
            os, "execvp", fake_execvp
        ), self.assertRaises(
            _ExecError
        ):
            compile_commands._exec(["program"])
        self.assertEqual(events, ["flush", "flush", "exec"])

    def test_a_program_that_cannot_start_gives_127(self) -> None:
        stderr = io.StringIO()
        with mock.patch.object(
            os, "execvp", side_effect=OSError("No such file or directory")
        ), contextlib.redirect_stderr(stderr):
            self.assertEqual(compile_commands._exec(["/nonexistent/program"]), 127)
        self.assertIn("cannot run /nonexistent/program", stderr.getvalue())


class RunTest(unittest.TestCase):
    """_run() reports a command a signal ended as the shell does, and one that could not start as 127."""

    def test_a_signal_gives_128_plus_its_number(self) -> None:
        status = compile_commands._run(
            [
                sys.executable,
                "-c",
                "import os, signal; os.kill(os.getpid(), signal.SIGTERM)",
            ]
        )
        self.assertEqual(status, 128 + signal.SIGTERM)

    def test_the_exit_status_is_kept(self) -> None:
        self.assertEqual(
            compile_commands._run([sys.executable, "-c", "raise SystemExit(3)"]), 3
        )

    def test_a_missing_program_gives_127(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(compile_commands._run(["/nonexistent/bazel"]), 127)


class CompileActionsQueryTest(unittest.TestCase):
    """The aquery expression quotes each label, since a label may hold a character the query language reserves (`+`)."""

    def test_quotes_every_label(self) -> None:
        self.assertEqual(
            compile_commands.compile_actions_query(["//a:b", "//c:d+e"]),
            'mnemonic("CppCompile", set("//a:b" "//c:d+e"))',
        )


if __name__ == "__main__":
    unittest.main()
