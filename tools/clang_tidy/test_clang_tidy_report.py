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

"""The clang-tidy collector, fix applier and self-test, on synthetic reports, fix-its and build event files."""

import contextlib
import io
import json
import os
import tempfile
import unittest

from tools.clang_tidy import clang_tidy_apply
from tools.clang_tidy import clang_tidy_report
from tools.clang_tidy import clang_tidy_selftest

SANDBOX = (
    "/home/u/.cache/bazel/_bazel_u/1/sandbox/processwrapper-sandbox/9/execroot/_main/"
)

HEADER_FINDING = """\
tools/x/include/x/Limits.h:5:22: error: invalid case style for global constant 'MAX_SIZE' \
[readability-identifier-naming,-warnings-as-errors]
    5 | inline constexpr int MAX_SIZE = 8;
      |                      ^~~~~~~~
      |                      kMaxSize
"""


def _write(directory: str, name: str, content: str) -> str:
    path = os.path.join(directory, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def _bep(directory: str, groups: dict[str, list[str]]) -> str:
    """A build event file in which one target completed with the given output groups."""
    events = []
    group_entries = []
    for index, (group, files) in enumerate(sorted(groups.items())):
        set_id = str(index)
        events.append(
            {
                "id": {"namedSet": {"id": set_id}},
                "namedSetOfFiles": {
                    "files": [
                        {"name": os.path.basename(f), "uri": "file://" + f}
                        for f in files
                    ]
                },
            }
        )
        group_entries.append({"name": group, "fileSets": [{"id": set_id}]})
    events.append(
        {
            "id": {
                "targetCompleted": {
                    "label": "//tools/x:x",
                    "aspect": "clang_tidy_aspect",
                }
            },
            "completed": {"success": True, "outputGroup": group_entries},
        }
    )
    return _write(
        directory, "bep.json", "\n".join(json.dumps(event) for event in events) + "\n"
    )


class ReportTest(unittest.TestCase):
    def test_a_finding_keeps_its_notes_and_source_lines(self):
        findings = clang_tidy_report.parse_report(HEADER_FINDING)
        self.assertEqual(len(findings), 1)
        finding = findings[0]
        self.assertEqual(
            finding.key,
            ("tools/x/include/x/Limits.h", 5, 22, "readability-identifier-naming"),
        )
        self.assertEqual(len(finding.details), 3)
        self.assertIn("[readability-identifier-naming]", finding.format())

    def test_a_header_finding_reported_by_several_actions_is_printed_once(self):
        # The header's own action names it with the sandbox's absolute path, each .cpp action with a relative one.
        parsed = []
        for report in (HEADER_FINDING, SANDBOX + HEADER_FINDING, "./" + HEADER_FINDING):
            parsed += clang_tidy_report.parse_report(report)
        findings = clang_tidy_report.deduplicate(parsed)
        self.assertEqual([f.path for f in findings], ["tools/x/include/x/Limits.h"])

    def test_a_compile_error_without_a_check_name_is_a_finding(self):
        findings = clang_tidy_report.parse_report(
            "tools/x/A.cpp:3:1: error: unknown type name 'Foo'\n"
        )
        self.assertEqual([f.check for f in findings], ["clang-diagnostic-error"])
        findings = clang_tidy_report.parse_report(
            "error: unable to handle compilation [clang-diagnostic-error]\n"
        )
        self.assertEqual(
            [(f.path, f.check) for f in findings], [("", "clang-diagnostic-error")]
        )

    def test_paths_keep_the_findings_of_the_given_directories(self):
        self.assertTrue(clang_tidy_report.in_paths("tools/x/A.cpp", []))
        self.assertTrue(clang_tidy_report.in_paths("tools/x/A.cpp", ["tools/x/"]))
        self.assertTrue(clang_tidy_report.in_paths("tools/x/A.cpp", ["tools/x"]))
        self.assertFalse(clang_tidy_report.in_paths("tools/xy/A.cpp", ["tools/x"]))

    def test_the_reports_of_a_build_come_from_its_build_event_file(self):
        with tempfile.TemporaryDirectory() as directory:
            first = _write(directory, "out/a.txt", HEADER_FINDING)
            second = _write(
                directory,
                "out/b.txt",
                "tools/y/B.cpp:1:2: error: m [google-runtime-int]\n",
            )
            stale = _write(
                directory,
                "out/stale.txt",
                "tools/z/C.cpp:1:2: error: m [google-runtime-int]\n",
            )
            fixes = _write(directory, "out/a.yaml", "")
            bep = _bep(
                directory, {"clang_tidy": [first, second], "clang_tidy_fixes": [fixes]}
            )
            files = clang_tidy_report.output_group_files(bep, "clang_tidy")
            self.assertEqual(files, sorted([first, second]))
            self.assertNotIn(stale, files)
            findings, missing = clang_tidy_report.collect(
                files + [os.path.join(directory, "gone.txt")], ["tools/y"]
            )
            self.assertEqual([f.path for f in findings], ["tools/y/B.cpp"])
            self.assertEqual(len(missing), 1)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(clang_tidy_report.main([bep, "--root", directory]), 1)
            self.assertIn("2 findings in 2 files", output.getvalue())

    def test_no_findings_is_success(self):
        with tempfile.TemporaryDirectory() as directory:
            bep = _bep(directory, {"clang_tidy": [_write(directory, "empty.txt", "")]})
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(clang_tidy_report.main([bep]), 0)

    def test_the_summary_counts_per_check_and_package(self):
        with tempfile.TemporaryDirectory() as directory:
            _write(directory, "tools/x/BUILD.bazel", "")
            findings = clang_tidy_report.parse_report(HEADER_FINDING)
            lines = clang_tidy_report.summary(findings, directory)
            self.assertIn("       1  readability-identifier-naming", lines)
            self.assertIn("       1  //tools/x", lines)


def _fixes(main_file: str, diagnostics: list[dict]) -> str:
    """An --export-fixes file as clang-tidy writes it."""
    entries = []
    for diagnostic in diagnostics:
        replacements = "".join(
            f"""
        - FilePath:        '{path}'
          Offset:          {offset}
          Length:          {length}
          ReplacementText: '{text}'"""
            for path, offset, length, text in diagnostic["replacements"]
        )
        entries.append(
            f"""
  - DiagnosticName:  {diagnostic["check"]}
    DiagnosticMessage:
      Message:         '{diagnostic["message"]}'
      FilePath:        '{diagnostic["path"]}'
      FileOffset:      {diagnostic["offset"]}
      Replacements:{replacements if replacements else " []"}
    Level:           Error"""
        )
    return f"---\nMainSourceFile:  '{SANDBOX}{main_file}'\nDiagnostics:{''.join(entries)}\n...\n"


RENAME = {
    "check": "readability-identifier-naming",
    "message": "invalid case style for MAX_SIZE",
    "offset": 0,
}


class ApplyTest(unittest.TestCase):
    def setUp(self):
        # pylint: disable-next=consider-using-with  # tearDown() deletes it.
        self.directory = tempfile.TemporaryDirectory()
        self.root = self.directory.name
        _write(self.root, "tools/x/Limits.h", "MAX_SIZE;\n")
        _write(self.root, "tools/x/A.cpp", "int a = MAX_SIZE;\n")
        _write(self.root, "tools/x/B.cpp", "int b = MAX_SIZE;\n")
        _write(self.root, "tools/y/C.cpp", "int c = MAX_SIZE;\n")

    def tearDown(self):
        self.directory.cleanup()

    def _rename(
        self, uses: list[tuple[str, int]], path: str = "tools/x/Limits.h"
    ) -> dict:
        replacements = [(path, 0, 8, "kMaxSize")] + [
            (SANDBOX + use, offset, 8, "kMaxSize") for use, offset in uses
        ]
        return dict(RENAME, path=path, replacements=replacements)

    def _diagnostics(self, *files: str) -> list[clang_tidy_apply.Diagnostic]:
        diagnostics = []
        for text in files:
            diagnostics += clang_tidy_apply.parse_fixes(text)
        return diagnostics

    def _read(self, path: str) -> str:
        with open(os.path.join(self.root, path), encoding="utf-8") as f:
            return f.read()

    def test_a_rename_reported_by_every_translation_unit_lands_once_everywhere(self):
        diagnostics = self._diagnostics(
            _fixes("tools/x/Limits.h", [self._rename([])]),
            _fixes("tools/x/A.cpp", [self._rename([("tools/x/A.cpp", 8)])]),
            _fixes("tools/x/B.cpp", [self._rename([("tools/x/B.cpp", 8)])]),
        )
        result = clang_tidy_apply.plan(diagnostics, ["readability-*"], [], self.root)
        self.assertEqual((result.applied, result.refused), (1, []))
        self.assertEqual(
            clang_tidy_apply.apply(result, self.root),
            ["tools/x/A.cpp", "tools/x/B.cpp", "tools/x/Limits.h"],
        )
        self.assertEqual(self._read("tools/x/Limits.h"), "kMaxSize;\n")
        self.assertEqual(self._read("tools/x/A.cpp"), "int a = kMaxSize;\n")
        self.assertEqual(self._read("tools/x/B.cpp"), "int b = kMaxSize;\n")

    def test_a_diagnostic_one_translation_unit_cannot_fix_is_applied_nowhere(self):
        # The renamer's "cannot be fixed because it is inside a macro": the same diagnostic without fix-its.
        diagnostics = self._diagnostics(
            _fixes("tools/x/A.cpp", [self._rename([("tools/x/A.cpp", 8)])]),
            _fixes(
                "tools/x/B.cpp",
                [dict(RENAME, path="tools/x/Limits.h", replacements=[])],
            ),
        )
        result = clang_tidy_apply.plan(diagnostics, ["*"], [], self.root)
        self.assertEqual(result.applied, 0)
        self.assertEqual(len(result.refused), 1)
        self.assertIn("without a fix-it", result.refused[0][1])
        self.assertEqual(clang_tidy_apply.apply(result, self.root), [])
        self.assertEqual(self._read("tools/x/A.cpp"), "int a = MAX_SIZE;\n")

    def test_a_fix_it_in_generated_code_is_never_applied(self):
        generated = "bazel-out/k8-opt/bin/tools/x/x.pb.h"
        diagnostics = self._diagnostics(
            _fixes(
                "tools/x/A.cpp", [self._rename([("tools/x/A.cpp", 8), (generated, 0)])]
            )
        )
        result = clang_tidy_apply.plan(diagnostics, ["*"], [], self.root)
        self.assertEqual(result.applied, 0)
        self.assertIn("outside the source tree", result.refused[0][1])
        self.assertEqual(self._read("tools/x/Limits.h"), "MAX_SIZE;\n")

    def test_paths_never_edit_another_directorys_files(self):
        diagnostics = self._diagnostics(
            _fixes(
                "tools/x/A.cpp",
                [self._rename([("tools/x/A.cpp", 8), ("tools/y/C.cpp", 8)])],
            )
        )
        result = clang_tidy_apply.plan(diagnostics, ["*"], ["tools/x"], self.root)
        self.assertEqual(result.applied, 0)
        self.assertIn("outside --paths", result.refused[0][1])
        # A diagnostic located elsewhere is not considered at all.
        self.assertEqual(
            clang_tidy_apply.plan(diagnostics, ["*"], ["tools/y"], self.root).refused,
            [],
        )

    def test_overlapping_fix_its_are_not_both_applied(self):
        other = {
            "check": "modernize-x",
            "message": "other",
            "offset": 2,
            "path": "tools/x/Limits.h",
        }
        other["replacements"] = [("tools/x/Limits.h", 2, 3, "Y")]
        diagnostics = self._diagnostics(
            _fixes("tools/x/Limits.h", [self._rename([]), other])
        )
        result = clang_tidy_apply.plan(diagnostics, ["*"], [], self.root)
        self.assertEqual(result.applied, 1)
        self.assertEqual(len(result.refused), 1)
        self.assertIn("overlaps", result.refused[0][1])

    def test_only_the_selected_checks_are_applied(self):
        diagnostics = self._diagnostics(_fixes("tools/x/Limits.h", [self._rename([])]))
        result = clang_tidy_apply.plan(diagnostics, ["modernize-*"], [], self.root)
        self.assertEqual(
            (result.applied, result.refused, result.replacements), (0, [], {})
        )

    def test_a_file_without_findings_has_no_diagnostics(self):
        self.assertEqual(clang_tidy_apply.parse_fixes(""), [])
        self.assertEqual(clang_tidy_apply.parse_fixes(_fixes("tools/x/A.cpp", [])), [])

    def test_the_command_line_applies_the_fixes_of_the_build(self):
        fixes = _write(
            self.root, "out/a.yaml", _fixes("tools/x/Limits.h", [self._rename([])])
        )
        bep = _bep(self.root, {"clang_tidy_fixes": [fixes]})
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = clang_tidy_apply.main(
                [
                    bep,
                    "--checks",
                    "readability-identifier-naming",
                    "--root",
                    self.root,
                    "--no-format",
                ]
            )
        self.assertEqual(status, 0)
        self.assertIn("1 applied in 1 files", output.getvalue())
        self.assertEqual(self._read("tools/x/Limits.h"), "kMaxSize;\n")


class SelftestTest(unittest.TestCase):
    def test_expected_files_allow_comments(self):
        text = "# why\n\ntools/x/A.cpp:1:2: google-runtime-int\n"
        self.assertEqual(
            clang_tidy_selftest.expected_findings(text),
            ["tools/x/A.cpp:1:2: google-runtime-int"],
        )

    def test_a_fixture_must_match_its_expected_file_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            report = _write(
                directory, "a.txt", HEADER_FINDING + SANDBOX + HEADER_FINDING
            )
            expected = _write(
                directory,
                "fixture.expected",
                "tools/x/include/x/Limits.h:5:22: readability-identifier-naming\n",
            )
            fixtures = [{"name": "fixture", "reports": [report]}]
            self.assertEqual(
                clang_tidy_selftest.check_fixtures(fixtures, [expected]), []
            )
            _write(directory, "fixture.expected", "")
            self.assertEqual(
                len(clang_tidy_selftest.check_fixtures(fixtures, [expected])), 1
            )
            self.assertEqual(len(clang_tidy_selftest.check_fixtures([], [expected])), 1)
            self.assertEqual(len(clang_tidy_selftest.check_fixtures(fixtures, [])), 1)


if __name__ == "__main__":
    unittest.main()
