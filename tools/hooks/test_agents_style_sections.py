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

"""The style sections of AGENTS.md name only checks that exist.

AGENTS.md says which tool enforces each rule: a check of the registry (tools/hooks/checks.py), a clang-tidy check, a
cpplint category or a GCC flag of bazel/copts.bzl. IFTTT directives prompt a review of those lists when the registry or
a configuration changes, but they cannot tell a misspelled or deleted name from a real one; this test can. A name that
fails here is either a typo in AGENTS.md or a check that was renamed or removed without updating it.
"""

import re
import unittest

from tools.hooks import check_test_support
from tools.hooks import checks

AGENTS = "AGENTS.md"
CLANG_TIDY = ".clang-tidy"
COPTS = "bazel/copts.bzl"

# The labeled blocks of AGENTS.md that the registry and the configurations point at (LINT.ThenChange targets).
STYLE_LABELS = (
    "cpp_style",
    "nullability",
    "totw_rules",
    "python_style",
    "style_commands",
    "protobuf_rules",
)

# The families of clang-tidy checks .clang-tidy draws from; a backticked name with one of these prefixes is a check.
_CLANG_TIDY_PREFIXES = (
    "abseil-",
    "bugprone-",
    "cert-",
    "clang-diagnostic-",
    "cppcoreguidelines-",
    "google-",
    "misc-",
    "modernize-",
    "performance-",
    "readability-",
)
_CHECK_ENTRY = re.compile(r'^\s+- "?(?P<name>-\*|[a-z][a-z0-9.-]*)"?\s*(#.*)?$')
_COPT = re.compile(r'^\s+"(?P<flag>-W[^"]+)",')
_BACKTICKED = re.compile(r"`([^`]+)`")
_NOLINT = re.compile(r"NOLINT(?:NEXTLINE|BEGIN|END)?\((?P<names>[a-z0-9, -]+)\)")
_CPPLINT_CATEGORY = re.compile(r"^cpplint `(?P<category>[a-z_+]+/[a-z_+]+)`$")


def _clang_tidy_checks() -> set[str]:
    """Every check of //:.clang-tidy."""
    names = set()
    in_checks = False
    for line in check_test_support.read_repository_file(CLANG_TIDY).splitlines():
        if line.startswith("Checks:"):
            in_checks = True
            continue
        if not in_checks or line.lstrip().startswith("#"):
            continue
        match = _CHECK_ENTRY.match(line)
        if not match:
            break
        names.add(match.group("name"))
    names.discard("-*")
    return names


def _copts() -> set[str]:
    """The flags of FIRST_PARTY_COPTS, one per line of bazel/copts.bzl."""
    flags = set()
    for line in check_test_support.read_repository_file(COPTS).splitlines():
        match = _COPT.match(line)
        if match:
            flags.add(match.group("flag"))
    return flags


def _block(text: str, label: str) -> str:
    """The lines between `<!-- LINT.IfChange(label) -->` and the next LINT.ThenChange."""
    start = text.index(f"<!-- LINT.IfChange({label}) -->")
    return text[start : text.index("LINT.ThenChange", start)]


def _style_sections(text: str) -> str:
    """AGENTS.md from the "Style guides" heading on."""
    return text[text.index("#### Style guides") :]


def _werror_flags(token: str) -> list[str]:
    """`-Werror=a,b,c` written as one token in AGENTS.md: the flags `-Werror=a`, `-Werror=b` and `-Werror=c`."""
    return [f"-Werror={name}" for name in token.removeprefix("-Werror=").split(",")]


class AgentsStyleSectionsTest(unittest.TestCase):
    """The names AGENTS.md cites resolve to a registry check, a clang-tidy check, a cpplint category or a GCC flag."""

    agents: str
    style: str
    clang_tidy: set[str]
    copts: set[str]

    @classmethod
    def setUpClass(cls) -> None:
        cls.agents = check_test_support.read_repository_file(AGENTS)
        cls.style = _style_sections(cls.agents)
        cls.clang_tidy = _clang_tidy_checks()
        cls.copts = _copts()

    def test_the_labeled_blocks_exist(self) -> None:
        for label in STYLE_LABELS:
            with self.subTest(label=label):
                self.assertIn(f"<!-- LINT.IfChange({label}) -->", self.agents)

    def test_the_parsers_see_the_configurations(self) -> None:
        # A parser that silently found nothing would make every other test here pass.
        self.assertIn("bugprone-use-after-move", self.clang_tidy)
        self.assertIn("-Werror=unused-result", self.copts)
        self.assertGreater(len(self.clang_tidy), 50)

    def test_every_tool_of_the_totw_table_exists(self) -> None:
        rows = [
            line
            for line in _block(self.agents, "totw_rules").splitlines()
            if line.startswith("| #")
        ]
        self.assertGreater(len(rows), 30)
        for row in rows:
            checked_by = row.rstrip("|").rsplit("|", 1)[1]
            for item in checked_by.split(", "):
                item = (
                    item.strip()
                    .removeprefix("clang-tidy ")
                    .removesuffix(" (fixed by `make format`)")
                )
                with self.subTest(tip=row.split("|")[1].strip(), item=item):
                    if _CPPLINT_CATEGORY.match(item):
                        continue
                    self.assertRegex(item, r"^`[^`]+`$", "one backticked name per item")
                    name = item.strip("`")
                    if name.startswith("-Werror="):
                        for flag in _werror_flags(name):
                            self.assertIn(flag, self.copts, f"{flag} is not in {COPTS}")
                    elif name.startswith(_CLANG_TIDY_PREFIXES):
                        self.assertIn(
                            name,
                            self.clang_tidy,
                            f"{name} is not in {CLANG_TIDY}",
                        )
                    else:
                        self.assertIn(
                            name,
                            checks.names(),
                            f"{name} is not a check of tools/hooks/checks.py",
                        )

    def test_every_clang_tidy_check_named_exists(self) -> None:
        for token in _BACKTICKED.findall(self.style):
            if token.startswith(_CLANG_TIDY_PREFIXES) and re.fullmatch(
                r"[a-z0-9-]+", token
            ):
                with self.subTest(check=token):
                    self.assertIn(token, self.clang_tidy)

    def test_every_gcc_flag_named_is_in_copts(self) -> None:
        for token in _BACKTICKED.findall(self.style):
            if token.startswith("-Werror="):
                for flag in _werror_flags(token):
                    with self.subTest(flag=flag):
                        self.assertIn(flag, self.copts)

    def test_every_nolint_category_named_exists(self) -> None:
        known = checks.names() | self.clang_tidy
        found = 0
        for match in _NOLINT.finditer(self.style):
            for name in match.group("names").split(","):
                name = name.strip()
                found += 1
                with self.subTest(name=name):
                    self.assertIn(name, known)
        self.assertGreater(found, 0)

    def test_every_registry_check_is_named(self) -> None:
        # checks.py's registry IFTTT block says that AGENTS.md lists what every check enforces; this keeps it true.
        for name in sorted(checks.names()):
            with self.subTest(check=name):
                named = f"`{name}`" in self.agents or re.search(
                    r"NOLINT(?:NEXTLINE|BEGIN|END)?\([^)]*\b" + re.escape(name) + r"\b",
                    self.agents,
                )
                self.assertTrue(named, f"AGENTS.md does not name `{name}`")

    def test_the_clang_tidy_checked_rules_are_not_agent_rules(self) -> None:
        agent_rules = self.agents[
            self.agents.index(
                "Agent rules (no tool checks these):"
            ) : self.agents.index("#### Raw pointers carry nullability")
        ]
        self.assertNotIn("Move constructors are `noexcept`", agent_rules)
        self.assertNotIn(
            "Default arguments go only on non-virtual functions", agent_rules
        )
        self.assertIn("performance-noexcept-move-constructor", self.style)
        self.assertIn("google-default-arguments", self.style)

    def test_a_misspelled_name_is_caught(self) -> None:
        self.assertNotIn("bugprone-use-after-mvoe", self.clang_tidy)
        self.assertNotIn("totw-at-all", checks.names())
        self.assertEqual(
            _werror_flags("-Werror=pessimizing-move,redundant-move"),
            ["-Werror=pessimizing-move", "-Werror=redundant-move"],
        )


if __name__ == "__main__":
    unittest.main()
