#!/usr/bin/env python3
"""Flags Abseil headers included with angle brackets.

Bazel makes the @abseil-cpp headers visible through `-iquote` paths, which only a quoted include searches. An angle
include, `#include <absl/log/log.h>`, skips them and falls through to the system include directories, which hold a
different Abseil release or none. The dev image once built Abseil 20240722 into /usr/local (removed from
docker/Dockerfile since), so such a file compiled there against headers of another version than the one it links
against, and failed in CI's clean ros:jazzy container with `absl/log/log.h: No such file or directory`. Write
`#include "absl/..."`.
"""

import re
import sys
from typing import Iterable, List, NamedTuple

_ANGLE_ABSL = re.compile(r"^\s*#\s*include\s*<(absl/[^>]+)>")


class Violation(NamedTuple):
    path: str
    line: int
    header: str

    def __str__(self) -> str:
        return (
            f'{self.path}:{self.line}: `#include <{self.header}>` - write `#include "{self.header}"`: Bazel exposes '
            "@abseil-cpp to quoted includes only, and an angle include finds a system Abseil or none (see "
            "tools/hooks/include_style.py)."
        )


def check_source(source: str, path: str = "<source>") -> List[Violation]:
    violations = []
    for number, line in enumerate(source.splitlines(), start=1):
        match = _ANGLE_ABSL.match(line)
        if match:
            violations.append(Violation(path, number, match.group(1)))
    return violations


def check_files(paths: Iterable[str], root: str) -> List[Violation]:
    import os

    violations: List[Violation] = []
    for path in paths:
        with open(path, encoding="utf-8", errors="ignore") as f:
            violations += check_source(f.read(), os.path.relpath(path, root))
    return violations


if __name__ == "__main__":
    import os

    found = check_files(sys.argv[1:], os.getcwd())
    for violation in found:
        print(violation)
    sys.exit(1 if found else 0)
