#!/usr/bin/env python3
"""
Linter for wb_humanoid_mpc repository.
- Validates Google LINT.IfChange / LINT.ThenChange cross-file directives.
- Checks trailing whitespace and missing EOF newlines.
- Checks C/C++ formatting with clang-format (--dry-run --Werror).
- Checks that bare literal arguments carry a Google-style argument comment (argument_comments.py).
- Checks that Abseil headers are included with quotes, as Bazel exposes them (include_style.py).
- Checks that the repository is written in American English (american_spelling.py).
- Checks Python formatting with black (--check).
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argument_comments  # noqa: E402
import include_style  # noqa: E402
import american_spelling  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

EXCLUDE_DIRS = {
    ".git",
    ".bazel",
    ".bazel_ros_install",
    "bazel-bin",
    "bazel-out",
    "bazel-testlogs",
    "bazel-wb_humanoid_mpc",
    "build",
    "install",
    "log",
    # LINT.IfChange(vendored_dirs)
    "lib/ocs2",
    "lib/mujoco_vendor",
    "tools/ifttt-lint",
    # LINT.ThenChange(//tools/hooks/argument_comments.py:vendored_dirs)
}

TEXT_EXTENSIONS = {
    ".py",
    ".cpp",
    ".h",
    ".hpp",
    ".bzl",
    ".bazel",
    ".sh",
    ".bash",
    ".yaml",
    ".yml",
    ".json",
    ".md",
    ".txt",
    ".cfg",
    ".xml",
    ".urdf",
    ".xacro",
    ".mjcf",
}

EXACT_FILES = {
    "Makefile",
    "BUILD",
    "BUILD.bazel",
    "MODULE.bazel",
    "WORKSPACE",
    ".bazelrc",
    ".bazelversion",
    ".clang-format",
    ".gitignore",
}


SPELLING_SKIPPED_EXTENSIONS = {".urdf", ".xacro", ".xml", ".mjcf"}
SPELLING_SKIPPED_FILES = {
    os.path.join("tools", "hooks", "american_spelling.py"),
    os.path.join("tools", "hooks", "test_american_spelling.py"),
}


def should_skip(path):
    rel = os.path.relpath(path, REPO_ROOT)
    for exc in EXCLUDE_DIRS:
        if rel == exc or rel.startswith(exc + os.sep) or (os.sep + exc + os.sep) in rel:
            return True
    return False


def check_trailing_newlines_and_whitespace(file_path):
    """Checks if file ends with exactly one newline and has no trailing whitespace."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return False, "Unable to read file"

    if not content:
        return True, ""

    lines = content.splitlines()
    trimmed_lines = [line.rstrip() for line in lines]
    expected_content = "\n".join(trimmed_lines) + "\n"

    if content != expected_content:
        return False, "Trailing whitespace or missing/extra EOF newline"
    return True, ""


def main():
    print("🔍 1/7 Checking IFTTT cross-file directives...")
    ifttt_script = os.path.join(os.path.dirname(__file__), "check_ifttt.py")
    ifttt_res = subprocess.run(
        [sys.executable, ifttt_script], capture_output=True, text=True
    )
    if ifttt_res.returncode != 0:
        print(ifttt_res.stdout)
        print(ifttt_res.stderr)
        return 1

    print("🔍 2/7 Checking trailing whitespace and EOF newlines...")
    whitespace_errors = []
    spelling_files = []
    cpp_files = []
    py_files = []

    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if not should_skip(os.path.join(root, d))]
        for f in files:
            full_path = os.path.join(root, f)
            if should_skip(full_path):
                continue

            ext = os.path.splitext(f)[1].lower()
            rel_path = os.path.relpath(full_path, REPO_ROOT)

            if ext in TEXT_EXTENSIONS or f in EXACT_FILES:
                valid, msg = check_trailing_newlines_and_whitespace(full_path)
                if not valid:
                    whitespace_errors.append(f"{rel_path}: {msg}")
                # Robot model files (URDF, MJCF) are upstream data; the checker's own word list is British on purpose.
                if (
                    ext not in SPELLING_SKIPPED_EXTENSIONS
                    and rel_path not in SPELLING_SKIPPED_FILES
                ):
                    spelling_files.append(full_path)

            if ext in {".cpp", ".h", ".hpp"}:
                cpp_files.append(full_path)
            elif ext == ".py":
                py_files.append(full_path)

    if whitespace_errors:
        print("❌ Whitespace/Newline Errors:")
        for err in whitespace_errors:
            print(f"  {err}")
        print("💡 Run 'make format' to auto-fix whitespace and newlines.")
        return 1

    print("🔍 3/7 Checking C++ formatting (clang-format)...")
    cpp_errors = []
    if cpp_files:
        try:
            res = subprocess.run(
                ["clang-format", "--dry-run", "--Werror"] + cpp_files,
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                print(res.stderr)
                cpp_errors.append("clang-format violations detected.")
        except FileNotFoundError:
            print("⚠️ Warning: clang-format not found, skipping C++ format lint.")

    if cpp_errors:
        print("💡 Run 'make format' to auto-format C++ code.")
        return 1

    print("🔍 4/7 Checking argument comments on literal arguments...")
    violations = argument_comments.check_files(cpp_files, REPO_ROOT)
    if violations:
        print("❌ Literal arguments without an argument comment:")
        for violation in violations:
            print(f"  {violation}")
        print(
            f"💡 {len(violations)} call site(s): write /*parameter_name=*/ in front of each literal, with the name "
            "from the callee's declaration (see tools/hooks/argument_comments.py for what is exempt)."
        )
        return 1

    print("🔍 5/7 Checking that Abseil headers are included with quotes...")
    include_violations = include_style.check_files(cpp_files, REPO_ROOT)
    if include_violations:
        print("❌ Abseil headers included with angle brackets:")
        for violation in include_violations:
            print(f"  {violation}")
        return 1

    print("🔍 6/7 Checking for British spellings (American English throughout)...")
    # Only files git would track: ignored side files (the tuning GUI's *.live.yaml copies, *.bak backups) are not ours
    # to fix and are rewritten from their sources anyway.
    try:
        listed = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split("\0")
        tracked = {os.path.join(REPO_ROOT, path) for path in listed if path}
        spelling_files = [path for path in spelling_files if path in tracked]
    except (OSError, subprocess.CalledProcessError):
        pass
    spelling_findings = american_spelling.check_files(spelling_files, REPO_ROOT)
    if spelling_findings:
        print("❌ British spellings:")
        for finding in spelling_findings:
            print(f"  {finding}")
        print(
            "💡 Fix them all with: python3 tools/hooks/american_spelling.py --fix <files>"
        )
        return 1

    print("🔍 7/7 Checking Python formatting (black)...")
    py_errors = []
    if py_files:
        try:
            res = subprocess.run(
                ["black", "--check"] + py_files,
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                print(res.stdout)
                print(res.stderr)
                py_errors.append("black format violations detected.")
        except FileNotFoundError:
            print("⚠️ Warning: black not found, skipping Python format lint.")

    if py_errors:
        print("💡 Run 'make format' to auto-format Python code.")
        return 1

    print("✅ All lint checks passed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
