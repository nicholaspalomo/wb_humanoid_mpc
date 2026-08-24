#!/usr/bin/env python3
"""
Linter for wb_humanoid_mpc repository.
- Validates Google LINT.IfChange / LINT.ThenChange cross-file directives.
- Checks trailing whitespace and missing EOF newlines.
- Checks C/C++ formatting with clang-format (--dry-run --Werror).
- Checks Python formatting with black (--check).
"""

import os
import subprocess
import sys

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
    "lib/ocs2",
    "lib/mujoco_vendor",
    "tools/ifttt-lint",
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


def should_skip(path):
    rel = os.path.relpath(path, REPO_ROOT)
    for exc in EXCLUDE_DIRS:
        if (
            rel == exc
            or rel.startswith(exc + os.sep)
            or (os.sep + exc + os.sep) in rel
        ):
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
    print("🔍 1/4 Checking IFTTT cross-file directives...")
    ifttt_script = os.path.join(os.path.dirname(__file__), "check_ifttt.py")
    ifttt_res = subprocess.run(
        [sys.executable, ifttt_script], capture_output=True, text=True
    )
    if ifttt_res.returncode != 0:
        print(ifttt_res.stdout)
        print(ifttt_res.stderr)
        return 1

    print("🔍 2/4 Checking trailing whitespace and EOF newlines...")
    whitespace_errors = []
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

    print("🔍 3/4 Checking C++ formatting (clang-format)...")
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
            print(
                "⚠️ Warning: clang-format not found, skipping C++ format lint."
            )

    if cpp_errors:
        print("💡 Run 'make format' to auto-format C++ code.")
        return 1

    print("🔍 4/4 Checking Python formatting (black)...")
    py_errors = []
    if py_files:
        try:
            res = subprocess.run(
                ["black", "--line-length", "80", "--check"] + py_files,
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
