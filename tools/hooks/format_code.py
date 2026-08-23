#!/usr/bin/env python3
"""
Code formatter and linter for wb_humanoid_mpc.
- Enforces trailing newlines on all source files.
- Trims trailing whitespace.
- Formats C/C++ files with clang-format.
- Formats Python files with black.
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
        if rel == exc or rel.startswith(exc + os.sep) or (os.sep + exc + os.sep) in rel:
            return True
    return False


def fix_trailing_newlines_and_whitespace(file_path):
    """Ensures file ends with exactly one newline and removes trailing whitespace."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return False

    if not content:
        return False

    lines = content.splitlines()
    trimmed_lines = [line.rstrip() for line in lines]
    new_content = "\n".join(trimmed_lines) + "\n"

    if new_content != content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return True
    return False


def main():
    modified_files = []
    cpp_files = []
    py_files = []

    # Collect files
    for root, dirs, files in os.walk(REPO_ROOT):
        # Prune excluded directories in-place
        dirs[:] = [d for d in dirs if not should_skip(os.path.join(root, d))]

        for f in files:
            full_path = os.path.join(root, f)
            if should_skip(full_path):
                continue

            ext = os.path.splitext(f)[1].lower()
            if ext in TEXT_EXTENSIONS or f in EXACT_FILES:
                if fix_trailing_newlines_and_whitespace(full_path):
                    modified_files.append(os.path.relpath(full_path, REPO_ROOT))

            if ext in {".cpp", ".h", ".hpp"}:
                cpp_files.append(full_path)
            elif ext == ".py":
                py_files.append(full_path)

    # Run clang-format on C++ files if available
    if cpp_files:
        try:
            subprocess.run(
                ["clang-format", "-i"] + cpp_files,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            pass

    # Run black on Python files if available
    if py_files:
        try:
            subprocess.run(
                ["black", "--quiet"] + py_files,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            pass

    if modified_files:
        print(
            f"✨ Formatted and ensured trailing newlines on {len(modified_files)} file(s)."
        )
    else:
        print(
            "✅ All files properly formatted with required blank lines and trailing newlines."
        )


if __name__ == "__main__":
    main()
