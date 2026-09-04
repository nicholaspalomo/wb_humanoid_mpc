"""****************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
****************************************************************************"""

import os
import re
import shutil
from typing import Any, Dict, List, Tuple
import yaml


def load_yaml_safe(file_path: str) -> Dict[str, Any]:
    """Load YAML file safely returning a dictionary."""
    if not file_path or not os.path.exists(file_path):
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def update_yaml_values_in_place(
    file_path: str,
    key_value_updates: List[Tuple[List[str], float]],
    create_backup: bool = True,
) -> bool:
    """
    Updates numeric parameter values directly in a YAML file while preserving all
    comments, annotations, whitespace, and formatting.

    Args:
        file_path: Absolute path to the YAML file.
        key_value_updates: List of (key_path, new_value) tuples, e.g.:
            [(['joint_gains', 'left_knee_joint', 'kp'], 250.0),
             (['Q', '"(8,8)"'], 35.0)]
        create_backup: If True, copies the original file to file_path + '.bak'.

    Returns:
        True if the file was updated successfully.
    """
    if not os.path.exists(file_path):
        return False

    if create_backup:
        shutil.copy2(file_path, file_path + ".bak")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for key_path, new_val in key_value_updates:
        lines = _update_single_key(lines, key_path, new_val)

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return True


def _update_single_key(
    lines: List[str], key_path: List[str], new_val: float
) -> List[str]:
    """Updates a single key specified by hierarchical path within lines."""
    if not key_path:
        return lines

    # Target key at the leaf of key_path
    leaf_key = key_path[-1].strip("\"'")
    parent_path = [k.strip("\"'") for k in key_path[:-1]]

    # Search through lines tracking the active section hierarchy based on indentation
    new_lines = []
    section_stack: List[Tuple[int, str]] = []  # (indent_level, section_name)

    updated = False

    for line in lines:
        stripped = line.strip()
        # Skip pure comments or blank lines
        if not stripped or stripped.startswith("#"):
            new_lines.append(line)
            continue

        indent = len(line) - len(line.lstrip())

        # Pop sections that are deeper or equal to current indent
        while section_stack and section_stack[-1][0] >= indent:
            section_stack.pop()

        current_sections = [s[1] for s in section_stack]

        # Check if line defines a section header or a key-value
        colon_idx = line.find(":")
        if colon_idx != -1:
            raw_key = line[:colon_idx].strip().strip("\"'")
            after_colon = line[colon_idx + 1 :].strip()

            # Check if this line is our target key within the correct parent hierarchy
            if not updated and raw_key == leaf_key and current_sections == parent_path:
                # Replace the numeric value before any inline comment
                # E.g. "  kp: 150.0  # units N*m" -> "  kp: 250.0  # units N*m"
                prefix = line[: colon_idx + 1] + " "
                comment_part = ""
                hash_idx = after_colon.find("#")
                if hash_idx != -1:
                    comment_part = "  " + after_colon[hash_idx:]

                # Format new numeric value
                val_str = f"{new_val:.6g}"
                new_line = prefix + val_str + comment_part + "\n"
                new_lines.append(new_line)
                updated = True
                continue

            # If it's a section header (value after colon is empty or just a comment)
            if not after_colon or after_colon.startswith("#"):
                section_stack.append((indent, raw_key))

        new_lines.append(line)

    return new_lines
