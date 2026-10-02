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

"""Which configuration files the GUI edits, found in the source checkout rather than in Bazel's runfiles.

The tuning tabs write their files back ("Save to YAML"), so they must edit the files of the checkout the operator is
working in. Under `bazel run` the GUI starts in its runfiles tree, where the robot's files are links into a build
output; this module finds the checkout (BUILD_WORKSPACE_DIRECTORY, or the directory above this source file) and
re-roots every robot_models/ path into it.

Free of Tk and of the bus, so that test/test_source_path_resolution.py covers it headless.
"""

import dataclasses
import logging
import os
from typing import Optional, Sequence

import yaml

_LOGGER = logging.getLogger(__name__)

# The directory of the robot packages, under the repository root.
ROBOT_MODELS_DIR = "robot_models"
# A file only the repository root has.
REPOSITORY_MARKER = "MODULE.bazel"

# The task file the GUI opens when it is given none and finds none next to its other files: the first that exists.
DEFAULT_TASK_FILES: Sequence[str] = (
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml",
    "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.yaml",
    "robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.yaml",
)
# Likewise for the joint PD gains.
DEFAULT_PD_GAINS_FILES: Sequence[str] = (
    "robot_models/unitree_g1/g1_wb_mpc/config/controller/joint_pd_gains.yaml",
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/controller/joint_pd_gains.yaml",
)

# The layout of a robot's config directory: config/mpc/task.yaml, config/command/reference.yaml and
# config/controller/joint_pd_gains.yaml.
_TASK_FROM_REFERENCE = ("..", "mpc", "task.yaml")
_REFERENCE_FROM_TASK = ("..", "command", "reference.yaml")
_PD_GAINS_FROM_TASK = ("..", "controller", "joint_pd_gains.yaml")

# The pelvis height the walking command starts at when reference.yaml names none [m].
DEFAULT_PELVIS_HEIGHT = 0.8


def _is_repository_root(directory: str) -> bool:
    return os.path.isfile(os.path.join(directory, REPOSITORY_MARKER)) and os.path.isdir(
        os.path.join(directory, ROBOT_MODELS_DIR)
    )


def _walk_up(start: str) -> Optional[str]:
    directory = os.path.abspath(start)
    while True:
        if _is_repository_root(directory):
            return directory
        parent = os.path.dirname(directory)
        if parent == directory:
            return None
        directory = parent


def find_repo_root(start: Optional[str] = None) -> Optional[str]:
    """The root of the source checkout, or None when there is none.

    In order: BUILD_WORKSPACE_DIRECTORY (set by `bazel run`), the checkout this source file lies in (its real path:
    Bazel's runfiles link to the sources), and the directories above `start` (default: the working directory). A root
    holds MODULE.bazel and robot_models/.
    """
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if workspace and _is_repository_root(workspace):
        return os.path.abspath(workspace)
    from_source = _walk_up(os.path.dirname(os.path.realpath(__file__)))
    if from_source:
        return from_source
    return _walk_up(start or os.getcwd())


def resolve_input_path(path: str, repo_root: Optional[str] = None) -> str:
    """Finds a file named on the command line.

    An absolute path is returned as it is. A relative one is looked up in the directory the program was started from
    (BUILD_WORKING_DIRECTORY under `bazel run`, which changes into the runfiles tree), then in the repository root.
    Returns the first candidate that exists, or the first candidate when none does, so that an error names the path
    the user meant.
    """
    if not path or os.path.isabs(path):
        return path
    candidates = [
        os.path.join(os.environ.get("BUILD_WORKING_DIRECTORY", os.getcwd()), path)
    ]
    root = repo_root or find_repo_root()
    if root:
        candidates.append(os.path.join(root, path))
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(candidates[0])


def resolve_source_path(path: str, repo_root: Optional[str]) -> str:
    """The copy of `path` in the source checkout `repo_root`, when `path` is a robot file that has one.

    A path through robot_models/ (a runfiles link, a copied tree) is re-rooted at `repo_root`, so that saving edits the
    file the developer is working on. Any other path, a path whose source copy does not exist, an empty path and a path
    that does not exist come back unchanged.
    """
    if not path or not repo_root or not os.path.exists(path):
        return path
    marker = os.sep + ROBOT_MODELS_DIR + os.sep
    normalized = os.path.abspath(path)
    index = normalized.rfind(marker)
    if index == -1:
        return path
    candidate = os.path.join(repo_root, normalized[index + 1 :])
    return candidate if os.path.exists(candidate) else path


def _sibling(path: str, relative: Sequence[str]) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(path), *relative))


def _first_existing(paths: Sequence[str]) -> str:
    for path in paths:
        if path and os.path.exists(path):
            return path
    return ""


@dataclasses.dataclass(frozen=True)
class GuiConfigFiles:
    """The files the GUI's tabs edit; an empty path is a file the GUI could not find."""

    task_file: str
    reference_file: str
    pd_gains_file: str


def resolve_config_files(
    task_file: str = "",
    reference_file: str = "",
    pd_gains_file: str = "",
    repo_root: Optional[str] = None,
) -> GuiConfigFiles:
    """The task, reference and PD gain files of the GUI, each in the source checkout.

    A file given and present is used (re-rooted into the checkout). A missing one is looked for next to the others,
    in a robot's config/ layout (config/mpc/task.yaml, config/command/reference.yaml,
    config/controller/joint_pd_gains.yaml), and the task and PD gain files then among the DEFAULT_*_FILES of the
    checkout.
    """
    task = resolve_source_path(task_file, repo_root) if task_file else ""
    reference = resolve_source_path(reference_file, repo_root) if reference_file else ""
    pd_gains = resolve_source_path(pd_gains_file, repo_root) if pd_gains_file else ""

    if not os.path.exists(task):
        candidates = [_sibling(reference, _TASK_FROM_REFERENCE)] if reference else []
        if repo_root:
            candidates.extend(
                os.path.join(repo_root, path) for path in DEFAULT_TASK_FILES
            )
        task = _first_existing(candidates)
    if not os.path.exists(reference):
        reference = _first_existing(
            [_sibling(task, _REFERENCE_FROM_TASK)] if task else []
        )
    if not os.path.exists(pd_gains):
        candidates = [_sibling(task, _PD_GAINS_FROM_TASK)] if task else []
        if repo_root:
            candidates.extend(
                os.path.join(repo_root, path) for path in DEFAULT_PD_GAINS_FILES
            )
        pd_gains = _first_existing(candidates)
    return GuiConfigFiles(
        task_file=task, reference_file=reference, pd_gains_file=pd_gains
    )


def read_default_pelvis_height(
    reference_file: str, fallback: float = DEFAULT_PELVIS_HEIGHT
) -> float:
    """reference.yaml's defaultBaseHeight [m], or `fallback` when the file has none or cannot be read."""
    if not reference_file or not os.path.exists(reference_file):
        return fallback
    try:
        with open(reference_file, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if isinstance(data, dict) and "defaultBaseHeight" in data:
            return float(data["defaultBaseHeight"])
    except (OSError, ValueError, TypeError, yaml.YAMLError) as error:
        _LOGGER.warning(
            "Could not read defaultBaseHeight from %s: %s", reference_file, error
        )
    return fallback


def read_online_tuning(task_file: str) -> bool:
    """The task file's enableOnlineTuning (or enable_online_tuning); True when it has neither or cannot be read."""
    if not task_file or not os.path.exists(task_file):
        return True
    try:
        with open(task_file, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as error:
        _LOGGER.warning(
            "Could not read the online tuning flag of %s: %s", task_file, error
        )
        return True
    if not isinstance(data, dict):
        return True
    for key in ("enableOnlineTuning", "enable_online_tuning"):
        if key in data:
            return bool(data[key])
    return True
