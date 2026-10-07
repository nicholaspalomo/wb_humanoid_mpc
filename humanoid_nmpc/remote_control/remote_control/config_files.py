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

"""Which configuration files the GUI edits, found in the source checkout rather than in Bazel's runfiles.

The tuning tabs write their files back ("Save"), so they must edit the files of the checkout the operator is working
in. Under `bazel run` the GUI starts in its runfiles tree, where the robot's files are links into a build output; this
module finds the checkout (BUILD_WORKSPACE_DIRECTORY, or the directory above this source file) and re-roots every
robot_models/ path into it.

The files are textprotos of the humanoid_mpc_config schemas (humanoid_nmpc/humanoid_mpc_config/README.md): a robot's
config/mpc/task.textproto, config/command/reference.textproto, config/controller/joint_pd_gains.textproto and
config/mpc/contact_planning.textproto.

Free of Tk and of the bus, so that test/test_source_path_resolution.py covers it headless.
"""

from collections.abc import Sequence
import dataclasses
import logging
import os

from humanoid_mpc_config import config_registries_pb2
from humanoid_mpc_config import joint_pd_gains_file_pb2
from humanoid_mpc_config import reference_file_pb2
from humanoid_mpc_config import task_file_pb2

import nproto_textproto

_LOGGER = logging.getLogger(__name__)

# The directory of the robot packages, under the repository root, which a configuration file's identity starts at
# (robot_config_save.config_path_of()).
# LINT.IfChange(robot_models_dir)
ROBOT_MODELS_DIR = "robot_models"
# LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/config/ConfigFiles.cpp:robot_models_directory)
# A file only the repository root has.
REPOSITORY_MARKER = "MODULE.bazel"
# The extension of the configuration files.
CONFIG_EXTENSION = ".textproto"

# The task file the GUI opens when it is given none and finds none next to its other files: the first that exists.
DEFAULT_TASK_FILES: Sequence[str] = (
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto",
    "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto",
    "robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto",
)
# Likewise for the joint PD gains.
DEFAULT_PD_GAINS_FILES: Sequence[str] = (
    "robot_models/unitree_g1/g1_wb_mpc/config/controller/joint_pd_gains.textproto",
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/controller/joint_pd_gains.textproto",
)

# The layout of a robot's config directory: config/mpc/task.textproto, config/mpc/contact_planning.textproto,
# config/command/reference.textproto and config/controller/joint_pd_gains.textproto.
# LINT.IfChange(config_layout)
_TASK_FROM_REFERENCE = ("..", "mpc", "task.textproto")
_REFERENCE_FROM_TASK = ("..", "command", "reference.textproto")
_PD_GAINS_FROM_TASK = ("..", "controller", "joint_pd_gains.textproto")
# LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/config/ConfigFiles.h:config_layout, //humanoid_nmpc/humanoid_mpc_validation/src/closed_loop/RobotConfiguration.cpp:config_layout)
# LINT.IfChange(contact_planning_file_name)
CONTACT_PLANNING_FILE_NAME = "contact_planning.textproto"
# LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/config/ConfigFiles.h:contact_planning_file_name)

# The pelvis height the walking command starts at when the reference file names none [m].
DEFAULT_PELVIS_HEIGHT = 0.8

# The names each registry accepts, by registry: the choices of a string field whose (humanoid_mpc_config.tuning)
# option names a registry. Written by `bazel run //tools/config_registries:print_config_registries`.
REGISTRIES_FILE = "humanoid_nmpc/humanoid_mpc_config/config_registries.textproto"


class ConfigFilesError(ValueError):
    """A configuration file the command line names that is not there: the GUI must not tune another robot's instead."""


def _is_repository_root(directory: str) -> bool:
    return os.path.isfile(os.path.join(directory, REPOSITORY_MARKER)) and os.path.isdir(
        os.path.join(directory, ROBOT_MODELS_DIR)
    )


def _walk_up(start: str) -> str | None:
    directory = os.path.abspath(start)
    while True:
        if _is_repository_root(directory):
            return directory
        parent = os.path.dirname(directory)
        if parent == directory:
            return None
        directory = parent


def find_repo_root(start: str | None = None) -> str | None:
    """The root of the source checkout, or None when there is none.

    In order: BUILD_WORKSPACE_DIRECTORY (set by `bazel run`), the checkout this source file lies in (its real path:
    Bazel's runfiles link to the sources), and the directories above `start` (default: the working directory). A root
    holds MODULE.bazel and robot_models/.

    Args:
        start: The directory the last search walks up from; None for the working directory.

    Returns:
        The absolute path of the checkout's root, or None when no candidate holds a checkout.
    """
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if workspace and _is_repository_root(workspace):
        return os.path.abspath(workspace)
    from_source = _walk_up(os.path.dirname(os.path.realpath(__file__)))
    if from_source:
        return from_source
    return _walk_up(start or os.getcwd())


def resolve_input_path(path: str, repo_root: str | None = None) -> str:
    """Finds a file named on the command line.

    An absolute path is returned as it is. A relative one is looked up in the directory the program was started from
    (BUILD_WORKING_DIRECTORY under `bazel run`, which changes into the runfiles tree), then in the repository root.

    Args:
        path: The path as the command line gave it; an empty one comes back unchanged.
        repo_root: The checkout to look in second; None to find it with find_repo_root().

    Returns:
        The first candidate that exists, or the first candidate when none does, so that an error names the path the
        user meant.
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


def resolve_source_path(path: str, repo_root: str | None) -> str:
    """The copy of `path` in the source checkout `repo_root`, when `path` is a robot file that has one.

    A path through robot_models/ (a runfiles link, a copied tree) is re-rooted at `repo_root`, so that saving edits the
    file the developer is working on.

    Args:
        path: The file the GUI was given.
        repo_root: The source checkout; None or empty leaves `path` unchanged.

    Returns:
        The checkout's copy of `path`; `path` itself when it is not under robot_models/, its source copy does not
        exist, it is empty or it does not exist.
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
    repo_root: str | None = None,
) -> GuiConfigFiles:
    """The task, reference and PD gain files of the GUI, each in the source checkout.

    A file given is used (re-rooted into the checkout); one given that does not exist is an error, never replaced by
    another robot's. A file not given is looked for next to the others, in a robot's config/ layout
    (config/mpc/task.textproto, config/command/reference.textproto, config/controller/joint_pd_gains.textproto), and
    the task and PD gain files then among the DEFAULT_*_FILES of the checkout.

    Args:
        task_file: The task file given on the command line, or empty.
        reference_file: The reference file given on the command line, or empty.
        pd_gains_file: The joint PD gains file given on the command line, or empty.
        repo_root: The source checkout the files are re-rooted into; None to use them where they are.

    Returns:
        The three files, each empty when it was not given and could not be found.

    Raises:
        ConfigFilesError: A file given does not exist; the message names it, and says that the configuration files
            are textprotos for a YAML file.
    """
    for label, given in (
        ("task file", task_file),
        ("reference file", reference_file),
        ("joint PD gains file", pd_gains_file),
    ):
        if given and not os.path.exists(given):
            hint = (
                f"; the configuration files are {CONFIG_EXTENSION} files"
                if given.endswith((".yaml", ".yml"))
                else ""
            )
            raise ConfigFilesError(f"the {label} {given} does not exist{hint}")
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


def task_file_beside(config_file: str) -> str:
    """The task file of the configuration a file of a robot's config/ directory belongs to.

    The task file is config/mpc/task.textproto, the same relative path from every config/<directory>/ file as from
    the reference file.

    Args:
        config_file: A file of a config/<directory>/, e.g. config/controller/joint_pd_gains.textproto.

    Returns:
        The absolute path of its configuration's task file, which need not exist.
    """
    return _sibling(config_file, _TASK_FROM_REFERENCE)


def contact_planning_file(task_file: str) -> str | None:
    """The contact planner's file next to a task file (config/mpc/contact_planning.textproto), or None without one."""
    if not task_file:
        return None
    candidate = os.path.join(
        os.path.dirname(os.path.abspath(task_file)), CONTACT_PLANNING_FILE_NAME
    )
    return candidate if os.path.isfile(candidate) else None


def read_default_pelvis_height(
    reference_file: str, fallback: float = DEFAULT_PELVIS_HEIGHT
) -> float:
    """The reference file's default_base_height [m], or `fallback` when it has none or cannot be read."""
    if not reference_file or not os.path.exists(reference_file):
        return fallback
    try:
        reference = nproto_textproto.load_textproto(
            reference_file, reference_file_pb2.ReferenceFile
        )
    except (OSError, nproto_textproto.TextprotoError) as error:
        _LOGGER.warning(
            "Could not read default_base_height from %s: %s", reference_file, error
        )
        return fallback
    if not reference.HasField("default_base_height"):
        return fallback
    return float(reference.default_base_height)


@dataclasses.dataclass(frozen=True)
class OnlineTuning:
    """Whether the tuning tabs publish and save, and why not when they do not.

    Attributes:
      enabled: Whether they do.
      reason: Why not, for the tabs' banner; empty when they do.
    """

    enabled: bool
    reason: str = ""


def read_online_tuning_state(task_file: str) -> OnlineTuning:
    """The task file's enable_online_tuning, with the reason when it is off.

    A file that says false turns tuning off; one that leaves the field out, or no file at all, leaves it on. A file that
    does not parse turns it off too, with its error (file:line:column), so that a file the operator wrote "tuning off"
    into, in a spelling the strict parser refuses, is not tuned.

    Args:
      task_file: The task file; empty for none.

    Returns:
      The state.
    """
    if not task_file or not os.path.exists(task_file):
        return OnlineTuning(enabled=True)
    try:
        task = nproto_textproto.load_textproto(task_file, task_file_pb2.TaskFile)
    except (OSError, nproto_textproto.TextprotoError) as error:
        _LOGGER.error("Online tuning is off: the task file does not parse: %s", error)
        return OnlineTuning(
            enabled=False, reason=f"the task file does not parse: {error}"
        )
    if not task.enable_online_tuning:
        return OnlineTuning(
            enabled=False, reason="enable_online_tuning: false in the task file"
        )
    return OnlineTuning(enabled=True)


def read_default_joint_state(reference_file: str | None) -> dict[str, float]:
    """The reference file's default_joint_state, joint name -> position [rad]; empty when it has none or cannot be read."""
    if not reference_file or not os.path.exists(reference_file):
        return {}
    try:
        reference = nproto_textproto.load_textproto(
            reference_file, reference_file_pb2.ReferenceFile
        )
    except (OSError, nproto_textproto.TextprotoError) as error:
        _LOGGER.warning(
            "Could not read default_joint_state from %s: %s", reference_file, error
        )
        return {}
    return {
        str(entry.joint): float(entry.value) for entry in reference.default_joint_state
    }


def read_pd_gains_joint_names(pd_gains_file: str | None) -> list[str]:
    """The joints the joint PD gains file gives gains of, in its order; empty when it cannot be read."""
    if not pd_gains_file or not os.path.exists(pd_gains_file):
        return []
    try:
        gains = nproto_textproto.load_textproto(
            pd_gains_file, joint_pd_gains_file_pb2.JointPdGainsFile
        )
    except (OSError, nproto_textproto.TextprotoError) as error:
        _LOGGER.warning("Could not read the joints of %s: %s", pd_gains_file, error)
        return []
    return [str(entry.joint) for entry in gains.joint_gains]


def load_registries(
    registries_file: str | None = None, repo_root: str | None = None
) -> dict[str, tuple[str, ...]]:
    """The names each registry accepts, by registry name, from config_registries.textproto.

    Without them the GUI still renders a registry string, as a choice the operator types into, which the receiving
    registry checks; so a file that is missing or does not parse is logged and gives no registries.

    Args:
        registries_file: The file; None for REGISTRIES_FILE in the checkout `repo_root`.
        repo_root: The checkout; None to find it with find_repo_root().

    Returns:
        Registry name -> its names, in the registry's order; empty when the file cannot be read.
    """
    if registries_file is None:
        root = repo_root or find_repo_root()
        if not root:
            _LOGGER.warning("No checkout to read %s from", REGISTRIES_FILE)
            return {}
        registries_file = os.path.join(root, REGISTRIES_FILE)
    try:
        registries = nproto_textproto.load_textproto(
            registries_file, config_registries_pb2.ConfigRegistries
        )
    except (OSError, nproto_textproto.TextprotoError) as error:
        _LOGGER.warning(
            "Could not read the registries' names from %s: %s", registries_file, error
        )
        return {}
    return {
        str(registry.name): tuple(str(name) for name in registry.names)
        for registry in registries.registries
    }
