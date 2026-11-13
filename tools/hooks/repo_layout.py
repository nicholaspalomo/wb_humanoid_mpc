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

"""The repository paths that the checks of tools/hooks scope by, in one place.

Every check that reads only part of the tree (the base orientation's checks, `frame-naming`, `config-yaml`,
`magic-number`'s test-only sources, lint_files.py's generated files) builds its directories and files from these
constants, so a moved directory is one change here. Directories end in `/`; package names carry no slash.
test_repo_layout.py checks that every path here, every path literal of the other modules of tools/hooks, CI's target
patterns and the clang-tidy header filter name something that exists.

This module imports nothing from tools/hooks, so every check module can import it.
"""

# The first-party top-level directories.
NMPC_DIR = "legged_nmpc/"
STATE_ESTIMATION_DIR = "legged_state_estimation/"
LEARNING_DIR = "legged_learning/"
ROBOT_MODELS_DIR = "robot_models/"
ROBOT_RUNTIME_DIR = "robot_runtime/"
# This project's fork of OCS2 (lib/ is vendored, lib/ocs2 is partly first-party: lint_files.py).
LIB_OCS2_DIR = "lib/ocs2/"

# The NMPC packages the checks name: the package name, which is also its include directory below include/, and the
# package directory.
COMMON_MPC_NAME = "legged_common_mpc"
COMMON_MPC_PACKAGE = NMPC_DIR + COMMON_MPC_NAME + "/"
COMMON_MPC_INCLUDE_DIR = COMMON_MPC_PACKAGE + "include/" + COMMON_MPC_NAME + "/"
WB_MPC_NAME = "legged_wb_mpc"
WB_MPC_PACKAGE = NMPC_DIR + WB_MPC_NAME + "/"
WB_MPC_INCLUDE_DIR = WB_MPC_PACKAGE + "include/" + WB_MPC_NAME + "/"
VALIDATION_NAME = "legged_mpc_validation"
VALIDATION_PACKAGE = NMPC_DIR + VALIDATION_NAME + "/"
VALIDATION_INCLUDE_DIR = VALIDATION_PACKAGE + "include/" + VALIDATION_NAME + "/"

# The design documents the checks enforce.
FRAME_NAMING_README = NMPC_DIR + "docs/frame_naming/README.md"
QUATERNION_README = NMPC_DIR + "docs/quaternion_base_orientation/README.md"


def all_scoped_paths() -> tuple[str, ...]:
    """Returns every directory (ending in `/`) and file path of this module, relative to the repository root."""
    return (
        NMPC_DIR,
        STATE_ESTIMATION_DIR,
        LEARNING_DIR,
        ROBOT_MODELS_DIR,
        ROBOT_RUNTIME_DIR,
        LIB_OCS2_DIR,
        COMMON_MPC_PACKAGE,
        COMMON_MPC_INCLUDE_DIR,
        WB_MPC_PACKAGE,
        WB_MPC_INCLUDE_DIR,
        VALIDATION_PACKAGE,
        VALIDATION_INCLUDE_DIR,
        FRAME_NAMING_README,
        QUATERNION_README,
    )
