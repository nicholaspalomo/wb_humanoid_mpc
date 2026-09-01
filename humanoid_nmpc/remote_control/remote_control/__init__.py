"""****************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

from .dashboard_backend import (
    SimProcessManager,
    VirtualJoystickROS2,
    ensure_ros2_paths,
)
from .humanoid_finite_state_machine import (
    ControlMode,
    HumanoidFSM,
    VirtualGantry,
    MODE_METADATA,
    CYCLE_MODES,
    load_robot_config,
    get_available_robots,
)

ensure_ros2_paths()

try:
    from .xbox_controller_interface import XBoxControllerInterface
    from . import xbox_walking_command_publisher
except ImportError:
    XBoxControllerInterface = None
    xbox_walking_command_publisher = None

__all__ = [
    "SimProcessManager",
    "VirtualJoystickROS2",
    "ensure_ros2_paths",
    "XBoxControllerInterface",
    "xbox_walking_command_publisher",
    "ControlMode",
    "HumanoidFSM",
    "VirtualGantry",
    "MODE_METADATA",
    "CYCLE_MODES",
    "load_robot_config",
    "get_available_robots",
]
