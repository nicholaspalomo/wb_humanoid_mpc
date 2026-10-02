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

"""remote_control: the operator GUI and the teleoperation publishers of the humanoid, on the IPC bus.

See README.md. The GUI is base_velocity_controller_gui.py; operator_bus.py is its side of the bus.
"""

from .dashboard_backend import SimProcessManager, VirtualJoystick
from .humanoid_finite_state_machine import (
    ControlMode,
    HumanoidFSM,
    VirtualGantry,
    MODE_METADATA,
    CYCLE_MODES,
    load_robot_config,
    get_available_robots,
)


# Lazy access for the Xbox controller, so that importing the package does not initialize pygame (in Jupyter, say).
def __getattr__(name):
    if name == "XBoxControllerInterface":
        from .xbox_controller_interface import XBoxControllerInterface

        return XBoxControllerInterface
    elif name == "xbox_walking_command_publisher":
        from . import xbox_walking_command_publisher

        return xbox_walking_command_publisher
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "SimProcessManager",
    "VirtualJoystick",
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
