# Copyright (c) 2026, Nicholas Palomo. All rights reserved.
# Copyright (c) 2024, 1X Technologies. All rights reserved.
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

"""The Xbox controller as a source of walking commands, read through pygame.

Free of the bus: the GUI (base_velocity_controller_gui.py) and xbox_walking_command_publisher.py publish what it
returns. The stick shaping, the two axis layouts and the pelvis height integration are plain functions over an object
with get_axis(), and pygame is behind a small backend, so test/test_xbox_controller.py covers all of it with a fake
joystick. pygame is not thread-safe: call everything here from one thread (the GUI calls it from Tk's).
"""

import dataclasses
import logging
from typing import Any, Protocol

from humanoid_mpc_msgs import walking_velocity_command_pb2

from remote_control import operator_bus

_LOGGER = logging.getLogger(__name__)

# A stick deflection below this, after shaping, reads as zero.
STICK_DEADBAND = 0.02
# A controller whose name contains this is connected over Bluetooth, which numbers its axes differently.
BLUETOOTH_NAME_MARKER = "Wireless Controller"


@dataclasses.dataclass
class ControllerInput:
    """The sticks, shaped, in [-1, 1] (x forward, y left), and the triggers in [0, 1] (0: released)."""

    x_left: float = 0.0
    y_left: float = 0.0
    x_right: float = 0.0
    y_right: float = 0.0
    lt: float = 0.0
    rt: float = 0.0


@dataclasses.dataclass(frozen=True)
class AxisLayout:
    """The pygame axis of each input. The sticks are read negated, so that forward and left are positive."""

    x_left: int
    y_left: int
    x_right: int
    y_right: int
    lt: int
    rt: int


USB_LAYOUT = AxisLayout(x_left=1, y_left=0, x_right=4, y_right=3, lt=2, rt=5)
BLUETOOTH_LAYOUT = AxisLayout(x_left=1, y_left=0, x_right=3, y_right=2, lt=6, rt=5)


class Joystick(Protocol):
    """What this module reads of a pygame joystick."""

    def get_name(self) -> str: ...

    def get_axis(self, axis: int) -> float: ...


def shape_stick(raw: float) -> float:
    """A stick deflection in [-1, 1], softened around the center (0.2 x + 0.8 x^3), with the deadband applied."""
    shaped = 0.2 * raw + 0.8 * raw**3
    return shaped if abs(shaped) >= STICK_DEADBAND else 0.0


def _trigger(raw: float) -> float:
    """A trigger axis, -1 released to 1 pressed, as 0 to 1."""
    return (raw + 1.0) / 2.0


def read_controller_input(joystick: Joystick, layout: AxisLayout) -> ControllerInput:
    """The inputs of `joystick` read in `layout`."""
    return ControllerInput(
        x_left=shape_stick(-joystick.get_axis(layout.x_left)),
        y_left=shape_stick(-joystick.get_axis(layout.y_left)),
        x_right=shape_stick(-joystick.get_axis(layout.x_right)),
        y_right=shape_stick(-joystick.get_axis(layout.y_right)),
        lt=_trigger(joystick.get_axis(layout.lt)),
        rt=_trigger(joystick.get_axis(layout.rt)),
    )


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


class JoystickBackend(Protocol):
    """The joystick API this module needs; PygameJoystickBackend is the real one."""

    def init(self) -> None: ...

    def open_first_joystick(self) -> Joystick | None: ...

    def joystick_count(self) -> int: ...

    def pump(self) -> None: ...


class PygameJoystickBackend:
    """pygame's joysticks. pygame is imported here rather than at module scope, so importing this module is cheap."""

    def __init__(self) -> None:
        import pygame  # pylint: disable=import-outside-toplevel  # Keeps importing this module cheap.

        self._pygame = pygame

    def init(self) -> None:
        self._pygame.init()

    def open_first_joystick(self) -> Joystick | None:
        self._pygame.joystick.quit()
        self._pygame.joystick.init()
        if self._pygame.joystick.get_count() <= 0:
            return None
        joystick = self._pygame.joystick.Joystick(0)
        joystick.init()
        return joystick

    def joystick_count(self) -> int:
        return self._pygame.joystick.get_count()

    def pump(self) -> None:
        self._pygame.event.pump()


class XBoxControllerInterface:
    """Walking commands from the first connected controller.

    The left stick commands the linear velocity, the right stick's horizontal axis the yaw rate, and the triggers move
    the pelvis height target (RT up, LT down) at up to 1 m/s.

    Args:
        publisher_rate: how often get_walking_command_msg() is called [Hz]; it scales the pelvis height rate.
        backend: the joysticks (default: pygame's).
    """

    def __init__(
        self, publisher_rate: float, backend: JoystickBackend | None = None
    ) -> None:
        self._backend = backend if backend is not None else PygameJoystickBackend()
        self._backend.init()
        self.publisher_rate = publisher_rate
        self.current_pelvis_height_target = 0.8
        self.min_pelvis_height = 0.2
        self.max_pelvis_height = 1.0
        self.joystick: Joystick | None = None
        self.joystick_connected = False
        self.bluetooth_connection = False
        self.get_joystick_connection()

    def get_joystick_connection(self) -> bool:
        """Rescans for a controller; returns whether one is connected."""
        joystick = self._backend.open_first_joystick()
        if joystick is None:
            self.joystick = None
            self.joystick_connected = False
            return False
        name = joystick.get_name()
        self.joystick = joystick
        self.bluetooth_connection = BLUETOOTH_NAME_MARKER in name
        self.joystick_connected = True
        _LOGGER.info(
            "Connected to %s via %s.",
            name,
            "Bluetooth" if self.bluetooth_connection else "USB",
        )
        return True

    def get_joystick_inputs(self) -> ControllerInput:
        """The controller's inputs now.

        Raises:
            ConnectionError: the controller is gone.
        """
        if self._backend.joystick_count() < 1 or self.joystick is None:
            raise ConnectionError("the controller is disconnected")
        self._backend.pump()
        layout = BLUETOOTH_LAYOUT if self.bluetooth_connection else USB_LAYOUT
        return read_controller_input(self.joystick, layout)

    def get_walking_command_msg(
        self,
    ) -> tuple[bool, walking_velocity_command_pb2.WalkingVelocityCommand | None]:
        """(True, the command of the controller's inputs), or (False, None) when no controller can be read.

        A controller that cannot be read is marked disconnected; GamepadPoller then scans for it again.

        Returns:
            Whether a controller was read, and the walking command of its inputs (None when none was read).
        """
        if not self.joystick_connected:
            return False, None
        try:
            controller_input = self.get_joystick_inputs()
        # pylint: disable-next=broad-exception-caught  # Pygame raises its own error types.
        except Exception as error:
            _LOGGER.warning(
                "Lost the controller (%s); scanning for it in the background.", error
            )
            self.joystick_connected = False
            return False, None
        # The triggers move the target by up to 1 m/s at the publisher's rate.
        pelvis_height_rate = controller_input.rt - controller_input.lt
        self.current_pelvis_height_target = clamp(
            self.current_pelvis_height_target
            + pelvis_height_rate / self.publisher_rate,
            self.min_pelvis_height,
            self.max_pelvis_height,
        )
        return True, operator_bus.walking_velocity_command(
            linear_velocity_x=controller_input.x_left,
            linear_velocity_y=controller_input.y_left,
            angular_velocity_z=controller_input.y_right,
            desired_pelvis_height=self.current_pelvis_height_target,
        )


class GamepadPoller:
    """Reads one walking command per tick from a controller, and scans for one every `scan_period` while none is.

    Args:
        controller: the controller interface.
        rate_hz: how often tick() is called [Hz].
        scan_period: how often to scan for a controller while none is connected [s]. A scan takes about a millisecond,
            and it must run on the thread that reads the controller: pygame deadlocks against SDL's own thread when
            pygame.joystick is reinitialized from another one.
    """

    def __init__(
        self, controller: Any, rate_hz: float, scan_period: float = 2.0
    ) -> None:
        # The not-form rejects NaN, which `rate_hz <= 0.0` would let through.
        if not (rate_hz > 0.0 and scan_period > 0.0):
            raise ValueError(
                f"the rate and the scan period must be positive, got {rate_hz} and {scan_period}"
            )
        self._controller = controller
        self._scan_ticks = max(1, int(round(scan_period * rate_hz)))
        self._ticks_since_scan = 0

    @property
    def connected(self) -> bool:
        return bool(self._controller.joystick_connected)

    def tick(self) -> walking_velocity_command_pb2.WalkingVelocityCommand | None:
        """The controller's command, or None when there is no controller (or it was just lost)."""
        if self._controller.joystick_connected:
            self._ticks_since_scan = 0
            success, command = self._controller.get_walking_command_msg()
            return command if success else None
        self._ticks_since_scan += 1
        if self._ticks_since_scan >= self._scan_ticks:
            self._ticks_since_scan = 0
            self._controller.get_joystick_connection()
        return None
