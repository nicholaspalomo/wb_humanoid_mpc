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

"""Publishes walking commands from the keyboard of the terminal it runs in.

    bazel run //humanoid_nmpc/remote_control:keyboard_velocity_publisher -- [--network_config=...] [--ipc_node=teleop]

Hold a key to command: the arrow keys walk (up/down forward and back, left/right sideways), `a` and `d` turn, `w` and
`s` raise and lower the pelvis height target by 1 cm per tick. Releasing every key stops the robot. Publishes
operator/walking_velocity_command as the `teleop` node at the topic's 25 Hz; Ctrl-C ends it and restores the terminal.

KeyboardCommand and read_key() are free of the terminal and of the bus, so test/test_keyboard_walking_command_publisher.py
covers them headless.
"""

import argparse
from collections.abc import Callable
import os
import select
import signal
import sys
import termios
import threading
import tty
from typing import TextIO

from humanoid_mpc_msgs import walking_velocity_command_pb2

from remote_control import operator_bus
from remote_control import teleop

# The keys read_key() returns for the arrow keys, from the last byte of their escape sequence ESC [ A..D.
KEY_UP = "up"
KEY_DOWN = "down"
KEY_RIGHT = "right"
KEY_LEFT = "left"
_ESCAPE = "\x1b"
_CONTROL_SEQUENCE_INTRODUCER = "["
_ARROW_KEYS = {"A": KEY_UP, "B": KEY_DOWN, "C": KEY_RIGHT, "D": KEY_LEFT}


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def read_key(read_char: Callable[[], str | None]) -> str | None:
    """The next key: an arrow key as KEY_UP..KEY_LEFT, any other key as its character, None when none is waiting.

    Args:
        read_char: the next character, or None (or "") when none is waiting.

    Returns:
        KEY_UP, KEY_DOWN, KEY_RIGHT or KEY_LEFT for an arrow key, the character of any other key, and None when no key
        is waiting or an escape sequence is not an arrow key's.
    """
    char = read_char()
    if not char:
        return None
    if char != _ESCAPE:
        return char
    if read_char() != _CONTROL_SEQUENCE_INTRODUCER:
        return None
    return _ARROW_KEYS.get(read_char() or "")


class KeyboardCommand:
    """The walking command of the key held down.

    Every tick applies the key that arrived (None when none did): each key commands one thing and releases the others,
    and no key releases everything, so that the robot stops when the operator lets go. The pelvis height target is the
    one thing that stays, moving by `pelvis_height_step` per tick of `w` or `s`.
    """

    def __init__(
        self,
        max_vel_x: float = 1.0,
        max_vel_y: float = 1.0,
        max_vel_yaw: float = 1.0,
        pelvis_height: float = 0.8,
        min_pelvis_height: float = 0.2,
        max_pelvis_height: float = 1.0,
        pelvis_height_step: float = 0.01,
    ) -> None:
        self.max_vel_x = max_vel_x
        self.max_vel_y = max_vel_y
        self.max_vel_yaw = max_vel_yaw
        self.min_pelvis_height = min_pelvis_height
        self.max_pelvis_height = max_pelvis_height
        self.pelvis_height_step = pelvis_height_step
        self.pelvis_height = clamp(pelvis_height, min_pelvis_height, max_pelvis_height)
        self.x_vel = 0.0
        self.y_vel = 0.0
        self.yaw_vel = 0.0

    def apply(
        self, key: str | None
    ) -> walking_velocity_command_pb2.WalkingVelocityCommand:
        """Applies one tick's key (None: no key) and returns the command to publish."""
        self.x_vel = 0.0
        self.y_vel = 0.0
        self.yaw_vel = 0.0
        height_change = 0.0
        if key == KEY_UP:
            self.x_vel = self.max_vel_x
        elif key == KEY_DOWN:
            self.x_vel = -self.max_vel_x
        elif key == KEY_RIGHT:
            self.y_vel = -self.max_vel_y
        elif key == KEY_LEFT:
            self.y_vel = self.max_vel_y
        elif key == "a":
            self.yaw_vel = self.max_vel_yaw
        elif key == "d":
            self.yaw_vel = -self.max_vel_yaw
        elif key == "w":
            height_change = self.pelvis_height_step
        elif key == "s":
            height_change = -self.pelvis_height_step
        self.pelvis_height = clamp(
            self.pelvis_height + height_change,
            self.min_pelvis_height,
            self.max_pelvis_height,
        )
        return self.command()

    def command(self) -> walking_velocity_command_pb2.WalkingVelocityCommand:
        return operator_bus.walking_velocity_command(
            linear_velocity_x=self.x_vel,
            linear_velocity_y=self.y_vel,
            angular_velocity_z=self.yaw_vel,
            desired_pelvis_height=self.pelvis_height,
        )

    def status_line(self) -> str:
        return (
            f"Command: x={self.x_vel:+.2f}, y={self.y_vel:+.2f}, yaw={self.yaw_vel:+.2f}, "
            f"pelvis height={self.pelvis_height:.2f} m"
        )


class TerminalInput:
    """The terminal in cbreak mode, read one character at a time without waiting; restored on exit."""

    def __init__(self, stream: TextIO = sys.stdin) -> None:
        self._fd = stream.fileno()
        self._saved_attributes: list | None = None

    def __enter__(self) -> "TerminalInput":
        self._saved_attributes = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._saved_attributes is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved_attributes)
            self._saved_attributes = None

    def read_char(self) -> str | None:
        """The next character typed, or None when none is waiting."""
        ready, _, _ = select.select([self._fd], [], [], 0.0)
        if not ready:
            return None
        data = os.read(self._fd, 1)
        return data.decode("utf-8", errors="replace") if data else None

    def discard_pending(self) -> None:
        """Drops what was typed but not read yet: a held key's auto-repeat must not queue up behind the loop."""
        termios.tcflush(self._fd, termios.TCIFLUSH)


class KeyboardWalkingCommandPublisher:
    """One tick: read a key, publish the command, show it.

    Args:
        publisher: the walking command publisher.
        read_char: the terminal's next character (TerminalInput.read_char).
        discard_pending: drops the characters still waiting (TerminalInput.discard_pending).
        state: the command state.
        output: where the status line goes.
    """

    def __init__(
        self,
        publisher: operator_bus.TopicPublisher,
        read_char: Callable[[], str | None],
        discard_pending: Callable[[], None],
        state: KeyboardCommand | None = None,
        output: TextIO = sys.stdout,
    ) -> None:
        self._publisher = publisher
        self._read_char = read_char
        self._discard_pending = discard_pending
        self.state = state if state is not None else KeyboardCommand()
        self._output = output

    def tick(self) -> None:
        key = read_key(self._read_char)
        if key is not None:
            self._discard_pending()
        self._publisher.publish(self.state.apply(key))
        self._output.write(self.state.status_line() + "\r")
        self._output.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publishes walking commands from the keyboard of this terminal on the IPC bus."
    )
    teleop.add_bus_flags(parser, default_node=operator_bus.TELEOP_NODE)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not sys.stdin.isatty():
        print(
            "keyboard_velocity_publisher reads the keyboard of a terminal; stdin is not one.",
            file=sys.stderr,
        )
        return 2
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda signum, frame: stop.set())
    signal.signal(signal.SIGTERM, lambda signum, frame: stop.set())

    publisher = teleop.connect_walking_command_publisher(
        args.network_config, args.ipc_node
    )
    try:
        with TerminalInput() as terminal:
            node = KeyboardWalkingCommandPublisher(
                publisher, terminal.read_char, terminal.discard_pending
            )
            teleop.run_periodically(teleop.WALKING_COMMAND_RATE_HZ, node.tick, stop)
    finally:
        publisher.bus.close()
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
