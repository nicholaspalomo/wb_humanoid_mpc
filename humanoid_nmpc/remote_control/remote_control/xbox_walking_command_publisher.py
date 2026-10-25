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

"""Publishes the Xbox controller's walking commands on the bus, without the GUI.

    bazel run //humanoid_nmpc/remote_control:xbox_velocity_publisher -- --reference_file=<reference.textproto> \\
        [--network_config=...] [--ipc_node=teleop]

Publishes operator/walking_velocity_command as the `teleop` node at the topic's 25 Hz while a controller is connected,
and scans for one every two seconds while none is. The pelvis height target starts at the reference file's
default_base_height. The MPC keeps the last command it received, so when the controller goes away, and when the
publisher ends, it publishes the stop command (zero velocities, the pelvis height kept) for a second; then nothing until
a controller is found again.

Do not run it next to a GUI that publishes walking commands (base_velocity_controller_gui --walking_commands=publish,
the launch files' default): both publish the same topic, and the MPC alternates between the two. Start the GUI with
--walking_commands=off, or untick its "Publish walking commands", to keep its FSM and tuning tabs. A GUI started next
to it is in charge only while no controller is connected here.
"""

import argparse
import logging
import signal
import threading

from humanoid_mpc_msgs import walking_velocity_command_pb2

from remote_control import operator_bus
from remote_control import teleop
from remote_control import xbox_controller_interface

_LOGGER = logging.getLogger(__name__)


class XBoxWalkingCommandPublisher:
    """One tick: the controller's command, published when there is one, and the stop command after it went away.

    Args:
        publisher: the walking command publisher.
        poller: the controller (GamepadPoller).
        stop_ticks: how many ticks the stop command is published after the controller went away.
    """

    def __init__(
        self,
        publisher: operator_bus.TopicPublisher,
        poller: xbox_controller_interface.GamepadPoller,
        stop_ticks: int = teleop.STOP_COMMAND_TICKS,
    ) -> None:
        if stop_ticks < 1:
            raise ValueError(f"stop_ticks must be positive, got {stop_ticks}")
        self._publisher = publisher
        self._poller = poller
        self._stop_ticks = stop_ticks
        # The last command of the controller, None before the first; the stop command keeps its pelvis height.
        self._last_command: (
            walking_velocity_command_pb2.WalkingVelocityCommand | None
        ) = None
        # Ticks of the stop command still to publish; stop_ticks while the controller is read.
        self._stop_ticks_left = 0

    def tick(self) -> None:
        command = self._poller.tick()
        if command is not None:
            self._last_command = command
            self._stop_ticks_left = self._stop_ticks
        elif self._last_command is not None and self._stop_ticks_left > 0:
            self._stop_ticks_left -= 1
            command = teleop.stop_command(
                self._last_command.desired_pelvis_height
            )
        if command is not None:
            self._publisher.publish(command)

    def stop_pelvis_height(self) -> float | None:
        """The pelvis height of the stop command to send when the publisher ends [m].

        Returns:
            The last command's pelvis height while this publisher commands the robot (a controller is read, or the stop
            command is still being published); None once it has handed the topic back, or before any command.
        """
        if self._last_command is None or self._stop_ticks_left == 0:
            return None
        return self._last_command.desired_pelvis_height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Publishes the Xbox controller's walking commands on the IPC bus."
        )
    )
    teleop.add_bus_flags(parser, default_node=operator_bus.TELEOP_NODE)
    teleop.add_reference_file_flag(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args(argv)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda signum, frame: stop.set())
    signal.signal(signal.SIGTERM, lambda signum, frame: stop.set())

    publisher = teleop.connect_walking_command_publisher(
        args.network_config, args.ipc_node
    )
    try:
        controller = xbox_controller_interface.XBoxControllerInterface(
            teleop.WALKING_COMMAND_RATE_HZ,
            initial_pelvis_height=teleop.initial_pelvis_height(
                args.reference_file
            ),
        )
    except BaseException:
        publisher.bus.close()
        raise
    node = XBoxWalkingCommandPublisher(
        publisher,
        xbox_controller_interface.GamepadPoller(
            controller, teleop.WALKING_COMMAND_RATE_HZ
        ),
    )
    teleop.publish_until_stopped(
        publisher, node.tick, stop, node.stop_pelvis_height
    )
    return 0
