"""****************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

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

"""Publishes the Xbox controller's walking commands on the bus, without the GUI.

    bazel run //humanoid_nmpc/remote_control:xbox_velocity_publisher -- [--network_config=...] [--ipc_node=teleop]

Publishes operator/walking_velocity_command as the `teleop` node at the topic's 25 Hz while a controller is connected,
and scans for one every two seconds while none is. Nothing is published without a controller, so that a GUI started
next to it stays in charge of the robot.
"""

import argparse
import logging
import signal
import sys
import threading
from typing import List, Optional

from remote_control import teleop
from remote_control.operator_bus import TELEOP_NODE, TopicPublisher
from remote_control.xbox_controller_interface import (
    GamepadPoller,
    XBoxControllerInterface,
)

_LOGGER = logging.getLogger(__name__)


class XBoxWalkingCommandPublisher:
    """One tick: the controller's command, published when there is one.

    Args:
        publisher: the walking command publisher.
        poller: the controller (GamepadPoller).
    """

    def __init__(self, publisher: TopicPublisher, poller: GamepadPoller) -> None:
        self._publisher = publisher
        self._poller = poller

    def tick(self) -> None:
        command = self._poller.tick()
        if command is not None:
            self._publisher.publish(command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publishes the Xbox controller's walking commands on the IPC bus."
    )
    teleop.add_bus_flags(parser, default_node=TELEOP_NODE)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
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
        controller = XBoxControllerInterface(teleop.WALKING_COMMAND_RATE_HZ)
        node = XBoxWalkingCommandPublisher(
            publisher, GamepadPoller(controller, teleop.WALKING_COMMAND_RATE_HZ)
        )
        teleop.run_periodically(teleop.WALKING_COMMAND_RATE_HZ, node.tick, stop)
    finally:
        publisher.bus.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
