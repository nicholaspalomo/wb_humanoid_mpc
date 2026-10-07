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

"""What the teleoperation publishers share: the walking command's rate, their node on the bus, and the timed loop.

The keyboard and Xbox publishers (keyboard_walking_command_publisher.py, xbox_walking_command_publisher.py) publish
operator/walking_velocity_command as the `teleop` node of the network file, at the rate of the README's topic table.
The GUI publishes the same topic as the `operator` node.
"""

import argparse
from collections.abc import Callable
import threading
import time

from humanoid_mpc_ipc import topics
from remote_control import config_files
from remote_control import operator_bus
import robot_ipc

# The rate of operator/walking_velocity_command (README "Topics"). The MPC takes the latest command, so a faster rate
# only costs bandwidth, and a slower one delays the stick.
# LINT.IfChange(walking_command_rate)
WALKING_COMMAND_RATE_HZ = 25.0
# LINT.ThenChange(//humanoid_nmpc/docs/distributed_runtime/README.md:topic_table)


def add_bus_flags(parser: argparse.ArgumentParser, default_node: str) -> None:
    """Adds --network_config and --ipc_node, the bus flags of every process (README "Configuration")."""
    parser.add_argument(
        "--network_config",
        default=operator_bus.DEFAULT_NETWORK_CONFIG,
        help="the network file of the bus (default: %(default)s, found from the repository root as well)",
    )
    parser.add_argument(
        "--ipc_node",
        default=default_node,
        help="the node of the network file this process publishes as (default: %(default)s)",
    )


def connect_walking_command_publisher(
    network_config: str, node_name: str = operator_bus.TELEOP_NODE
) -> operator_bus.TopicPublisher:
    """A started bus publishing as `node_name`, and the walking command publisher on it.

    The bus is the publisher's `bus`; close it when done.

    Args:
        network_config: the network file, as the command line gave it (config_files.resolve_input_path).
        node_name: the node of the network file to publish as.

    Returns:
        The publisher of operator/walking_velocity_command on the started bus.
    """
    path = config_files.resolve_input_path(network_config)
    bus = robot_ipc.Bus(node_name, robot_ipc.load_network_config(path))
    bus.start()
    return operator_bus.TopicPublisher.for_topic(
        bus, topics.OPERATOR_WALKING_VELOCITY_COMMAND
    )


def run_periodically(
    rate_hz: float,
    tick: Callable[[], None],
    stop: threading.Event,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], object] | None = None,
    max_ticks: int | None = None,
) -> int:
    """Calls `tick` at `rate_hz` on absolute deadlines until `stop` is set; returns the number of ticks.

    A tick that overruns its period does not make the next ones late: the missed deadlines are skipped, so the loop
    never bursts to catch up (a burst of identical walking commands carries nothing new).

    Args:
        rate_hz: the rate, positive.
        tick: called once per period, on this thread.
        stop: ends the loop at the next deadline once set.
        clock: monotonic seconds; the tests pass a fake one with `sleep`.
        sleep: waits for the given number of seconds (default: stop.wait, so that setting `stop` ends the wait).
        max_ticks: stop after this many ticks (None: never).

    Returns:
        The number of times `tick` was called.
    """
    # The not-form rejects NaN, which `rate_hz <= 0.0` would let through to a NaN period.
    if not rate_hz > 0.0:
        raise ValueError(f"the rate must be positive, got {rate_hz}")
    period = 1.0 / rate_hz
    wait = stop.wait if sleep is None else sleep
    ticks = 0
    deadline = clock()
    while not stop.is_set() and (max_ticks is None or ticks < max_ticks):
        tick()
        ticks += 1
        deadline += period
        now = clock()
        if deadline < now:
            # Overran: the next deadline is the first one still ahead.
            missed = int((now - deadline) / period) + 1
            deadline += missed * period
        wait(max(0.0, deadline - now))
    return ticks
