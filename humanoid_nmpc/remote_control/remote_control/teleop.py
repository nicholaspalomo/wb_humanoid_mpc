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

The MPC keeps the last walking command it received, so a source of commands that goes away must leave a stop command
(zero velocities, the pelvis height kept) behind: publish_until_stopped() sends one when its loop ends, and the Xbox
publisher and the GUI send one when their controller goes away.
"""

import argparse
from collections.abc import Callable
import logging
import threading
import time

from humanoid_mpc_msgs import walking_velocity_command_pb2

from humanoid_mpc_ipc import topics
from remote_control import config_files
from remote_control import operator_bus
import robot_ipc

_LOGGER = logging.getLogger(__name__)

# The rate of operator/walking_velocity_command (README "Topics"). The MPC takes the latest command, so a faster rate
# only costs bandwidth, and a slower one delays the stick.
# LINT.IfChange(walking_command_rate)
WALKING_COMMAND_RATE_HZ = 25.0
# LINT.ThenChange(//humanoid_nmpc/docs/distributed_runtime/README.md:topic_table, //humanoid_nmpc/humanoid_mpc_config/interface_config.proto:walking_command_timeout)

# How long a publisher whose controller went away keeps publishing the stop command [s], and that many ticks at the
# topic's rate: long enough that a dropped message cannot leave the robot walking, short enough to hand the topic back.
STOP_COMMAND_DURATION = 1.0
STOP_COMMAND_TICKS = max(
    1, round(STOP_COMMAND_DURATION * WALKING_COMMAND_RATE_HZ)
)
# How long close_after_stop() waits for the bus to hand its stop command to its DDS writer [s], and then before it closes
# the bus [s]: operator/walking_velocity_command is BEST_EFFORT ("latest"), so nothing repairs a datagram that was still
# on its way out when the writer was deleted, and the grace period lets the network carry it first.
STOP_COMMAND_SEND_TIMEOUT = 1.0
STOP_COMMAND_GRACE_PERIOD = 0.1
# How often close_after_stop() looks at the bus's counter [s].
_SENT_POLL_PERIOD = 0.005


def add_bus_flags(parser: argparse.ArgumentParser, default_node: str) -> None:
    """Adds --network_config and --ipc_node, the bus flags of every process (README "Configuration")."""
    parser.add_argument(
        "--network_config",
        default=operator_bus.DEFAULT_NETWORK_CONFIG,
        help=(
            "the network file of the bus (default: %(default)s, found from the"
            " repository root as well)"
        ),
    )
    parser.add_argument(
        "--ipc_node",
        default=default_node,
        help=(
            "the node of the network file this process publishes as (default:"
            " %(default)s)"
        ),
    )


def add_reference_file_flag(parser: argparse.ArgumentParser) -> None:
    """Adds --reference_file, the robot's reference file, whose nom_pos_base_rt_world_ewrt_world_z the pelvis height starts from."""
    parser.add_argument(
        "--reference_file",
        default="",
        help=(
            "the robot's config/command/reference.textproto: the pelvis height"
            " starts at its nom_pos_base_rt_world_ewrt_world_z (default: none,"
            f" {config_files.DEFAULT_PELVIS_HEIGHT} m)"
        ),
    )


def initial_pelvis_height(reference_file: str) -> float:
    """The pelvis height a publisher starts from [m]: the nom_pos_base_rt_world_ewrt_world_z of `reference_file`.

    Args:
        reference_file: the --reference_file flag, as the command line gave it (config_files.resolve_input_path).

    Returns:
        The file's nom_pos_base_rt_world_ewrt_world_z; config_files.DEFAULT_PELVIS_HEIGHT, with a warning, when no file is given or it
        names none.
    """
    if not reference_file:
        _LOGGER.warning(
            "No --reference_file: the pelvis height starts at %.2f m, not at"
            " the robot's nom_pos_base_rt_world_ewrt_world_z.",
            config_files.DEFAULT_PELVIS_HEIGHT,
        )
        return config_files.DEFAULT_PELVIS_HEIGHT
    return config_files.read_default_pelvis_height(
        config_files.resolve_input_path(reference_file)
    )


def stop_command(
    pelvis_height: float,
) -> walking_velocity_command_pb2.WalkingVelocityCommand:
    """The walking command that stops the robot where it stands: zero velocities at `pelvis_height` [m]."""
    return operator_bus.walking_velocity_command(
        cmd_linvel_base_rt_world_ewrt_heading_x=0.0,
        cmd_linvel_base_rt_world_ewrt_heading_y=0.0,
        cmd_angvel_base_rt_world_ewrt_world_z=0.0,
        cmd_pos_base_rt_world_ewrt_world_z=pelvis_height,
    )


def close_after_stop(
    publisher: operator_bus.TopicPublisher,
    pelvis_height: float | None,
    timeout: float = STOP_COMMAND_SEND_TIMEOUT,
    grace_period: float = STOP_COMMAND_GRACE_PERIOD,
) -> bool:
    """Publishes the stop command, waits until the bus has sent it, then closes the publisher's bus.

    The MPC keeps the last walking command it received, so a publisher that exits without a stop command leaves the
    robot walking at its last command. The bus writes from its receive thread, so its `written` counter is waited for
    (the command reached the DDS writer) and then `grace_period` more, before close() deletes the writer.

    Args:
        publisher: the walking command publisher; its bus is closed in every case.
        pelvis_height: the stop command's pelvis height [m]; None sends none (a publisher that is not commanding the
            robot, such as the Xbox publisher without a controller).
        timeout: how long to wait for the bus to send it [s].
        grace_period: how long to wait after that before the bus is closed [s].

    Returns:
        Whether the bus counted the stop command as sent; False when none was asked for or it was not sent in time.
    """
    bus = publisher.bus
    try:
        if pelvis_height is None:
            return False
        before = bus.topic_statistics(publisher.topic).written
        if not publisher.publish(stop_command(pelvis_height)):
            _LOGGER.warning(
                "The stop command was dropped: the bus's queue is full."
            )
            return False
        deadline = time.monotonic() + timeout
        while bus.topic_statistics(publisher.topic).written <= before:
            if time.monotonic() >= deadline:
                _LOGGER.warning(
                    "The stop command was not sent within %.1f s.", timeout
                )
                return False
            time.sleep(_SENT_POLL_PERIOD)
        time.sleep(grace_period)
        return True
    finally:
        bus.close()


def publish_until_stopped(
    publisher: operator_bus.TopicPublisher,
    tick: Callable[[], None],
    stop: threading.Event,
    stop_pelvis_height: Callable[[], float | None],
    max_ticks: int | None = None,
) -> int:
    """Runs a publisher's `tick` at the walking command's rate until `stop` is set, then sends the stop command and closes.

    Args:
        publisher: the walking command publisher; its bus is closed when the loop ends, also by an exception.
        tick: one period of the publisher.
        stop: ends the loop once set.
        stop_pelvis_height: the pelvis height of the stop command sent at the end [m], read then; None sends none.
        max_ticks: stop after this many ticks (None: never), for the tests.

    Returns:
        The number of ticks run.
    """
    ticks = 0
    try:
        ticks = run_periodically(
            WALKING_COMMAND_RATE_HZ, tick, stop, max_ticks=max_ticks
        )
    finally:
        close_after_stop(publisher, stop_pelvis_height())
    return ticks


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
