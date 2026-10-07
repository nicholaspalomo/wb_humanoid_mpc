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

"""push_robot_config: sends a configuration file to the robot's store, as the GUI's Save does, and prints the answer.

    bazel run //humanoid_nmpc/remote_control:push_robot_config -- \\
        robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto

It delivers an editor's save of a file the robot process reads - the task file, the reference file or the joint PD
gains file, told apart by the file's `# proto-message:` header - to a running robot, which otherwise reads it at its
next start, and it is the scripted check of the two-process simulation. It publishes as the config_push node of the
network file, since a running GUI holds the operator node.

The first messages of a freshly bound socket can be lost before the robot's subscriber has connected to it, so it sends
one save, under one sequence, every RESEND_PERIOD_SECONDS until the robot answers that sequence or
ANSWER_TIMEOUT_SECONDS pass. The robot answers a repeat with the status it recorded, so a repeat is never stored twice
and a lost answer is not turned into "unchanged".

Exit status: 0 (EXIT_IN_SYNC) when the robot's copy is the file: it stored it, already had it, or has no store and reads
this very file in place; 1 (EXIT_NOT_SAVED) when it refused it, could not store it, or did not answer; 2
(EXIT_CANNOT_SEND) when the file, the network file or the bus does not let it send; 3 (EXIT_NOT_STORED) when the robot
has no store and reads another file in place, which the push did not change.
"""

import argparse
from collections.abc import Callable, Sequence
import sys
import time

from humanoid_mpc_ipc import topics
import nproto_textproto
from remote_control import config_files
from remote_control import operator_bus
from remote_control import robot_config_save
from remote_control import teleop
import robot_ipc

# How often the save is sent again while the robot has not answered it [s].
RESEND_PERIOD_SECONDS = 0.5
# How long it waits for the answer [s].
ANSWER_TIMEOUT_SECONDS = 10.0

# The exit statuses (the module docstring).
EXIT_IN_SYNC = 0
EXIT_NOT_SAVED = 1
EXIT_CANNOT_SEND = 2
EXIT_NOT_STORED = 3


class PushError(ValueError):
    """A file this tool cannot send: unreadable, of no kind the robot stores, not parsing, or naming no robot."""


def push(
    saver: robot_config_save.RobotConfigSaver,
    kind: int,
    path: str,
    text: str,
    robot_name: str,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], object] = time.sleep,
) -> robot_config_save.SaveOutcome:
    """Sends one save until the robot answers it or ANSWER_TIMEOUT_SECONDS pass, repeating it unchanged.

    Args:
      saver: The saver, on a started bus, whose timeout is the time to wait for the answer.
      kind: The file's kind.
      path: The file.
      text: Its content.
      robot_name: The robot it is for.
      clock: Monotonic seconds, the saver's clock; the tests pass a fake one with `sleep`.
      sleep: Waits for the given number of seconds.

    Returns:
      The answer; TIMED_OUT without one, NOT_SENT when the bus never took the save.
    """
    sequence = saver.send(kind, path, text, robot_name)
    deadline = clock() + saver.timeout
    while True:
        sleep(RESEND_PERIOD_SECONDS)
        saver.poll(clock())
        outcome = saver.outcome(sequence)
        if outcome is None:
            raise RuntimeError(f"the saver forgot the save {sequence} it is sending")
        answered = (
            not outcome.waiting
            and outcome.state != robot_config_save.SaveState.NOT_SENT
        )
        if (
            answered
            or outcome.state == robot_config_save.SaveState.TIMED_OUT
            or clock() >= deadline
        ):
            return outcome
        saver.resend(sequence)


def read_file(path: str) -> tuple[robot_config_save.FileKind, str]:
    """The kind and the text of a configuration file, which must parse strictly into its schema.

    Args:
      path: The file.

    Returns:
      Its kind, from its `# proto-message:` header, and its text, as the file holds it.

    Raises:
      PushError: The file cannot be read, names no schema the robot stores, or does not parse.
    """
    try:
        with open(path, encoding="utf-8", newline="") as file:
            text = file.read()
    except OSError as error:
        raise PushError(f"cannot read {path}: {error}") from error
    header = nproto_textproto.header_message(text)
    entry = (
        None if header is None else robot_config_save.file_kind_of_message(header[1])
    )
    if entry is None:
        stored = ", ".join(
            kind.message_class.DESCRIPTOR.full_name
            for kind in robot_config_save.FILE_KINDS
        )
        raise PushError(
            f"{path} names no file the robot stores in its '# proto-message:' header (one of: {stored})"
        )
    try:
        nproto_textproto.parse_textproto(text, entry.message_class(), path)
    except nproto_textproto.TextprotoError as error:
        raise PushError(str(error)) from error
    return entry, text


def describe(outcome: robot_config_save.SaveOutcome, timeout: float) -> str:
    """What the robot did with the file, for the terminal.

    Args:
      outcome: The answer (push()).
      timeout: How long it was waited for [s].

    Returns:
      One line.
    """
    state = outcome.state
    if state == robot_config_save.SaveState.SAVED:
        return f"stored on the robot at {outcome.stored_path}"
    if state == robot_config_save.SaveState.UNCHANGED:
        return f"unchanged: the robot's copy at {outcome.stored_path} is the same"
    if state == robot_config_save.SaveState.NOT_STORED:
        if outcome.reads_laptop_file:
            return f"the robot has no store and reads this file in place ({outcome.stored_path})"
        read = outcome.stored_path or "its own copy"
        return f"the robot has no store: it reads {read} in place, which this push did not change"
    if state == robot_config_save.SaveState.REFUSED:
        return f"the robot refused it: {outcome.message}"
    if state == robot_config_save.SaveState.FAILED:
        return f"the robot could not store it: {outcome.message}"
    if state == robot_config_save.SaveState.NOT_SENT:
        return f"not sent: {outcome.message}"
    return f"the robot did not answer in {timeout:g} s: its copy is unknown"


def build_parser() -> argparse.ArgumentParser:
    """The tool's flags: the file and the bus flags (teleop.add_bus_flags)."""
    parser = argparse.ArgumentParser(
        description="Sends a configuration file to the robot's store on the IPC bus and prints the robot's answer."
    )
    parser.add_argument(
        "file",
        help="the task, reference or joint PD gains textproto (relative: to the directory the tool was started from, "
        "then to the checkout)",
    )
    teleop.add_bus_flags(parser, default_node=operator_bus.CONFIG_PUSH_NODE)
    return parser


def _connect(
    network_config: str, node_name: str
) -> tuple[robot_ipc.Bus, robot_config_save.RobotConfigSaver]:
    """A started bus publishing as `node_name`, and a saver on it that waits ANSWER_TIMEOUT_SECONDS for an answer."""
    network = robot_ipc.load_network_config(network_config)
    bus = robot_ipc.Bus(node_name, network)
    statuses = operator_bus.ConfigSaveStatusMailbox()
    status = operator_bus.operator_topic(topics.ROBOT_CONFIG_SAVE_STATUS)
    bus.subscribe(
        status.topic, status.message_class, statuses.put, delivery=status.delivery
    )
    publisher = operator_bus.TopicPublisher.for_topic(bus, topics.OPERATOR_CONFIG_SAVE)
    bus.start()
    return bus, robot_config_save.RobotConfigSaver(
        publisher, statuses, timeout=ANSWER_TIMEOUT_SECONDS
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Sends the file the command line names and prints the robot's answer.

    Args:
      argv: The arguments; None: the command line's.

    Returns:
      The exit status (the module docstring).
    """
    args = build_parser().parse_args(argv)
    repo_root = config_files.find_repo_root()
    path = config_files.resolve_input_path(args.file, repo_root)
    try:
        entry, text = read_file(path)
        robot_name = robot_config_save.robot_name_of(entry.kind, path)
    except (PushError, robot_config_save.RobotNameError) as error:
        print(f"push_robot_config: {error}", file=sys.stderr)
        return EXIT_CANNOT_SEND
    network_config = config_files.resolve_input_path(args.network_config, repo_root)
    try:
        bus, saver = _connect(network_config, args.ipc_node)
    except (
        OSError,
        robot_ipc.NetworkConfigError,
        robot_ipc.BusError,
        ValueError,
    ) as error:
        print(f"push_robot_config: {error}", file=sys.stderr)
        return EXIT_CANNOT_SEND
    try:
        print(
            f"push_robot_config: sending the {entry.name} {robot_config_save.config_path_of(path)} "
            f"for robot '{robot_name}'"
        )
        outcome = push(saver, entry.kind, path, text, robot_name)
    finally:
        bus.close()
    print(f"push_robot_config: {describe(outcome, saver.timeout)}")
    return exit_status(outcome)


def exit_status(outcome: robot_config_save.SaveOutcome) -> int:
    """The exit status of a push that ended in `outcome` (the module docstring)."""
    if outcome.state == robot_config_save.SaveState.NOT_STORED:
        return EXIT_NOT_STORED if outcome.problem else EXIT_IN_SYNC
    return EXIT_NOT_SAVED if outcome.problem else EXIT_IN_SYNC


if __name__ == "__main__":
    sys.exit(main())
