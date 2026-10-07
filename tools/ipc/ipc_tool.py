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

"""Inspects the IPC bus from the command line: its topics, their messages and their rates.

    bazel run //tools/ipc:ipc_tool -- list                          # topics seen within 2 s, with type and rate
    bazel run //tools/ipc:ipc_tool -- echo mpc/status --count 1     # decode and print messages as textprotos
    bazel run //tools/ipc:ipc_tool -- echo mpc/status --fields solver_status.healthy,observation_time
    bazel run //tools/ipc:ipc_tool -- hz robot/mpc_observation      # rate, period jitter, payload size

Every subcommand takes --network_config (default config/ipc/network.textproto, relative to the directory the tool was
started from or the repository root). The tool only receives: it connects one SUB socket to every node of the network
file (humanoid_nmpc/docs/distributed_runtime/README.md), so it sees every topic whichever process publishes it, and
decodes payloads by the type name of their second frame with the default descriptor pool, into which it imports every
humanoid_mpc_msgs and humanoid_mpc_config module.

Exit status: 0 on success, 1 when the expected messages did not arrive, 2 for a usage or configuration error.
"""

import argparse
from collections.abc import Sequence
import math
import sys
import time
from typing import TextIO

from google.protobuf import message

import bus_listener
import message_decoding
import topic_statistics

EXIT_OK = 0
EXIT_NO_MESSAGES = 1
EXIT_USAGE = 2

DEFAULT_LIST_DURATION_S = 2.0
DEFAULT_HZ_WINDOW = 1000
DEFAULT_REPORT_PERIOD_S = 1.0
# The longest a receive waits before a loop checks its deadlines again.
POLL_PERIOD_S = 0.1


def _remaining(deadline: float | None) -> float:
    return math.inf if deadline is None else deadline - time.monotonic()


def _receive_until(
    listener: bus_listener.BusListener, deadline: float | None
) -> bus_listener.BusMessage | None:
    return listener.receive(max(min(POLL_PERIOD_S, _remaining(deadline)), 0.0))


def _warn_malformed(listener: bus_listener.BusListener, err: TextIO) -> None:
    if listener.malformed_messages:
        err.write(
            f"ipc_tool: dropped {listener.malformed_messages} message(s) that were not "
            f"[topic][type name][payload]\n"
        )


def run_list(
    listener: bus_listener.BusListener, duration_s: float, out: TextIO, err: TextIO
) -> int:
    """Listens to every topic for `duration_s` seconds and prints the topics seen, with type and rate."""
    census = topic_statistics.TopicCensus()
    deadline = time.monotonic() + duration_s
    try:
        while _remaining(deadline) > 0.0:
            received = _receive_until(listener, deadline)
            if received is not None:
                census.add(received.topic, received.type_name, received.receive_time)
    except KeyboardInterrupt:
        pass
    _warn_malformed(listener, err)
    summaries = census.summaries()
    if not summaries:
        err.write(
            f"ipc_tool: no messages within {duration_s:g} s from {', '.join(listener.endpoints)}\n"
        )
        return EXIT_NO_MESSAGES
    out.write(topic_statistics.format_census(summaries) + "\n")
    out.flush()
    return EXIT_OK


def render_message(
    received: bus_listener.BusMessage, fields: Sequence[str], output_format: str
) -> str:
    """One received message as `echo` prints it; a payload that does not decode is described instead.

    Args:
      received: The message as it arrived.
      fields: Dotted field paths to print instead of the whole message (message_decoding.select_field).
      output_format: A name of message_decoding.OUTPUT_FORMATS.

    Returns:
      The printed message, or a `#` comment naming its type and size when it does not decode.

    Raises:
      message_decoding.FieldPathError: A path of `fields` is not a field of the message.
    """
    try:
        decoded = message_decoding.decode(received.type_name, received.payload)
    except message_decoding.UnknownMessageTypeError:
        return f"# {received.type_name}: unknown message type, {len(received.payload)} bytes"
    except message.DecodeError:
        return f"# {received.type_name}: payload does not parse, {len(received.payload)} bytes"
    return message_decoding.format_fields(decoded, fields, output_format)


def run_echo(
    listener: bus_listener.BusListener,
    count: int,
    fields: Sequence[str],
    output_format: str,
    timeout_s: float,
    out: TextIO,
    err: TextIO,
) -> int:
    """Prints the messages of the listener's topic, each followed by a `---` line.

    Args:
      listener: The listener of the topic.
      count: The number of messages to print; 0 prints until interrupted.
      fields: Dotted field paths to print instead of the whole message.
      output_format: A name of message_decoding.OUTPUT_FORMATS.
      timeout_s: How long to wait for the messages, in seconds; 0 waits without a limit.
      out: Where the messages are printed.
      err: Where errors and warnings are printed.

    Returns:
      EXIT_OK; EXIT_NO_MESSAGES when fewer than `count` messages, or none at all without a count, arrived; EXIT_USAGE
      when a path of `fields` is not a field of the message.
    """
    deadline = None if timeout_s <= 0.0 else time.monotonic() + timeout_s
    printed = 0
    try:
        while (count <= 0 or printed < count) and _remaining(deadline) > 0.0:
            received = _receive_until(listener, deadline)
            if received is None:
                continue
            try:
                text = render_message(received, fields, output_format)
            except message_decoding.FieldPathError as error:
                err.write(f"ipc_tool: {error}\n")
                return EXIT_USAGE
            out.write(text + "\n---\n")
            out.flush()
            printed += 1
    except KeyboardInterrupt:
        pass
    _warn_malformed(listener, err)
    if printed == 0 or printed < count:
        err.write(
            f"ipc_tool: received {printed} message(s) on {listener.topic}"
            + (f", expected {count}\n" if count > 0 else "\n")
        )
        return EXIT_NO_MESSAGES
    return EXIT_OK


def run_hz(
    listener: bus_listener.BusListener,
    window: int,
    report_period_s: float,
    duration_s: float,
    out: TextIO,
    err: TextIO,
) -> int:
    """Prints the rate, period jitter and payload size of the listener's topic every `report_period_s` seconds.

    Args:
      listener: The listener of the topic.
      window: The number of most recent messages the statistics are computed over.
      report_period_s: The time between two reports, in seconds.
      duration_s: How long to run, in seconds; 0 runs until interrupted.
      out: Where the reports are printed.
      err: Where errors and warnings are printed.

    Returns:
      EXIT_OK; EXIT_NO_MESSAGES when fewer than two messages arrived, since no rate follows from one.
    """
    topic = listener.topic or ""
    statistics = topic_statistics.RateStatistics(window)
    start = time.monotonic()
    deadline = None if duration_s <= 0.0 else start + duration_s
    next_report = start + report_period_s
    reported_count = 0

    def report() -> None:
        nonlocal reported_count
        snapshot = statistics.snapshot()
        if statistics.total_count == reported_count:
            out.write(f"{topic}: no new messages\n")
        elif snapshot is None:
            out.write(f"{topic}: waiting for a second message\n")
        else:
            out.write(topic_statistics.format_snapshot(topic, snapshot) + "\n")
        out.flush()
        reported_count = statistics.total_count

    try:
        while _remaining(deadline) > 0.0:
            wait = max(
                min(
                    POLL_PERIOD_S, _remaining(deadline), next_report - time.monotonic()
                ),
                0.0,
            )
            received = listener.receive(wait)
            if received is not None:
                statistics.add(received.receive_time, len(received.payload))
            if time.monotonic() >= next_report:
                report()
                # On schedule, without a burst of reports to catch up after a slow one.
                while next_report <= time.monotonic():
                    next_report += report_period_s
    except KeyboardInterrupt:
        pass
    # The messages since the last periodic report, so that a short --duration still prints a rate.
    if statistics.total_count != reported_count:
        report()
    _warn_malformed(listener, err)
    if statistics.total_count < 2:
        err.write(
            f"ipc_tool: received {statistics.total_count} message(s) on {topic}; a rate needs two\n"
        )
        return EXIT_NO_MESSAGES
    return EXIT_OK


def _window_size(text: str) -> int:
    value = int(text)
    if value < 2:
        raise argparse.ArgumentTypeError(f"must be at least 2, got {value}")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be non-negative, got {value}")
    return value


def _non_negative_float(text: str) -> float:
    """An argparse type: a float that is not negative; NaN is rejected too (AGENTS.md, Python)."""
    value = float(text)
    if not value >= 0.0:
        raise argparse.ArgumentTypeError(f"must be non-negative, got {text}")
    return value


def _positive_float(text: str) -> float:
    """An argparse type: a float greater than zero; NaN is rejected too (AGENTS.md, Python)."""
    value = float(text)
    if not value > 0.0:
        raise argparse.ArgumentTypeError(f"must be positive, got {text}")
    return value


def _field_paths(text: str) -> list[str]:
    return [path.strip() for path in text.split(",") if path.strip()]


def build_parser() -> argparse.ArgumentParser:
    """The command line: the subcommands list, echo and hz, each with --network_config."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--network_config",
        default=bus_listener.DEFAULT_NETWORK_CONFIG,
        help="the network file naming the endpoints of the bus (default: %(default)s)",
    )

    parser = argparse.ArgumentParser(
        prog="ipc_tool",
        description="Inspect the topics of the IPC bus.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    list_parser = subcommands.add_parser(
        "list", parents=[common], help="the topics seen on the bus, with type and rate"
    )
    list_parser.add_argument(
        "--duration",
        type=_positive_float,
        default=DEFAULT_LIST_DURATION_S,
        help="seconds to listen (default: %(default)s); slower topics may be missed",
    )

    echo_parser = subcommands.add_parser(
        "echo", parents=[common], help="decode and print the messages of a topic"
    )
    echo_parser.add_argument("topic")
    echo_parser.add_argument(
        "--count",
        type=_non_negative_int,
        default=0,
        help="stop after this many messages (default: 0, until interrupted)",
    )
    echo_parser.add_argument(
        "--fields",
        type=_field_paths,
        default=[],
        help="comma-separated dotted field paths to print instead of the whole message, e.g. "
        "solver_status.healthy,state_trajectory.0.data",
    )
    echo_parser.add_argument(
        "--format",
        dest="output_format",
        choices=sorted(message_decoding.OUTPUT_FORMATS),
        default="text",
        help="text (protobuf text format) or json (default: %(default)s)",
    )
    echo_parser.add_argument(
        "--timeout",
        type=_non_negative_float,
        default=0.0,
        help="give up after this many seconds (default: 0, no limit)",
    )

    hz_parser = subcommands.add_parser(
        "hz", parents=[common], help="rate, period jitter and payload size of a topic"
    )
    hz_parser.add_argument("topic")
    hz_parser.add_argument(
        "--window",
        type=_window_size,
        default=DEFAULT_HZ_WINDOW,
        help="messages the statistics are computed over (default: %(default)s)",
    )
    hz_parser.add_argument(
        "--report_period",
        type=_positive_float,
        default=DEFAULT_REPORT_PERIOD_S,
        help="seconds between reports (default: %(default)s)",
    )
    hz_parser.add_argument(
        "--duration",
        type=_non_negative_float,
        default=0.0,
        help="seconds to run (default: 0, until interrupted)",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    out: TextIO = sys.stdout,
    err: TextIO = sys.stderr,
) -> int:
    """Runs one subcommand of the command line `argv` and returns the exit status."""
    args = build_parser().parse_args(argv)
    topic: str | None = getattr(args, "topic", None)
    if topic is not None and not topic:
        err.write("ipc_tool: the topic must not be empty\n")
        return EXIT_USAGE
    try:
        nodes = bus_listener.load_network_config(
            bus_listener.resolve_input_path(args.network_config)
        )
    except bus_listener.NetworkConfigError as error:
        err.write(f"ipc_tool: {error}\n")
        return EXIT_USAGE

    if args.command == "echo":
        message_decoding.import_message_modules()
    with bus_listener.BusListener.from_nodes(nodes, topic=topic) as listener:
        if args.command == "list":
            return run_list(listener, args.duration, out, err)
        if args.command == "echo":
            return run_echo(
                listener,
                args.count,
                args.fields,
                args.output_format,
                args.timeout,
                out,
                err,
            )
        return run_hz(
            listener, args.window, args.report_period, args.duration, out, err
        )


if __name__ == "__main__":
    sys.exit(main())
