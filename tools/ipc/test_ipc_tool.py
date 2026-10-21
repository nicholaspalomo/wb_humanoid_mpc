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

"""End-to-end tests of the ipc_tool subcommands against publishers of raw messages on the test's own DDS domain.

Each test writes the network file of the test process's domain and runs ipc_tool.main() on it, as the command line
would. The publishers write the messages repeatedly, so that the tool has discovered them whenever it joins.
"""

import contextlib
import io
import json
import math
import os
import random
import re
import tempfile
import unittest

from google.protobuf import text_format
from humanoid_mpc_msgs import mpc_status_pb2

import bus_listener
import bus_test_publisher
import ipc_tool

STATUS_TYPE = "humanoid_mpc_msgs.MpcStatus"
# A passing test returns as soon as it has its messages; the timeout only bounds a failing one.
TIMEOUT_S = "10"


def make_status() -> mpc_status_pb2.MpcStatus:
    status = mpc_status_pb2.MpcStatus(observation_time=2.5, resets_served=4)
    status.solver_status.healthy = True
    return status


def documents(text: str) -> list[str]:
    """The messages `echo` printed, each followed by a `---` line."""
    parts = text.split("\n---\n")
    return [part for part in parts if part.strip()]


class IpcToolTestCase(unittest.TestCase):
    """A temporary directory for the network file, and ipc_tool.main() run on captured output."""

    def setUp(self) -> None:
        # pylint: disable-next=consider-using-with  # The cleanup deletes it.
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)

    def network_file(self) -> str:
        return bus_test_publisher.write_network_file(self._directory.name)

    def run_tool(self, *argv: str) -> tuple[int, str, str]:
        out = io.StringIO()
        err = io.StringIO()
        code = ipc_tool.main(list(argv), out=out, err=err)
        return code, out.getvalue(), err.getvalue()


class EchoTest(IpcToolTestCase):

    def test_prints_count_messages_as_textprotos_of_every_field(self) -> None:
        topic = bus_test_publisher.unique_topic("status")
        status = make_status()
        with bus_test_publisher.MessagePublisher() as publisher:
            publisher.start(
                [
                    bus_test_publisher.message(
                        topic, STATUS_TYPE, status.SerializeToString()
                    )
                ]
            )
            network = self.network_file()
            code, out, err = self.run_tool(
                "echo",
                topic,
                "--count",
                "2",
                "--timeout",
                TIMEOUT_S,
                "--network_config",
                network,
            )
        self.assertEqual(code, ipc_tool.EXIT_OK, err)
        printed = documents(out)
        self.assertEqual(len(printed), 2)
        for document in printed:
            # Each message parses back strictly into the message that was sent.
            value = text_format.Parse(document, mpc_status_pb2.MpcStatus())
            self.assertEqual(value.observation_time, 2.5)
            self.assertEqual(value.resets_served, 4)
            self.assertTrue(value.solver_status.healthy)
            # Fields at their default are printed too.
            self.assertIn("observations_skipped: 0", document.splitlines())

    def test_fields_and_json_select_and_format_the_output(self) -> None:
        topic = bus_test_publisher.unique_topic("status")
        with bus_test_publisher.MessagePublisher() as publisher:
            publisher.start(
                [
                    bus_test_publisher.message(
                        topic, STATUS_TYPE, make_status().SerializeToString()
                    )
                ]
            )
            network = self.network_file()
            code, out, err = self.run_tool(
                "echo",
                topic,
                "--count",
                "1",
                "--timeout",
                TIMEOUT_S,
                "--fields",
                "solver_status.healthy,observation_time",
                "--format",
                "json",
                "--network_config",
                network,
            )
        self.assertEqual(code, ipc_tool.EXIT_OK, err)
        self.assertEqual(
            json.loads(documents(out)[0]),
            {"solver_status.healthy": True, "observation_time": 2.5},
        )

    def test_a_bad_field_path_is_a_usage_error(self) -> None:
        topic = bus_test_publisher.unique_topic("status")
        with bus_test_publisher.MessagePublisher() as publisher:
            publisher.start(
                [
                    bus_test_publisher.message(
                        topic, STATUS_TYPE, make_status().SerializeToString()
                    )
                ]
            )
            network = self.network_file()
            code, _, err = self.run_tool(
                "echo",
                topic,
                "--count",
                "1",
                "--timeout",
                TIMEOUT_S,
                "--fields",
                "solver_status.healty",
                "--network_config",
                network,
            )
        self.assertEqual(code, ipc_tool.EXIT_USAGE)
        self.assertIn("has no field 'healty'", err)

    def test_an_unknown_type_is_described_instead_of_decoded(self) -> None:
        topic = bus_test_publisher.unique_topic("unknown")
        with bus_test_publisher.MessagePublisher() as publisher:
            publisher.start(
                [bus_test_publisher.message(topic, "test.NotAMessage", b"\x01\x02\x03")]
            )
            network = self.network_file()
            code, out, err = self.run_tool(
                "echo",
                topic,
                "--count",
                "1",
                "--timeout",
                TIMEOUT_S,
                "--network_config",
                network,
            )
        self.assertEqual(code, ipc_tool.EXIT_OK, err)
        self.assertIn("test.NotAMessage: unknown message type, 3 bytes", out)

    def test_no_message_before_the_timeout_exits_with_status_one(self) -> None:
        network = self.network_file()
        code, out, err = self.run_tool(
            "echo",
            bus_test_publisher.unique_topic("silent"),
            "--count",
            "1",
            "--timeout",
            "0.3",
            "--network_config",
            network,
        )
        self.assertEqual(code, ipc_tool.EXIT_NO_MESSAGES)
        self.assertEqual(out, "")
        self.assertIn("received 0 message(s)", err)


class _BacklogListener:
    """Stands in for a BusListener that fell behind: one message from receive(), four more already waiting."""

    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.malformed_messages = 0
        status = make_status()
        self._messages = []
        for sequence in range(5):
            status.observation_time = float(sequence)
            self._messages.append(
                bus_listener.BusMessage(
                    topic, STATUS_TYPE, status.SerializeToString(), float(sequence)
                )
            )

    def receive(self, timeout_s: float) -> bus_listener.BusMessage | None:
        del timeout_s  # Unused.
        return self._messages[0]

    def receive_pending(self) -> list[bus_listener.BusMessage]:
        return self._messages[1:]


class EchoBacklogTest(unittest.TestCase):
    def test_echo_prints_the_newest_waiting_message_and_says_how_many_it_skipped(
        self,
    ) -> None:
        out = io.StringIO()
        err = io.StringIO()
        code = ipc_tool.run_echo(
            _BacklogListener("mpc/status"),  # type: ignore[arg-type]  # A stand-in with the methods run_echo calls.
            count=1,
            fields=[],
            output_format="text",
            timeout_s=1.0,
            out=out,
            err=err,
        )
        self.assertEqual(code, ipc_tool.EXIT_OK, err.getvalue())
        (printed,) = documents(out.getvalue())
        self.assertEqual(
            text_format.Parse(printed, mpc_status_pb2.MpcStatus()).observation_time,
            4.0,
        )
        self.assertIn("skipped 4 message(s)", err.getvalue())


class ListTest(IpcToolTestCase):

    def test_lists_the_topics_of_every_node_with_their_type(self) -> None:
        robot_topic = bus_test_publisher.unique_topic("robot")
        mpc_topic = bus_test_publisher.unique_topic("mpc")
        payload = make_status().SerializeToString()
        with bus_test_publisher.MessagePublisher(
            "robot"
        ) as robot, bus_test_publisher.MessagePublisher("mpc") as mpc:
            robot.start(
                [
                    bus_test_publisher.message(
                        robot_topic, "humanoid_mpc_msgs.LoopTiming", b""
                    )
                ]
            )
            mpc.start([bus_test_publisher.message(mpc_topic, STATUS_TYPE, payload)])
            network = self.network_file()
            code, out, err = self.run_tool(
                "list", "--duration", "1.0", "--network_config", network
            )
        self.assertEqual(code, ipc_tool.EXIT_OK, err)
        lines = {line.split()[0]: line.split() for line in out.splitlines()[1:]}
        self.assertEqual(lines[robot_topic][1], "humanoid_mpc_msgs.LoopTiming")
        self.assertEqual(lines[mpc_topic][1], STATUS_TYPE)

    def test_a_silent_bus_exits_with_status_one(self) -> None:
        network = self.network_file()
        code, out, err = self.run_tool(
            "list", "--duration", "0.2", "--network_config", network
        )
        self.assertEqual(code, ipc_tool.EXIT_NO_MESSAGES)
        self.assertEqual(out, "")
        self.assertIn("no messages within 0.2 s", err)


class HzTest(IpcToolTestCase):

    def test_reports_a_rate_and_the_payload_size(self) -> None:
        topic = bus_test_publisher.unique_topic("observation")
        payload = make_status().SerializeToString()
        with bus_test_publisher.MessagePublisher() as publisher:
            publisher.start(
                [bus_test_publisher.message(topic, STATUS_TYPE, payload)],
                period_s=0.005,
            )
            network = self.network_file()
            code, out, err = self.run_tool(
                "hz",
                topic,
                "--duration",
                "1.0",
                "--report_period",
                "0.4",
                "--network_config",
                network,
            )
        self.assertEqual(code, ipc_tool.EXIT_OK, err)
        rates = [float(rate) for rate in re.findall(r"average rate ([0-9.]+) Hz", out)]
        self.assertTrue(rates, out)
        self.assertTrue(all(rate > 0.0 for rate in rates), out)
        # Every message has the same payload, so the statistics of its size are exact.
        self.assertIn(f"size mean {len(payload)} B", out)
        self.assertIn(f"max {len(payload)} B", out)

    def test_fewer_than_two_messages_exit_with_status_one(self) -> None:
        network = self.network_file()
        code, out, err = self.run_tool(
            "hz",
            bus_test_publisher.unique_topic("silent"),
            "--duration",
            "0.3",
            "--report_period",
            "0.1",
            "--network_config",
            network,
        )
        self.assertEqual(code, ipc_tool.EXIT_NO_MESSAGES)
        self.assertIn("no new messages", out)
        self.assertIn("a rate needs two", err)


class NextReportTimeTest(unittest.TestCase):
    def test_is_the_first_grid_time_after_now(self) -> None:
        rng = random.Random(3)
        for trial in range(10000):
            scheduled = 100.0 * rng.random()
            period = 1e-3 + rng.random()
            now = (
                scheduled + (-1.0 if trial % 4 == 0 else 300.0) * period * rng.random()
            )
            following = ipc_tool.next_report_time(scheduled, period, now)
            self.assertGreater(following, now, msg=trial)
            if scheduled > now:
                self.assertEqual(following, scheduled, msg=trial)
            else:
                self.assertLessEqual(following - period, now + 1e-12, msg=trial)
                # On the grid: a whole number of periods from `scheduled`.
                periods = (following - scheduled) / period
                self.assertAlmostEqual(periods, round(periods), delta=1e-9, msg=trial)

    def test_one_period_ahead_is_the_sum(self) -> None:
        rng = random.Random(4)
        for _ in range(10000):
            scheduled = 100.0 * rng.random()
            period = 1e-3 + rng.random()
            now = scheduled + 0.999 * period * rng.random()
            self.assertEqual(
                ipc_tool.next_report_time(scheduled, period, now), scheduled + period
            )

    def test_skips_any_number_of_periods_at_once(self) -> None:
        period = 0.5
        now = 1e12 * period + 0.2
        following = ipc_tool.next_report_time(0.0, period, now)
        self.assertGreater(following, now)
        self.assertLessEqual(following - period, now)
        self.assertTrue(math.isfinite(following))


class ConfigurationTest(IpcToolTestCase):

    def test_a_missing_network_file_is_a_usage_error(self) -> None:
        code, _, err = self.run_tool(
            "list",
            "--network_config",
            os.path.join(self._directory.name, "absent.textproto"),
        )
        self.assertEqual(code, ipc_tool.EXIT_USAGE)
        self.assertIn("cannot read the network file", err)

    def test_the_default_network_file_is_the_one_of_the_bus(self) -> None:
        args = ipc_tool.build_parser().parse_args(["list"])
        self.assertEqual(
            args.network_config, os.path.join("config", "ipc", "network.textproto")
        )

    def test_durations_and_periods_reject_nan_and_out_of_range_values(self) -> None:
        # The validators are written `not value >= 0.0`, which rejects NaN, where `value < 0.0` would let it through.
        for argv in (
            ["echo", "a", "--timeout", "nan"],
            ["echo", "a", "--timeout", "-1"],
            ["hz", "a", "--duration", "nan"],
            ["list", "--duration", "nan"],
            ["list", "--duration", "0"],
            ["hz", "a", "--report_period", "nan"],
            ["hz", "a", "--report_period", "-0.5"],
        ):
            with self.subTest(argv=argv):
                err = io.StringIO()
                with contextlib.redirect_stderr(err), self.assertRaises(
                    SystemExit
                ) as raised:
                    ipc_tool.build_parser().parse_args(argv)
                self.assertEqual(raised.exception.code, ipc_tool.EXIT_USAGE)
                self.assertIn(f"got {argv[-1]}", err.getvalue())

    def test_durations_and_periods_accept_their_boundaries(self) -> None:
        args = ipc_tool.build_parser().parse_args(["echo", "a", "--timeout", "0"])
        self.assertEqual(args.timeout, 0.0)
        args = ipc_tool.build_parser().parse_args(
            ["hz", "a", "--duration", "0", "--report_period", "0.25"]
        )
        self.assertEqual((args.duration, args.report_period), (0.0, 0.25))

    def test_every_subcommand_takes_the_network_file(self) -> None:
        for argv in (["list"], ["echo", "a"], ["hz", "a"]):
            with self.subTest(command=argv[0]):
                args = ipc_tool.build_parser().parse_args(
                    argv + ["--network_config", "other.textproto"]
                )
                self.assertEqual(args.network_config, "other.textproto")


if __name__ == "__main__":
    unittest.main()
