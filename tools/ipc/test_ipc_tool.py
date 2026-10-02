"""End-to-end tests of the ipc_tool subcommands against raw pyzmq publishers on loopback.

Each test writes a network file naming its publishers' ephemeral ports and runs ipc_tool.main() on it, as the command
line would. The publishers frame messages as the bus README specifies, without any bus library.
"""

import io
import json
import os
import re
import tempfile
import unittest
from typing import Dict, List, Tuple

from google.protobuf import text_format

import bus_test_publisher
import ipc_tool
from humanoid_mpc_msgs import mpc_status_pb2

STATUS_TYPE = "humanoid_mpc_msgs.MpcStatus"
# A passing test returns as soon as it has its messages; the timeout only bounds a failing one.
TIMEOUT_S = "10"


def make_status() -> mpc_status_pb2.MpcStatus:
    status = mpc_status_pb2.MpcStatus(observation_time=2.5, resets_served=4)
    status.solver_status.healthy = True
    return status


def documents(text: str) -> List[str]:
    """The messages `echo` printed, each followed by a `---` line."""
    parts = text.split("\n---\n")
    return [part for part in parts if part.strip()]


class IpcToolTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)

    def network_file(self, ports: Dict[str, int]) -> str:
        return bus_test_publisher.write_network_file(self._directory.name, ports)

    def run_tool(self, *argv: str) -> Tuple[int, str, str]:
        out = io.StringIO()
        err = io.StringIO()
        code = ipc_tool.main(list(argv), out=out, err=err)
        return code, out.getvalue(), err.getvalue()


class EchoTest(IpcToolTestCase):

    def test_prints_count_messages_as_textprotos_of_every_field(self) -> None:
        topic = bus_test_publisher.unique_topic("status")
        status = make_status()
        with bus_test_publisher.FramePublisher() as publisher:
            publisher.start(
                [
                    bus_test_publisher.frames(
                        topic, STATUS_TYPE, status.SerializeToString()
                    )
                ]
            )
            network = self.network_file({"mpc": publisher.port})
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
        with bus_test_publisher.FramePublisher() as publisher:
            publisher.start(
                [
                    bus_test_publisher.frames(
                        topic, STATUS_TYPE, make_status().SerializeToString()
                    )
                ]
            )
            network = self.network_file({"mpc": publisher.port})
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
        with bus_test_publisher.FramePublisher() as publisher:
            publisher.start(
                [
                    bus_test_publisher.frames(
                        topic, STATUS_TYPE, make_status().SerializeToString()
                    )
                ]
            )
            network = self.network_file({"mpc": publisher.port})
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
        with bus_test_publisher.FramePublisher() as publisher:
            publisher.start(
                [bus_test_publisher.frames(topic, "test.NotAMessage", b"\x01\x02\x03")]
            )
            network = self.network_file({"mpc": publisher.port})
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
        network = self.network_file({"mpc": bus_test_publisher.unused_port()})
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


class ListTest(IpcToolTestCase):

    def test_lists_the_topics_of_every_node_with_their_type(self) -> None:
        robot_topic = bus_test_publisher.unique_topic("robot")
        mpc_topic = bus_test_publisher.unique_topic("mpc")
        payload = make_status().SerializeToString()
        with bus_test_publisher.FramePublisher() as robot, bus_test_publisher.FramePublisher() as mpc:
            robot.start(
                [
                    bus_test_publisher.frames(
                        robot_topic, "humanoid_mpc_msgs.LoopTiming", b""
                    )
                ]
            )
            mpc.start([bus_test_publisher.frames(mpc_topic, STATUS_TYPE, payload)])
            network = self.network_file({"robot": robot.port, "mpc": mpc.port})
            code, out, err = self.run_tool(
                "list", "--duration", "1.0", "--network_config", network
            )
        self.assertEqual(code, ipc_tool.EXIT_OK, err)
        lines = {line.split()[0]: line.split() for line in out.splitlines()[1:]}
        self.assertEqual(lines[robot_topic][1], "humanoid_mpc_msgs.LoopTiming")
        self.assertEqual(lines[mpc_topic][1], STATUS_TYPE)

    def test_a_silent_bus_exits_with_status_one(self) -> None:
        network = self.network_file({"robot": bus_test_publisher.unused_port()})
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
        with bus_test_publisher.FramePublisher() as publisher:
            publisher.start(
                [bus_test_publisher.frames(topic, STATUS_TYPE, payload)], period_s=0.005
            )
            network = self.network_file({"robot": publisher.port})
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
        network = self.network_file({"robot": bus_test_publisher.unused_port()})
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

    def test_every_subcommand_takes_the_network_file(self) -> None:
        for argv in (["list"], ["echo", "a"], ["hz", "a"]):
            with self.subTest(command=argv[0]):
                args = ipc_tool.build_parser().parse_args(
                    argv + ["--network_config", "other.textproto"]
                )
                self.assertEqual(args.network_config, "other.textproto")


if __name__ == "__main__":
    unittest.main()
