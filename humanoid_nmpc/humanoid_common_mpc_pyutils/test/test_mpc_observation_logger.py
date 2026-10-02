"""The MPC observation logger writes one CSV row per observation of the bus, in order, under generic column names, and
export_rollouts.py reads what it writes."""

import argparse
import csv
import datetime
import io
import os
import socket
import tempfile
import threading
import time
import unittest
from typing import Callable, List

import numpy as np
import robot_ipc
from humanoid_common_mpc_pyutils import export_rollouts
from humanoid_common_mpc_pyutils import mpc_observation_logger
from humanoid_mpc_ipc import topics
from humanoid_mpc_msgs import mpc_observation_pb2

TIMEOUT_S = 20.0
ROBOT = "robot"


def observation(
    time_s: float, state: List[float], inputs: List[float], mode: int = 3
) -> mpc_observation_pb2.MpcObservation:
    message = mpc_observation_pb2.MpcObservation(sequence=int(abs(time_s) * 1000))
    message.observation.time = time_s
    message.observation.mode = mode
    message.observation.state.extend(state)
    message.observation.input.extend(inputs)
    return message


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def wait_for(condition: Callable[[], bool]) -> bool:
    deadline = time.monotonic() + TIMEOUT_S
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return condition()


class ColumnsTest(unittest.TestCase):
    def test_the_header_names_time_mode_then_every_state_and_input_component(
        self,
    ) -> None:
        self.assertEqual(
            mpc_observation_logger.column_names(state_dim=3, input_dim=2),
            ["time", "mode", "x0", "x1", "x2", "u0", "u1"],
        )

    def test_a_row_is_time_mode_state_input(self) -> None:
        row = mpc_observation_logger.observation_row(
            observation(1.5, [0.1, 0.2], [7.0], mode=15)
        )
        self.assertEqual(row, [1.5, 15.0, 0.1, 0.2, 7.0])

    def test_the_file_name_is_the_one_export_rollouts_looks_for(self) -> None:
        name = mpc_observation_logger.log_file_name(
            datetime.datetime(2026, 10, 1, 9, 30, 5)
        )
        self.assertEqual(name, "mpc_observation_20261001_093005.csv")
        with tempfile.TemporaryDirectory() as directory:
            open(os.path.join(directory, name), "w", encoding="utf-8").close()
            self.assertEqual(
                export_rollouts.find_csv_files(directory),
                [os.path.join(directory, name)],
            )


class WriterTest(unittest.TestCase):
    def test_writes_the_header_once_then_a_row_per_observation(self) -> None:
        stream = io.StringIO()
        writer = mpc_observation_logger.ObservationCsvWriter(stream)
        for i in range(3):
            self.assertTrue(writer.write(observation(0.01 * i, [i, -i], [2.0 * i])))
        rows = list(csv.reader(io.StringIO(stream.getvalue())))
        self.assertEqual(rows[0], ["time", "mode", "x0", "x1", "u0"])
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            [float(value) for value in rows[3]], [0.02, 3.0, 2.0, -2.0, 4.0]
        )
        self.assertEqual((writer.written, writer.skipped), (3, 0))

    def test_an_observation_of_other_dimensions_is_skipped_and_counted(self) -> None:
        stream = io.StringIO()
        writer = mpc_observation_logger.ObservationCsvWriter(stream)
        writer.write(observation(0.0, [1.0, 2.0], [3.0]))
        self.assertFalse(writer.write(observation(0.1, [1.0], [3.0])))
        self.assertTrue(writer.write(observation(0.2, [1.0, 2.0], [3.0])))
        self.assertEqual((writer.written, writer.skipped), (2, 1))
        self.assertEqual(len(stream.getvalue().splitlines()), 3)


class BusTest(unittest.TestCase):
    """The logger's subscription over a loopback bus, as `run` sets it up."""

    def test_every_observation_published_reaches_the_file_in_order(self) -> None:
        robot = robot_ipc.Bus(
            ROBOT,
            robot_ipc.NetworkConfig(
                nodes=(
                    robot_ipc.NodeEndpoint(
                        ROBOT, "127.0.0.1", robot_ipc.EPHEMERAL_PORT
                    ),
                )
            ),
        )
        self.addCleanup(robot.close)
        logger_bus = robot_ipc.Bus(
            "",
            robot_ipc.NetworkConfig(
                nodes=(robot_ipc.NodeEndpoint(ROBOT, "127.0.0.1", robot.bound_port),)
            ),
        )
        self.addCleanup(logger_bus.close)
        stream = io.StringIO()
        writer = mpc_observation_logger.ObservationCsvWriter(stream)
        mpc_observation_logger.subscribe(logger_bus, writer)
        robot.start()
        logger_bus.start()

        # ZeroMQ drops what is published before the subscription reaches the publisher: publish probes, of the same
        # dimensions and at negative times, until one arrives.
        def probe_arrived() -> bool:
            robot.publish(
                topics.ROBOT_MPC_OBSERVATION, observation(-1.0, [0.0, 0.0], [0.0])
            )
            time.sleep(0.005)
            return writer.written > 0

        self.assertTrue(wait_for(probe_arrived))
        before = writer.written
        count = 200
        for i in range(count):
            robot.publish(
                topics.ROBOT_MPC_OBSERVATION,
                observation(0.001 * i, [float(i), 1.0], [2.0]),
            )
        self.assertTrue(wait_for(lambda: writer.written >= before + count))

        rows = list(csv.reader(io.StringIO(stream.getvalue())))
        times = [float(row[0]) for row in rows[1:] if float(row[0]) >= 0.0]
        self.assertEqual(times, [0.001 * i for i in range(count)])


class RunTest(unittest.TestCase):
    def test_run_records_for_the_duration_then_writes_a_file_export_rollouts_reads(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            network_file = os.path.join(directory, "network.textproto")
            with open(network_file, "w", encoding="utf-8") as stream:
                stream.write(
                    robot_ipc.format_network_config(
                        robot_ipc.NetworkConfig(
                            nodes=(
                                robot_ipc.NodeEndpoint(ROBOT, "127.0.0.1", free_port()),
                            )
                        )
                    )
                )
            stop = threading.Event()
            stop.set()
            path = mpc_observation_logger.run(
                argparse.Namespace(
                    network_config=network_file, output_dir=directory, duration=0.0
                ),
                stop,
            )
            self.assertTrue(os.path.basename(path).startswith("mpc_observation_"))
            self.assertTrue(os.path.exists(path))

            # A recording as the logger writes it, exported with export_rollouts.py's defaults.
            with open(path, "w", newline="", encoding="utf-8") as stream:
                writer = mpc_observation_logger.ObservationCsvWriter(stream)
                for i in range(5):
                    writer.write(
                        observation(0.1 * i, [i, 2.0 * i, 3.0], [-1.0 * i, 0.5])
                    )
            output = export_rollouts.export_csv_to_h5(
                [path], os.path.join(directory, "demos.npz")
            )
            assert output is not None
            data = np.load(output)
            np.testing.assert_allclose(data["observations"][4], [4.0, 8.0, 3.0])
            np.testing.assert_allclose(data["actions"][4], [-4.0, 0.5])


if __name__ == "__main__":
    unittest.main()
