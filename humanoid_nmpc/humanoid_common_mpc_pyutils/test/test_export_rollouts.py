"""export_rollouts.py picks the MPC state and input columns of the logger's files by default, still reads the ROS-era
logger's files, and writes every row of every file."""

import csv
import os
import tempfile
import unittest
from typing import List, Sequence

import numpy as np
from humanoid_common_mpc_pyutils import export_rollouts


def write_log(path: str, header: Sequence[str], rows: List[List[float]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


class DefaultColumnsTest(unittest.TestCase):
    def test_the_logger_columns_are_the_state_and_the_input(self) -> None:
        self.assertEqual(
            export_rollouts.default_columns(["time", "mode", "x0", "x1", "u0"]),
            (["x0", "x1"], ["u0"]),
        )

    def test_the_ros_era_columns_are_selected_by_prefix(self) -> None:
        header = [
            "h_x",
            "L_x",
            "p_x",
            "euler_z",
            "q_j_hip",
            "F_l_x",
            "M_l_x",
            "qd_j_hip",
            "time",
        ]
        self.assertEqual(
            export_rollouts.default_columns(header),
            (
                ["h_x", "L_x", "p_x", "euler_z", "q_j_hip"],
                ["F_l_x", "M_l_x", "qd_j_hip"],
            ),
        )


class ExportTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory.name

    def test_every_row_of_every_file_is_exported_in_order(self) -> None:
        header = ["time", "mode", "x0", "x1", "u0"]
        write_log(
            os.path.join(self.directory, "mpc_observation_1.csv"),
            header,
            [[0.0, 3, 1.0, 2.0, 3.0], [0.1, 3, 4.0, 5.0, 6.0]],
        )
        write_log(
            os.path.join(self.directory, "mpc_observation_2.csv"),
            header,
            [[0.0, 3, 7.0, 8.0, 9.0]],
        )
        output = export_rollouts.export_csv_to_h5(
            export_rollouts.find_csv_files(self.directory),
            os.path.join(self.directory, "demos.npz"),
        )
        assert output is not None
        data = np.load(output)
        np.testing.assert_allclose(data["observations"], [[1, 2], [4, 5], [7, 8]])
        np.testing.assert_allclose(data["actions"], [[3], [6], [9]])

    def test_named_columns_override_the_defaults_and_a_missing_one_is_an_error(
        self,
    ) -> None:
        path = os.path.join(self.directory, "mpc_observation_1.csv")
        write_log(path, ["time", "mode", "x0", "x1", "u0"], [[0.0, 3, 1.0, 2.0, 3.0]])
        output = export_rollouts.export_csv_to_h5(
            [path], os.path.join(self.directory, "demos.npz"), ["x1"], ["x0", "u0"]
        )
        assert output is not None
        data = np.load(output)
        np.testing.assert_allclose(data["observations"], [[2.0]])
        np.testing.assert_allclose(data["actions"], [[1.0, 3.0]])
        with self.assertRaisesRegex(ValueError, "x9"):
            export_rollouts.export_csv_to_h5(
                [path], os.path.join(self.directory, "other.npz"), ["x9"]
            )

    def test_an_empty_log_exports_nothing(self) -> None:
        path = os.path.join(self.directory, "mpc_observation_1.csv")
        write_log(path, ["time", "mode", "x0", "u0"], [])
        self.assertIsNone(
            export_rollouts.export_csv_to_h5(
                [path], os.path.join(self.directory, "demos.npz")
            )
        )


if __name__ == "__main__":
    unittest.main()
