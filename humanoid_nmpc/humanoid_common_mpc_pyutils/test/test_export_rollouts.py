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

"""Tests for export_rollouts.py.

It picks the MPC state and input columns of the logger's files by default, still reads the ROS-era logger's files, and
writes every row of every file.
"""

from collections.abc import Sequence
import csv
import os
import shutil
import tempfile
import unittest

import numpy as np

from humanoid_common_mpc_pyutils import export_rollouts


def write_log(path: str, header: Sequence[str], rows: list[list[float]]) -> None:
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
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory)

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
