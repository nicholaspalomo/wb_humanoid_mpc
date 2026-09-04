"""****************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

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

import os
import shutil
import tempfile
import unittest
import yaml

from remote_control.tk_app.yaml_editor_utils import (
    load_yaml_safe,
    update_yaml_values_in_place,
    _update_single_key,
)


class TestYamlEditorUtils(unittest.TestCase):
    def setUp(self):
        self.sample_yaml = """# Top-level comment
model_settings:
  # Robot model parameters
  gravity: 9.81
  friction_coefficient: 0.7  # inline comment

cost_weights:
  Q:
    "(0,0)": 10.0  # pos_x
    "(1,1)": 20.0  # pos_y
  R:
    "(0,0)": 0.01

# Foot constraints
constraints:
  mu: 0.5
"""

    def test_load_yaml_safe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.yaml")
            with open(file_path, "w") as f:
                f.write(self.sample_yaml)

            data = load_yaml_safe(file_path)
            self.assertIn("model_settings", data)
            self.assertEqual(data["model_settings"]["gravity"], 9.81)

            # Test nonexistent file
            self.assertEqual(load_yaml_safe("/nonexistent/file.yaml"), {})

    def test_update_single_key(self):
        lines = self.sample_yaml.splitlines(keepends=True)
        updated_lines = _update_single_key(
            lines, ["model_settings", "gravity"], 9.80665
        )
        updated_text = "".join(updated_lines)

        self.assertIn("# Top-level comment", updated_text)
        self.assertIn("# Robot model parameters", updated_text)
        self.assertIn("gravity: 9.80665", updated_text)
        self.assertIn("friction_coefficient: 0.7  # inline comment", updated_text)

        parsed = yaml.safe_load(updated_text)
        self.assertAlmostEqual(parsed["model_settings"]["gravity"], 9.80665)
        self.assertAlmostEqual(parsed["model_settings"]["friction_coefficient"], 0.7)

    def test_update_yaml_values_in_place_with_backup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.yaml")
            with open(file_path, "w") as f:
                f.write(self.sample_yaml)

            updates = [
                (["model_settings", "gravity"], 9.80665),
                (["cost_weights", "Q", "(0,0)"], 15.5),
                (["constraints", "mu"], 0.8),
            ]
            success = update_yaml_values_in_place(
                file_path, updates, create_backup=True
            )
            self.assertTrue(success)

            # Check backup was created
            bak_path = file_path + ".bak"
            self.assertTrue(os.path.exists(bak_path))
            with open(bak_path, "r") as f:
                self.assertEqual(f.read(), self.sample_yaml)

            # Check file was updated and preserved comments
            with open(file_path, "r") as f:
                new_text = f.read()

            self.assertIn("# Top-level comment", new_text)
            self.assertIn("# Robot model parameters", new_text)
            self.assertIn("# pos_x", new_text)

            parsed = yaml.safe_load(new_text)
            self.assertAlmostEqual(parsed["model_settings"]["gravity"], 9.80665)
            self.assertAlmostEqual(parsed["cost_weights"]["Q"]["(0,0)"], 15.5)
            self.assertAlmostEqual(parsed["constraints"]["mu"], 0.8)


class TestTuningTabsWithFiles(unittest.TestCase):
    """Test tab widgets against actual workspace YAML files."""

    def setUp(self):
        self.repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../..")
        )
        self.g1_pd_file = os.path.join(
            self.repo_root,
            "robot_models/unitree_g1/g1_wb_mpc/config/controller/joint_pd_gains.yaml",
        )
        self.atlas_task_file = os.path.join(
            self.repo_root,
            "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml",
        )

    def test_joint_pd_yaml_syntax_and_structure(self):
        """Verify that G1 and Atlas joint_pd_gains.yaml parse cleanly."""
        self.assertTrue(os.path.exists(self.g1_pd_file), f"Missing {self.g1_pd_file}")
        with open(self.g1_pd_file, "r") as f:
            data = yaml.safe_load(f)
        self.assertIn("default_gains", data)
        self.assertIn("joint_gains", data)
        self.assertGreater(len(data["joint_gains"]), 0)

    def test_all_task_yaml_enable_telemetry_and_online_tuning_flags(self):
        """Verify that all task.yaml files define enableTelemetry and enableOnlineTuning."""
        task_files = [
            "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml",
            "robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.yaml",
            "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.yaml",
            "robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/mpc/task.yaml",
        ]
        for rel_path in task_files:
            abs_path = os.path.join(self.repo_root, rel_path)
            self.assertTrue(os.path.exists(abs_path), f"Missing task file: {rel_path}")
            data = load_yaml_safe(abs_path)
            self.assertIn(
                "enableTelemetry", data, f"Missing enableTelemetry in {rel_path}"
            )
            self.assertTrue(
                data["enableTelemetry"], f"enableTelemetry should be True in {rel_path}"
            )
            self.assertIn(
                "enableOnlineTuning", data, f"Missing enableOnlineTuning in {rel_path}"
            )
            self.assertTrue(
                data["enableOnlineTuning"],
                f"enableOnlineTuning should be True in {rel_path}",
            )

    def test_joint_pd_tab_online_tuning_toggle(self):
        """Verify that JointPdGainsTab disables interaction when enable_online_tuning=False."""
        import tkinter as tk
        from remote_control.tk_app.joint_pd_tab import JointPdGainsTab

        root = tk.Tk()
        root.withdraw()
        try:
            tab = JointPdGainsTab(
                root, pd_gains_file=self.g1_pd_file, enable_online_tuning=False
            )
            self.assertFalse(tab.enable_online_tuning)
            self.assertEqual(str(tab.save_btn.cget("state")), "disabled")
            self.assertEqual(str(tab.reset_btn.cget("state")), "disabled")

            # Check that slider rows are disabled
            for row in tab.slider_rows.values():
                self.assertEqual(str(row.scale.cget("state")), "disabled")

            # Enable tuning dynamically
            tab.set_online_tuning_enabled(True)
            self.assertTrue(tab.enable_online_tuning)
            self.assertEqual(str(tab.save_btn.cget("state")), "normal")
            self.assertEqual(str(tab.reset_btn.cget("state")), "normal")
            for row in tab.slider_rows.values():
                self.assertEqual(str(row.scale.cget("state")), "normal")
        finally:
            root.destroy()

    def test_mpc_params_tab_online_tuning_toggle(self):
        """Verify that MpcParamsTab disables interaction when enable_online_tuning=False."""
        import tkinter as tk
        from remote_control.tk_app.mpc_params_tab import MpcParamsTab

        root = tk.Tk()
        root.withdraw()
        try:
            tab = MpcParamsTab(
                root, task_file=self.atlas_task_file, enable_online_tuning=False
            )
            self.assertFalse(tab.enable_online_tuning)
            self.assertEqual(str(tab.save_btn.cget("state")), "disabled")
            self.assertEqual(str(tab.reset_btn.cget("state")), "disabled")

            # Check that active category slider rows are disabled
            for row in tab.slider_rows.values():
                self.assertEqual(str(row.scale.cget("state")), "disabled")

            # Enable tuning dynamically
            tab.set_online_tuning_enabled(True)
            self.assertTrue(tab.enable_online_tuning)
            self.assertEqual(str(tab.save_btn.cget("state")), "normal")
            self.assertEqual(str(tab.reset_btn.cget("state")), "normal")
            for row in tab.slider_rows.values():
                self.assertEqual(str(row.scale.cget("state")), "normal")
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
