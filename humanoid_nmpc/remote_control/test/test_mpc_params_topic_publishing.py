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

"""
Tests for the ROS topic-based MPC parameter publishing pipeline.

Verifies the decoupling between real-time MPC parameter updates (via ROS topic)
and explicit YAML file saves (via "Save to YAML" button).  The tests use a mock
publisher to capture published messages without requiring a running ROS graph.
"""

import os
import shutil
import tempfile
import unittest

from remote_control.tk_app.yaml_editor_utils import load_yaml_safe


class MockPublisher:
    """Stand-in for rclpy Publisher that records published messages."""

    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)

    @property
    def last_data(self):
        if not self.messages:
            return None
        return self.messages[-1].data

    @property
    def publish_count(self):
        return len(self.messages)


class MockStringMsg:
    """Minimal stand-in for std_msgs.msg.String."""

    def __init__(self):
        self.data = ""


class TestMpcParamsTopicPublishing(unittest.TestCase):
    """Test suite verifying that MpcParamsTab publishes slider values to a
    ROS topic without modifying task.yaml on disk."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../..")
        )
        cls.atlas_task_file = os.path.join(
            cls.repo_root,
            "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml",
        )

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.tmp_task_file = os.path.join(self.tmpdir, "task.yaml")
        shutil.copy2(self.atlas_task_file, self.tmp_task_file)
        self.mock_publisher = MockPublisher()
        # Snapshot file content to detect unintended writes
        with open(self.tmp_task_file, "r") as f:
            self.original_content = f.read()
        self.original_mtime = os.path.getmtime(self.tmp_task_file)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_tab(self, root, category=None):
        """Create MpcParamsTab with a mock publisher."""
        from remote_control.tk_app.mpc_params_tab import MpcParamsTab

        tab = MpcParamsTab(
            root,
            task_file=self.tmp_task_file,
            enable_online_tuning=True,
            param_publisher=self.mock_publisher,
        )
        if category:
            tab.active_category.set(category)
            tab._render_active_category()
        return tab

    def _file_was_modified(self):
        """Check if task.yaml on disk was modified since setUp."""
        with open(self.tmp_task_file, "r") as f:
            current_content = f.read()
        return current_content != self.original_content

    # ──────────────────────────────────────────────────────────
    #  1. Slider changes publish to topic, NOT to YAML
    # ──────────────────────────────────────────────────────────
    def test_slider_change_publishes_to_topic(self):
        """Moving a slider should trigger a publish to the mock publisher."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            tab = self._create_tab(root, "State Cost (Q)")

            # Find the first slider and change it
            key = list(tab.slider_rows.keys())[0]
            row = tab.slider_rows[key]
            original = row.get_value()
            row.set_value(original * 1.5)

            # Directly call the publish method (bypasses debounce timer)
            tab._publish_to_topic()

            self.assertGreater(
                self.mock_publisher.publish_count,
                0,
                "Expected at least one publish after slider change",
            )
        finally:
            root.destroy()

    def test_slider_change_does_not_modify_yaml(self):
        """Moving a slider and publishing should NOT modify task.yaml on disk."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            tab = self._create_tab(root, "State Cost (Q)")

            # Change a slider value
            key = list(tab.slider_rows.keys())[0]
            row = tab.slider_rows[key]
            row.set_value(row.get_value() * 2.0)

            # Trigger the publish path (not auto-save)
            tab._publish_to_topic()

            # Verify YAML was NOT modified
            self.assertFalse(
                self._file_was_modified(),
                "task.yaml should NOT be modified by slider changes — "
                "only the ROS topic should be used",
            )
        finally:
            root.destroy()

    # ──────────────────────────────────────────────────────────
    #  2. "Save to YAML" writes to file
    # ──────────────────────────────────────────────────────────
    def test_save_to_yaml_writes_file(self):
        """Clicking 'Save to YAML' should write slider values to task.yaml."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            tab = self._create_tab(root, "State Cost (Q)")

            # Change a slider
            key = list(tab.slider_rows.keys())[0]
            row = tab.slider_rows[key]
            row.set_value(row.get_value() * 2.0)

            # Explicit save
            tab.save_and_checkpoint()

            self.assertTrue(
                self._file_was_modified(),
                "task.yaml SHOULD be modified after 'Save to YAML'",
            )
        finally:
            root.destroy()

    def test_save_and_checkpoint_updates_defaults(self):
        """save_and_checkpoint should update slider defaults for Reset All."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            tab = self._create_tab(root, "State Cost (Q)")

            key = list(tab.slider_rows.keys())[0]
            row = tab.slider_rows[key]
            new_val = row.get_value() * 2.0
            row.set_value(new_val)

            tab.save_and_checkpoint()

            # The default should now be the new value
            self.assertAlmostEqual(
                row.default_value,
                new_val,
                places=3,
                msg="Default value should be updated after save_and_checkpoint",
            )
        finally:
            root.destroy()

    # ──────────────────────────────────────────────────────────
    #  3. "Reset All" publishes reset values to topic
    # ──────────────────────────────────────────────────────────
    def test_reset_all_publishes_to_topic(self):
        """Reset All should publish default values to the ROS topic."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            tab = self._create_tab(root, "State Cost (Q)")

            # Change a slider, then reset
            key = list(tab.slider_rows.keys())[0]
            row = tab.slider_rows[key]
            row.set_value(row.get_value() * 3.0)

            pub_count_before = self.mock_publisher.publish_count
            tab.reset_all_defaults()

            self.assertGreater(
                self.mock_publisher.publish_count,
                pub_count_before,
                "Reset All should publish to the ROS topic",
            )
        finally:
            root.destroy()

    def test_reset_all_restores_default_values(self):
        """Reset All should snap sliders back to default values."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            tab = self._create_tab(root, "State Cost (Q)")

            key = list(tab.slider_rows.keys())[0]
            row = tab.slider_rows[key]
            original_val = row.default_value
            row.set_value(original_val * 3.0)

            tab.reset_all_defaults()

            self.assertAlmostEqual(
                row.get_value(),
                original_val,
                places=3,
                msg="Slider should be restored to default after Reset All",
            )
        finally:
            root.destroy()

    def test_reset_all_does_not_modify_yaml(self):
        """Reset All should publish to topic but NOT modify task.yaml."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            tab = self._create_tab(root, "State Cost (Q)")

            key = list(tab.slider_rows.keys())[0]
            row = tab.slider_rows[key]
            row.set_value(row.get_value() * 3.0)

            tab.reset_all_defaults()

            self.assertFalse(
                self._file_was_modified(),
                "Reset All should NOT write to task.yaml",
            )
        finally:
            root.destroy()

    # ──────────────────────────────────────────────────────────
    #  4. Reload publishes loaded values to topic
    # ──────────────────────────────────────────────────────────
    def test_reload_publishes_to_topic(self):
        """Reload should publish the freshly-loaded values to the ROS topic."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            tab = self._create_tab(root, "State Cost (Q)")

            pub_count_before = self.mock_publisher.publish_count
            tab.reload_file()

            self.assertGreater(
                self.mock_publisher.publish_count,
                pub_count_before,
                "Reload should publish to the ROS topic",
            )
        finally:
            root.destroy()

    def test_reload_does_not_modify_yaml(self):
        """Reload reads the file but should not write back to it."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            tab = self._create_tab(root, "State Cost (Q)")
            tab.reload_file()

            self.assertFalse(
                self._file_was_modified(),
                "Reload should NOT modify task.yaml",
            )
        finally:
            root.destroy()

    # ──────────────────────────────────────────────────────────
    #  5. _build_yaml_with_slider_values correctness
    # ──────────────────────────────────────────────────────────
    def test_build_yaml_contains_slider_values(self):
        """The YAML string builder should reflect current slider values."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            tab = self._create_tab(root, "State Cost (Q)")

            # Set a known distinctive value on the first slider
            key = list(tab.slider_rows.keys())[0]
            row = tab.slider_rows[key]
            distinctive_val = 999.123
            row.set_value(distinctive_val)

            yaml_str = tab._build_yaml_with_slider_values()

            self.assertIn(
                "999.123",
                yaml_str,
                "Built YAML string should contain the distinctive slider value",
            )
        finally:
            root.destroy()

    def test_build_yaml_preserves_original_structure(self):
        """The YAML string should preserve comments and structure from original file."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            tab = self._create_tab(root, "State Cost (Q)")

            yaml_str = tab._build_yaml_with_slider_values()

            # Should contain original YAML structural elements
            self.assertIn("Q:", yaml_str, "YAML string should contain 'Q:' section")
            self.assertIn("R:", yaml_str, "YAML string should contain 'R:' section")
            self.assertIn(
                "Q_final:", yaml_str, "YAML string should contain 'Q_final:' section"
            )
            # Should preserve inline comments
            self.assertIn("#", yaml_str, "YAML string should preserve inline comments")
        finally:
            root.destroy()

    def test_build_yaml_returns_empty_without_file(self):
        """_build_yaml_with_slider_values should return '' if no task_file."""
        import tkinter as tk
        from remote_control.tk_app.mpc_params_tab import MpcParamsTab

        root = tk.Tk()
        root.withdraw()
        try:
            tab = MpcParamsTab(
                root,
                task_file=None,
                enable_online_tuning=True,
                param_publisher=self.mock_publisher,
            )
            result = tab._build_yaml_with_slider_values()
            self.assertEqual(result, "", "Should return empty string with no task_file")
        finally:
            root.destroy()

    # ──────────────────────────────────────────────────────────
    #  6. No-publisher graceful degradation
    # ──────────────────────────────────────────────────────────
    def test_publish_without_publisher_does_not_crash(self):
        """If param_publisher is None, _publish_to_topic should be a no-op."""
        import tkinter as tk
        from remote_control.tk_app.mpc_params_tab import MpcParamsTab

        root = tk.Tk()
        root.withdraw()
        try:
            tab = MpcParamsTab(
                root,
                task_file=self.tmp_task_file,
                enable_online_tuning=True,
                param_publisher=None,  # No publisher
            )
            tab.active_category.set("State Cost (Q)")
            tab._render_active_category()

            # Change a slider and try to publish — should not raise
            key = list(tab.slider_rows.keys())[0]
            tab.slider_rows[key].set_value(42.0)
            tab._publish_to_topic()  # Should be a silent no-op
        finally:
            root.destroy()

    # ──────────────────────────────────────────────────────────
    #  7. Published YAML is parseable by load_yaml_safe
    # ──────────────────────────────────────────────────────────
    def test_published_yaml_is_parseable(self):
        """The YAML string sent via topic should be parseable back to a dict."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            tab = self._create_tab(root, "State Cost (Q)")

            # Change a slider and publish
            key = list(tab.slider_rows.keys())[0]
            row = tab.slider_rows[key]
            row.set_value(row.get_value() * 1.5)
            tab._publish_to_topic()

            # The published message should be parseable YAML
            self.assertGreater(self.mock_publisher.publish_count, 0)
            yaml_str = self.mock_publisher.last_data

            # Write to temp file and parse (same as C++ side does)
            tmp_parse = os.path.join(self.tmpdir, "parse_test.yaml")
            with open(tmp_parse, "w") as f:
                f.write(yaml_str)
            parsed = load_yaml_safe(tmp_parse)

            self.assertIn("Q", parsed, "Parsed YAML should contain 'Q' section")
            self.assertIn("R", parsed, "Parsed YAML should contain 'R' section")
        finally:
            root.destroy()

    # ──────────────────────────────────────────────────────────
    #  8. Debounce replaces pending publishes
    # ──────────────────────────────────────────────────────────
    def test_debounce_replaces_pending(self):
        """Multiple rapid slider changes should only schedule one publish."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            tab = self._create_tab(root, "State Cost (Q)")

            key = list(tab.slider_rows.keys())[0]
            row = tab.slider_rows[key]

            # Simulate 5 rapid slider moves
            for i in range(5):
                row.set_value(row.get_value() + 1.0)
                tab._on_any_slider_change(key, row.get_value())

            # Only one pending after() should be active
            self.assertIsNotNone(
                tab._debounce_publish_id,
                "A debounce timer should be pending after rapid slider moves",
            )

            # Cancel it to avoid interference
            tab.after_cancel(tab._debounce_publish_id)
            tab._debounce_publish_id = None
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
