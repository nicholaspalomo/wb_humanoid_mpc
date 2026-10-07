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

"""Tests for plot_config.py and configured blueprint generation."""

import os
import tempfile
import unittest

import rerun.blueprint as rrb

from humanoid_rerun_viewer import blueprint
from humanoid_rerun_viewer import plot_config
from humanoid_rerun_viewer import status_contract
from humanoid_rerun_viewer import telemetry_contract


class ParsePlotConfigTest(unittest.TestCase):
    def test_parse_textproto(self) -> None:
        text = """# proto-file: humanoid_nmpc/humanoid_rerun_viewer/proto/plot_config.proto
# proto-message: humanoid_rerun_viewer_proto.PlotConfig
signals: "Base Pose & Euler"
signals: "contact_forces/*"
signals: "status/mpc/solve_time"
custom_tab_title: "MySignals"
"""
        config = plot_config.parse_plot_config(text)
        self.assertEqual(
            config.signals,
            ("Base Pose & Euler", "contact_forces/*", "status/mpc/solve_time"),
        )
        self.assertEqual(config.custom_tab_title, "MySignals")

    def test_parse_plain_text(self) -> None:
        text = """# Comments are ignored
Base Pose & Euler

contact_forces/*
custom_tab_title: Selected
custom/signal_1
"""
        config = plot_config.parse_plot_config(text)
        self.assertEqual(
            config.signals,
            ("Base Pose & Euler", "contact_forces/*", "custom/signal_1"),
        )
        self.assertEqual(config.custom_tab_title, "Selected")

    def test_malformed_textproto_raises_error(self) -> None:
        text = """# proto-file: humanoid_nmpc/humanoid_rerun_viewer/proto/plot_config.proto
# proto-message: humanoid_rerun_viewer_proto.PlotConfig
invalid_field: "foo"
"""
        with self.assertRaises(plot_config.PlotConfigError):
            plot_config.parse_plot_config(text, source="test.textproto")

    def test_load_file(self) -> None:
        with tempfile.NamedTemporaryFile("w", delete=False) as temp_file:
            temp_file.write("base_pose/position_x\njoints/*\n")
            temp_path = temp_file.name
        self.addCleanup(os.remove, temp_path)

        config = plot_config.load_plot_config(temp_path)
        self.assertEqual(config.signals, ("base_pose/position_x", "joints/*"))


class FilteringContractTabsTest(unittest.TestCase):
    def test_filter_by_tab_title(self) -> None:
        filtered, matched = plot_config.filter_contract_tabs(
            telemetry_contract.TABS, ["Base Pose & Euler"]
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].title, "Base Pose & Euler")
        self.assertEqual(len(filtered[0].panels()), 6)
        self.assertIn("Base Pose & Euler", matched)

    def test_filter_by_panel_title(self) -> None:
        filtered, matched = plot_config.filter_contract_tabs(
            telemetry_contract.TABS, ["Base Pos X [m]"]
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].title, "Base Pose & Euler")
        self.assertEqual(len(filtered[0].panels()), 1)
        self.assertEqual(filtered[0].panels()[0].title, "Base Pos X [m]")
        self.assertIn("Base Pos X [m]", matched)

    def test_filter_by_path_wildcard(self) -> None:
        filtered, matched = plot_config.filter_contract_tabs(
            telemetry_contract.TABS, ["base_pose/position_*"]
        )
        self.assertEqual(len(filtered), 1)
        panel_titles = [p.title for p in filtered[0].panels()]
        self.assertEqual(
            panel_titles,
            ["Base Pos X [m]", "Base Pos Y [m]", "Base Pos Z (Height) [m]"],
        )
        self.assertIn("base_pose/position_*", matched)

    def test_filter_status_series(self) -> None:
        filtered, matched = plot_config.filter_status_series(["status/mpc/*"])
        self.assertGreater(len(filtered), 0)
        for s in filtered:
            self.assertTrue(s.path.startswith("status/mpc/"))
        self.assertIn("status/mpc/*", matched)

    def test_filter_full_status_tab(self) -> None:
        filtered, matched = plot_config.filter_status_series(["Status"])
        self.assertEqual(len(filtered), len(status_contract.STATUS_SERIES))
        self.assertIn("Status", matched)

    def test_unmatched_custom_signals(self) -> None:
        signals = ["base_pose/position_x", "custom/random_signal"]
        _, matched = plot_config.filter_contract_tabs(telemetry_contract.TABS, signals)
        _, status_matched = plot_config.filter_status_series(signals)
        matched.update(status_matched)
        unmatched = plot_config.unmatched_signals(signals, matched)
        self.assertEqual(unmatched, ["custom/random_signal"])


class BlueprintConfiguredTabsTest(unittest.TestCase):
    def test_build_blueprint_with_filtered_and_custom_signals(self) -> None:
        config = plot_config.PlotConfig(
            signals=("base_pose/position_x", "status/mpc/solve_time", "custom/sensor"),
            custom_tab_title="CustomSignals",
        )
        bp = blueprint.build_blueprint(tracked_link="pelvis", config=config)
        root = bp.root_container
        self.assertIsInstance(root, rrb.Horizontal)
        _, tabs_container = root.contents
        self.assertIsInstance(tabs_container, rrb.Tabs)
        tab_names = [tab.name for tab in tabs_container.contents]
        self.assertEqual(
            tab_names,
            ["Base Pose & Euler", "Status", "CustomSignals"],
        )

    def test_build_blueprint_with_empty_signals_has_no_plot_tabs(self) -> None:
        config = plot_config.PlotConfig(signals=())
        bp = blueprint.build_blueprint(tracked_link="pelvis", config=config)
        root = bp.root_container
        self.assertIsInstance(root, rrb.Vertical)

    def test_has_telemetry_signals(self) -> None:
        self.assertTrue(plot_config.has_telemetry_signals(None))
        self.assertFalse(
            plot_config.has_telemetry_signals(plot_config.PlotConfig(signals=()))
        )
        self.assertFalse(
            plot_config.has_telemetry_signals(
                plot_config.PlotConfig(signals=("status/mpc/solve_time",))
            )
        )
        self.assertTrue(
            plot_config.has_telemetry_signals(
                plot_config.PlotConfig(signals=("base_pose/position_x",))
            )
        )
        self.assertTrue(
            plot_config.has_telemetry_signals(
                plot_config.PlotConfig(signals=("custom/telemetry",))
            )
        )


class RobotInstancesConfigTest(unittest.TestCase):
    def test_parse_textproto_with_robot_instances(self) -> None:
        content = """# proto-file: humanoid_nmpc/humanoid_rerun_viewer/proto/plot_config.proto
# proto-message: humanoid_rerun_viewer_proto.PlotConfig
signals: "base_pose/*"
robot_instances: "measured"
draw_terminal_state: false
"""
        cfg = plot_config.parse_plot_config(content)
        self.assertEqual(cfg.robot_instances, ("measured",))
        self.assertFalse(cfg.draw_terminal_state)

    def test_parse_plain_text_with_robot_instances(self) -> None:
        content = """# signals
base_pose/*
robot_instances: measured, terminal_state
draw_terminal_state: false
"""
        cfg = plot_config.parse_plot_config(content)
        self.assertEqual(cfg.signals, ("base_pose/*",))
        self.assertEqual(cfg.robot_instances, ("measured", "terminal_state"))
        self.assertFalse(cfg.draw_terminal_state)

    def test_load_example_plot_config(self) -> None:
        cfg = plot_config.load_plot_config(
            "humanoid_nmpc/humanoid_rerun_viewer/config/example_plot_config.textproto"
        )
        self.assertFalse(cfg.draw_terminal_state)
        self.assertEqual(
            plot_config.resolve_allowed_instances(cfg),
            ("measured",),
        )

    def test_load_default_plot_config(self) -> None:
        cfg = plot_config.load_plot_config(
            "humanoid_nmpc/humanoid_rerun_viewer/config/plot_config.textproto"
        )
        self.assertFalse(cfg.draw_terminal_state)
        self.assertEqual(
            plot_config.resolve_allowed_instances(cfg),
            ("measured",),
        )


class ResolveAllowedInstancesTest(unittest.TestCase):
    def test_default_returns_none(self) -> None:
        self.assertIsNone(plot_config.resolve_allowed_instances(None))

    def test_cli_flags_take_precedence(self) -> None:
        cfg = plot_config.PlotConfig(draw_terminal_state=True)
        self.assertEqual(
            plot_config.resolve_allowed_instances(cfg, cli_terminal_state=False),
            ("measured",),
        )
        self.assertEqual(
            plot_config.resolve_allowed_instances(cfg, cli_measured_only=True),
            ("measured",),
        )
        self.assertEqual(
            plot_config.resolve_allowed_instances(
                cfg, cli_robot_instances=("custom_robot",)
            ),
            ("custom_robot",),
        )

    def test_config_draw_terminal_state_false_disables_it(self) -> None:
        cfg = plot_config.PlotConfig(draw_terminal_state=False)
        self.assertEqual(
            plot_config.resolve_allowed_instances(cfg),
            ("measured",),
        )

    def test_config_robot_instances_used(self) -> None:
        cfg = plot_config.PlotConfig(robot_instances=("measured",))
        self.assertEqual(
            plot_config.resolve_allowed_instances(cfg),
            ("measured",),
        )


if __name__ == "__main__":
    unittest.main()
