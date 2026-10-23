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
        self.assertEqual(len(filtered[0].panels()), 12)
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

    def test_load_default_plot_config(self) -> None:
        cfg = plot_config.load_plot_config(
            "humanoid_nmpc/humanoid_rerun_viewer/config/plot_config.textproto"
        )
        self.assertFalse(cfg.draw_terminal_state)
        self.assertEqual(
            plot_config.resolve_allowed_instances(cfg),
            ("measured",),
        )


_HEADER = """# proto-file: humanoid_nmpc/humanoid_rerun_viewer/proto/plot_config.proto
# proto-message: humanoid_rerun_viewer_proto.PlotConfig
"""


class MujocoViewerConfigTest(unittest.TestCase):
    """mujoco_viewer: the robot process's MuJoCo viewer by registry name, checked as the robot checks it."""

    def test_the_field_names_a_viewer_or_a_preference_list(self) -> None:
        cfg = plot_config.parse_plot_config(f'{_HEADER}mujoco_viewer: "glfw"\n')
        self.assertEqual(cfg.mujoco_viewer, ("glfw",))
        cfg = plot_config.parse_plot_config(
            f'{_HEADER}mujoco_viewer: ["web", "none"]\n'
        )
        self.assertEqual(cfg.mujoco_viewer, ("web", "none"))

    def test_without_the_field_the_list_is_empty_and_the_robot_chooses(self) -> None:
        # The robot process's built-in default (web) applies; the C++ tests pin it (testRobotAppOptions.cpp).
        cfg = plot_config.parse_plot_config(f'{_HEADER}signals: "base_pose/*"\n')
        self.assertEqual(cfg.mujoco_viewer, ())
        self.assertEqual(plot_config.PlotConfig().mujoco_viewer, ())
        self.assertEqual(plot_config.MUJOCO_VIEWER_NAMES, ("glfw", "web", "none"))

    def test_an_unknown_name_is_refused_with_the_valid_ones(self) -> None:
        with self.assertRaises(plot_config.PlotConfigError) as raised:
            plot_config.parse_plot_config(
                f'{_HEADER}mujoco_viewer: "vnc"\n', source="my.textproto"
            )
        message = str(raised.exception)
        self.assertIn("my.textproto", message)
        self.assertIn("'vnc'", message)
        self.assertIn("glfw, web, none", message)
        # Exactly as written: the robot does not strip a name either.
        with self.assertRaises(plot_config.PlotConfigError):
            plot_config.check_mujoco_viewer(["web "])
        with self.assertRaises(plot_config.PlotConfigError):
            plot_config.check_mujoco_viewer([""])

    def test_a_list_whose_later_entry_could_never_start_is_refused(self) -> None:
        for names, reason in (
            (["web", "web"], "listed twice"),
            (["none", "web"], "always starts"),
            (["glfw", "none"], "cannot fail at start-up"),
            (["glfw", "web"], "cannot fail at start-up"),
        ):
            with self.subTest(names=names):
                with self.assertRaises(plot_config.PlotConfigError) as raised:
                    plot_config.check_mujoco_viewer(names)
                self.assertIn(reason, str(raised.exception))
        for names in (["web"], ["web", "none"], ["web", "glfw"], ["none"], []):
            with self.subTest(names=names):
                self.assertEqual(plot_config.check_mujoco_viewer(names), tuple(names))

    def test_the_shipped_plot_config_starts_the_web_viewer(self) -> None:
        cfg = plot_config.load_plot_config(
            "humanoid_nmpc/humanoid_rerun_viewer/config/plot_config.textproto"
        )
        self.assertEqual(cfg.mujoco_viewer, ("web",))


class ShippedPlotConfigTest(unittest.TestCase):
    """The shipped plot configuration shows the base's Euler angles, Euler rates and body angular velocity."""

    def test_the_default_tabs_plot_the_angles_and_both_angular_rates(self) -> None:
        cfg = plot_config.load_plot_config(
            "humanoid_nmpc/humanoid_rerun_viewer/config/plot_config.textproto"
        )
        filtered, _ = plot_config.filter_contract_tabs(
            telemetry_contract.TABS, list(cfg.signals)
        )
        plotted: set[str] = set()
        for tab in filtered:
            for panel in tab.panels():
                plotted.update(panel.paths)
        for angle in ("roll", "pitch", "yaw"):
            self.assertIn(f"base_pose/{angle}", plotted)
            self.assertIn(f"base_pose/{angle}_rate", plotted)
        for axis in ("x", "y", "z"):
            self.assertIn(f"base_pose/angular_velocity_{axis}", plotted)


class TelemetryRecordingConfigTest(unittest.TestCase):
    """telemetry_rate_hz and record_unplotted_telemetry, and the groups telemetry_selection() records."""

    def test_the_rate_is_parsed_and_defaults_when_unset(self) -> None:
        cfg = plot_config.parse_plot_config(f"{_HEADER}telemetry_rate_hz: 25\n")
        self.assertEqual(cfg.telemetry_rate_hz, 25.0)
        cfg = plot_config.parse_plot_config(f"{_HEADER}telemetry_rate_hz: 0\n")
        self.assertEqual(cfg.telemetry_rate_hz, 0.0)
        cfg = plot_config.parse_plot_config(f'{_HEADER}signals: "base_pose/*"\n')
        self.assertEqual(cfg.telemetry_rate_hz, plot_config.DEFAULT_TELEMETRY_RATE_HZ)
        self.assertEqual(
            plot_config.PlotConfig().telemetry_rate_hz,
            plot_config.DEFAULT_TELEMETRY_RATE_HZ,
        )
        cfg = plot_config.parse_plot_config("base_pose/*\ntelemetry_rate_hz: 12.5\n")
        self.assertEqual(cfg.telemetry_rate_hz, 12.5)
        self.assertEqual(cfg.signals, ("base_pose/*",))

    def test_a_negative_or_non_finite_rate_is_refused(self) -> None:
        for value in ("-1", "nan", "inf"):
            for text in (
                f"{_HEADER}telemetry_rate_hz: {value}\n",
                f"base_pose/*\ntelemetry_rate_hz: {value}\n",
            ):
                with self.subTest(text=text):
                    with self.assertRaises(plot_config.PlotConfigError) as raised:
                        plot_config.parse_plot_config(text, source="my.textproto")
                    self.assertIn("telemetry_rate_hz", str(raised.exception))
                    self.assertIn("my.textproto", str(raised.exception))
        with self.assertRaises(plot_config.PlotConfigError):
            plot_config.parse_plot_config("base_pose/*\ntelemetry_rate_hz: fast\n")

    def test_record_unplotted_telemetry_is_parsed(self) -> None:
        self.assertFalse(plot_config.PlotConfig().record_unplotted_telemetry)
        cfg = plot_config.parse_plot_config(
            f"{_HEADER}record_unplotted_telemetry: true\n"
        )
        self.assertTrue(cfg.record_unplotted_telemetry)
        cfg = plot_config.parse_plot_config(
            "base_pose/*\nrecord_unplotted_telemetry: true\n"
        )
        self.assertTrue(cfg.record_unplotted_telemetry)
        self.assertEqual(cfg.signals, ("base_pose/*",))

    def test_the_selection_records_exactly_the_plotted_groups(self) -> None:
        cfg = plot_config.PlotConfig(
            signals=("Base Pose & Euler", "frames/pose/*", "my/custom", "status/mpc/*")
        )
        records = plot_config.telemetry_selection(cfg)
        assert records is not None
        tabs, _ = plot_config.filter_contract_tabs(telemetry_contract.TABS, cfg.signals)
        panel_paths: list[str] = []
        for tab in tabs:
            for panel in tab.panels():
                panel_paths.extend(panel.paths)
        frames = ("foot_l_contact", "pelvis")
        for group in telemetry_contract.all_groups(frames):
            plotted = any(
                group.path == path
                or (path.endswith("/**") and group.path.startswith(path[: -len("**")]))
                for path in panel_paths
            )
            with self.subTest(path=group.path):
                self.assertEqual(records(group.path), plotted)
        self.assertTrue(records("base_pose/roll"))
        frame_poses = [
            group.path
            for group in telemetry_contract.frame_groups("pelvis")
            if group.path.startswith("frames/pose/")
        ]
        self.assertTrue(frame_poses)
        for path in frame_poses:
            self.assertTrue(records(path), path)
        self.assertTrue(records("my/custom"))
        self.assertFalse(records("joints/position/measured"))
        self.assertFalse(records("mpc/status"))

    def test_without_a_selection_every_group_is_recorded(self) -> None:
        self.assertIsNone(plot_config.telemetry_selection(None))
        self.assertIsNone(
            plot_config.telemetry_selection(
                plot_config.PlotConfig(
                    signals=("base_pose/*",), record_unplotted_telemetry=True
                )
            )
        )

    def test_the_shipped_file_records_its_plots_at_50_hz(self) -> None:
        cfg = plot_config.load_plot_config(
            "humanoid_nmpc/humanoid_rerun_viewer/config/plot_config.textproto"
        )
        self.assertEqual(cfg.telemetry_rate_hz, 50.0)
        self.assertFalse(cfg.record_unplotted_telemetry)
        records = plot_config.telemetry_selection(cfg)
        assert records is not None
        # The plots the user asked for: Euler angles and rates, the balance quantities, the per-term costs and
        # constraints.
        for path in (
            "base_pose/roll",
            "base_pose/yaw_rate",
            "base_pose/angular_velocity_x",
            "com/position_x",
            "dcm/x",
            "zmp/x",
            "acom/roll",
            "mpc_solution/costs",
            "mpc_solution/constraints",
        ):
            with self.subTest(path=path):
                self.assertTrue(records(path), path)
        self.assertFalse(records("dofs/position/measured"))


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

    def test_an_empty_config_draws_every_instance(self) -> None:
        # Every instance the scene carries, terminal_target included (which the blueprint hides until shown).
        self.assertIsNone(
            plot_config.resolve_allowed_instances(plot_config.PlotConfig())
        )

    def test_draw_terminal_state_false_overrides_robot_instances(self) -> None:
        cfg = plot_config.PlotConfig(
            robot_instances=("measured", "terminal_state", "terminal_target"),
            draw_terminal_state=False,
        )
        self.assertEqual(plot_config.resolve_allowed_instances(cfg), ("measured",))

    def test_the_flag_brings_back_the_robots_the_shipped_file_drops(self) -> None:
        cfg = plot_config.load_plot_config(
            "humanoid_nmpc/humanoid_rerun_viewer/config/plot_config.textproto"
        )
        self.assertIsNone(
            plot_config.resolve_allowed_instances(cfg, cli_terminal_state=True)
        )

    def test_the_schema_comments_describe_the_resolution(self) -> None:
        # proto/plot_config.proto's comments are the schema's documentation: they must say what the code above does.
        with open(
            "humanoid_nmpc/humanoid_rerun_viewer/proto/plot_config.proto",
            encoding="utf-8",
        ) as schema:
            text = schema.read()
        instances = text[: text.index("repeated string robot_instances")]
        instances = instances[instances.rindex("string custom_tab_title") :]
        self.assertNotIn('defaults to ("measured", "terminal_state")', instances)
        self.assertIn("every instance", instances)
        self.assertIn("terminal_target", instances)
        terminal = text[: text.index("optional bool draw_terminal_state")]
        terminal = terminal[terminal.rindex("repeated string robot_instances") :]
        self.assertIn('only "measured"', terminal)
        self.assertIn("whatever robot_instances", terminal)


if __name__ == "__main__":
    unittest.main()
