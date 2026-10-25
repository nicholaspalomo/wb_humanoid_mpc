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

"""The viewer's periods reject zero, negative and NaN values: the BusBridge options, --flush_period and --max_scene_frequency.

The checks are written `not period > 0.0`: `period <= 0.0` is False for NaN, which would then reach the bus's loop.
"""

import contextlib
import io
import math
import unittest
from unittest import mock

from humanoid_rerun_viewer import bridge
from humanoid_rerun_viewer import bus_bridge
from humanoid_rerun_viewer import cli
from humanoid_rerun_viewer import plot_config as plot_config_module
import robot_ipc

INVALID_PERIODS = (0.0, -1.0, math.nan)


def _network() -> robot_ipc.NetworkConfig:
    """A network of one node of this test process's own DDS domain."""
    return robot_ipc.isolated_test_network(["publisher"])


class BusBridgePeriodTest(unittest.TestCase):
    def test_the_periods_must_be_positive(self) -> None:
        the_bridge = mock.create_autospec(bridge.RerunBridge, instance=True)
        for period in INVALID_PERIODS:
            with self.subTest(option="flush_period", period=period):
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    bus_bridge.BusBridge(
                        the_bridge, _network(), flush_period=period
                    )
            with self.subTest(option="report_period", period=period):
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    bus_bridge.BusBridge(
                        the_bridge, _network(), report_period=period
                    )

    def test_bus_bridge_without_telemetry(self) -> None:
        the_bridge = mock.create_autospec(bridge.RerunBridge, instance=True)
        bus = bus_bridge.BusBridge(the_bridge, _network(), telemetry=False)
        try:
            self.assertNotIn("viz/telemetry", bus.subscribed_topics)
        finally:
            bus.bus.close()


class FlushPeriodFlagTest(unittest.TestCase):
    def test_the_flag_must_be_positive(self) -> None:
        for period in INVALID_PERIODS:
            with self.subTest(period=period):
                errors = io.StringIO()
                with contextlib.redirect_stderr(errors):
                    with self.assertRaises(SystemExit) as raised:
                        cli.main(["--flush_period", str(period)])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(
                    "--flush_period must be positive", errors.getvalue()
                )


class MaxSceneFrequencyFlagTest(unittest.TestCase):
    def test_the_flag_must_be_non_negative(self) -> None:
        for frequency in (-1.0, math.nan, math.inf):
            with self.subTest(frequency=frequency):
                errors = io.StringIO()
                with contextlib.redirect_stderr(errors):
                    with self.assertRaises(SystemExit) as raised:
                        cli.main(["--max_scene_frequency", str(frequency)])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(
                    "--max_scene_frequency must be non-negative",
                    errors.getvalue(),
                )


def _help(flag: str) -> str:
    """The help text of `flag` in the bridge's parser."""
    for action in cli.build_parser()._actions:
        if flag in action.option_strings:
            return action.help or ""
    raise AssertionError(f"no flag {flag}")


def _shipped_plot_config() -> plot_config_module.PlotConfig:
    return plot_config_module.load_plot_config(
        cli.resolve_input_path(cli.DEFAULT_PLOT_CONFIG)
    )


class PlotsDefaultTest(unittest.TestCase):
    """Whether the bridge plots: what --plots and its help say, and what every shipped launch then gets."""

    def test_the_flag_wins(self) -> None:
        config = plot_config_module.PlotConfig(signals=("x",))
        self.assertFalse(cli.plots_enabled(False, "spawn", config))
        self.assertTrue(cli.plots_enabled(True, "serve_web", None))

    def test_a_plot_configuration_turns_the_plots_on(self) -> None:
        config = plot_config_module.PlotConfig(signals=("x",))
        for sink in ("serve_web", "spawn"):
            with self.subTest(sink=sink):
                self.assertTrue(cli.plots_enabled(None, sink, config))

    def test_without_a_configuration_only_serve_web_is_off(self) -> None:
        self.assertFalse(cli.plots_enabled(None, "serve_web", None))
        self.assertTrue(cli.plots_enabled(None, "spawn", None))

    def test_the_shipped_launches_plot(self) -> None:
        # Every launch file runs serve_web with the shipped configuration, the bridge's default --plot_config.
        args = cli.build_parser().parse_args([])
        self.assertTrue(
            cli.plots_enabled(args.plots, "serve_web", _shipped_plot_config())
        )

    def test_the_help_says_so(self) -> None:
        text = _help("--plots")
        self.assertNotIn("off for serve_web, on otherwise", text)
        self.assertIn("plot configuration", text)


class TerminalStateFlagTest(unittest.TestCase):
    def test_the_help_names_the_plot_configurations_default(self) -> None:
        # The shipped configuration sets draw_terminal_state false, so the flag's effective default is off.
        self.assertEqual(
            plot_config_module.resolve_allowed_instances(
                config=_shipped_plot_config(),
                cli_robot_instances=None,
                cli_terminal_state=None,
            ),
            ("measured",),
        )
        text = _help("--terminal_state")
        self.assertNotIn("default: on", text)
        self.assertIn("draw_terminal_state", text)
        self.assertIn("plot configuration", _help("--robot_instances"))

    def test_default_is_none(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args([])
        self.assertIsNone(args.terminal_state)

    def test_flag_can_be_enabled_and_disabled(self) -> None:
        parser = cli.build_parser()
        self.assertTrue(parser.parse_args(["--terminal_state"]).terminal_state)
        self.assertFalse(
            parser.parse_args(["--no-terminal_state"]).terminal_state
        )


class RobotInstancesFlagTest(unittest.TestCase):
    def test_default_is_empty(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.robot_instances, "")

    def test_flag_parses_comma_separated_instances(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(
            ["--robot_instances", "measured,terminal_state"]
        )
        self.assertEqual(args.robot_instances, "measured,terminal_state")


class PlotConfigFlagTest(unittest.TestCase):
    def test_default_is_global_plot_config(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.plot_config, cli.DEFAULT_PLOT_CONFIG)

    def test_flag_can_be_custom_or_none(self) -> None:
        parser = cli.build_parser()
        self.assertEqual(
            parser.parse_args(
                ["--plot_config", "custom.textproto"]
            ).plot_config,
            "custom.textproto",
        )
        self.assertEqual(
            parser.parse_args(["--plot_config", "none"]).plot_config,
            "none",
        )


if __name__ == "__main__":
    unittest.main()
