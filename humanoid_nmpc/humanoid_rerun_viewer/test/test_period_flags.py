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
import robot_ipc

INVALID_PERIODS = (0.0, -1.0, math.nan)


def _network() -> robot_ipc.NetworkConfig:
    """A network of one loopback node, which the rejected options never reach."""
    return robot_ipc.NetworkConfig(
        nodes=(
            robot_ipc.NodeEndpoint("publisher", "127.0.0.1", robot_ipc.EPHEMERAL_PORT),
        )
    )


class BusBridgePeriodTest(unittest.TestCase):
    def test_the_periods_must_be_positive(self) -> None:
        the_bridge = mock.create_autospec(bridge.RerunBridge, instance=True)
        for period in INVALID_PERIODS:
            with self.subTest(option="flush_period", period=period):
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    bus_bridge.BusBridge(the_bridge, _network(), flush_period=period)
            with self.subTest(option="report_period", period=period):
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    bus_bridge.BusBridge(the_bridge, _network(), report_period=period)

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
                self.assertIn("--flush_period must be positive", errors.getvalue())


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
                    "--max_scene_frequency must be non-negative", errors.getvalue()
                )


class MeasuredOnlyFlagTest(unittest.TestCase):
    def test_default_is_false(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args([])
        self.assertFalse(args.measured_only)

    def test_flag_can_be_enabled_and_disabled(self) -> None:
        parser = cli.build_parser()
        self.assertTrue(parser.parse_args(["--measured_only"]).measured_only)
        self.assertFalse(parser.parse_args(["--no-measured_only"]).measured_only)


class TerminalStateFlagTest(unittest.TestCase):
    def test_default_is_none(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args([])
        self.assertIsNone(args.terminal_state)

    def test_flag_can_be_enabled_and_disabled(self) -> None:
        parser = cli.build_parser()
        self.assertTrue(parser.parse_args(["--terminal_state"]).terminal_state)
        self.assertFalse(parser.parse_args(["--no-terminal_state"]).terminal_state)


class RobotInstancesFlagTest(unittest.TestCase):
    def test_default_is_empty(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.robot_instances, "")

    def test_flag_parses_comma_separated_instances(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(["--robot_instances", "measured,terminal_state"])
        self.assertEqual(args.robot_instances, "measured,terminal_state")


class PlotConfigFlagTest(unittest.TestCase):
    def test_default_is_global_plot_config(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.plot_config, cli.DEFAULT_PLOT_CONFIG)

    def test_flag_can_be_custom_or_none(self) -> None:
        parser = cli.build_parser()
        self.assertEqual(
            parser.parse_args(["--plot_config", "custom.textproto"]).plot_config,
            "custom.textproto",
        )
        self.assertEqual(
            parser.parse_args(["--plot_config", "none"]).plot_config,
            "none",
        )


if __name__ == "__main__":
    unittest.main()
