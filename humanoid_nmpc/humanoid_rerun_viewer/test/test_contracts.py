"""The telemetry contract's panels plot series of its own groups, every group is plotted, and README.md tabulates it;
the scene and status contracts are consistent."""

import os
import unittest
from typing import Dict, Tuple

from humanoid_rerun_viewer import palette
from humanoid_rerun_viewer import scene_contract
from humanoid_rerun_viewer import status_contract
from humanoid_rerun_viewer import telemetry_contract

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(PACKAGE_DIR, "README.md")
SAMPLE_FRAME = "foot_l_contact"


def plotted_by(panel_path: str, group_path: str) -> bool:
    if panel_path.endswith("/**"):
        return group_path.startswith(panel_path[: -len("**")])
    return panel_path == group_path


class PanelTabsTest(unittest.TestCase):
    """Every panel of the panel tabs plots its group's series in its group's colors."""

    def test_the_tab_and_panel_titles_are_unique(self) -> None:
        titles = [tab.title for tab in telemetry_contract.PANEL_TABS]
        self.assertEqual(len(titles), len(set(titles)))
        for tab in telemetry_contract.PANEL_TABS:
            with self.subTest(tab=tab.title):
                self.assertTrue(tab.rows)
                panels = [panel.title for panel in tab.panels()]
                self.assertEqual(len(panels), len(set(panels)))

    def test_every_curve_is_a_series_of_a_group_its_panel_plots(self) -> None:
        for tab in telemetry_contract.PANEL_TABS:
            for panel in tab.panels():
                with self.subTest(panel=panel.title):
                    self.assertTrue(
                        panel.curves, "a panel of the panel tabs names its curves"
                    )
                    for curve in panel.curves:
                        group = telemetry_contract.find_group(curve.path)
                        self.assertIsNotNone(group, curve.path)
                        assert group is not None
                        self.assertIn(curve.path, panel.paths)
                        self.assertTrue(
                            group.has_series(curve.series),
                            f"'{curve.series}' is not a series of {curve.path}",
                        )

    def test_the_panel_groups_are_colored_from_the_palette(self) -> None:
        curve_colors = {color.lower() for color in palette_colors()}
        for group in telemetry_contract.PANEL_GROUPS:
            with self.subTest(group=group.path):
                self.assertEqual(len(group.colors), len(group.names))
                self.assertLessEqual(set(group.colors), curve_colors)

    def test_every_panel_group_is_reproduced_by_exactly_one_panel(self) -> None:
        plotted: Dict[str, int] = {}
        for tab in telemetry_contract.PANEL_TABS:
            for panel in tab.panels():
                for path in panel.paths:
                    plotted[path] = plotted.get(path, 0) + 1
        for group in telemetry_contract.PANEL_GROUPS:
            self.assertEqual(plotted.get(group.path), 1, group.path)


def palette_colors() -> Tuple[str, ...]:
    """Every curve color palette.py defines, as "#rrggbb"."""
    return tuple(
        value
        for name, value in vars(palette).items()
        if name.isupper() and isinstance(value, str) and value.startswith("#")
    )


class TelemetryGroupsTest(unittest.TestCase):
    def all_groups(self) -> Tuple[telemetry_contract.SeriesGroup, ...]:
        return telemetry_contract.all_groups([SAMPLE_FRAME, "pelvis"])

    def test_paths_are_unique_and_valid(self) -> None:
        paths = [group.path for group in self.all_groups()]
        self.assertEqual(len(paths), len(set(paths)))
        for path in paths:
            self.assertTrue(scene_contract.is_valid_relative_path(path), path)

    def test_names_colors_and_rules_are_consistent(self) -> None:
        for group in self.all_groups():
            with self.subTest(group=group.path):
                self.assertEqual(len(group.names), len(set(group.names)))
                if group.colors:
                    self.assertEqual(len(group.colors), len(group.names))
                    for color in group.colors:
                        self.assertRegex(color, r"^#[0-9a-f]{6}$")
                self.assertEqual(group.robot_dependent, bool(group.names_rule))
                self.assertTrue(group.description)

    def test_every_group_is_plotted_by_a_tab(self) -> None:
        panel_paths = [
            path
            for tab in telemetry_contract.TABS
            for panel in tab.panels()
            for path in panel.paths
        ]
        for group in self.all_groups():
            self.assertTrue(
                any(plotted_by(path, group.path) for path in panel_paths), group.path
            )

    def test_every_panel_plots_groups_of_the_contract(self) -> None:
        groups = self.all_groups()
        for tab in telemetry_contract.TABS:
            for panel in tab.panels():
                for path in panel.paths:
                    self.assertTrue(
                        any(plotted_by(path, group.path) for group in groups),
                        f"{panel.title}: {path}",
                    )

    def test_find_group_knows_every_frame_group_and_nothing_else(self) -> None:
        for group in telemetry_contract.frame_groups("any_frame"):
            self.assertEqual(telemetry_contract.find_group(group.path), group)
        for path in (
            "frames/pose/foot/target",
            "frames/velocity/foot/measured",
            "frames/pose/measured",
            "base_pose/position_w",
            "",
        ):
            self.assertIsNone(telemetry_contract.find_group(path), path)

    def test_complete_groups_carry_every_series_of_the_panels(self) -> None:
        # The panels duplicate values for plotting; the complete groups lose nothing the ROS-era topics had. Spot-check
        # the kinds of data: per DOF, per joint, per contact, per frame.
        complete = {
            group.path
            for group in self.all_groups()
            if group not in telemetry_contract.PANEL_GROUPS
        }
        for path in (
            "dofs/position/measured",
            "dofs/velocity/reference",
            "dofs/position/plan",
            "dofs/velocity/plan",
            "dofs/force/measured",
            "joints/effort/target",
            "contact_wrenches/left/mpc",
            "contact_wrenches/right/measured",
            f"frames/acceleration/{SAMPLE_FRAME}/reference",
            f"frames/wrench/{SAMPLE_FRAME}/measured",
            f"frames/pose/{SAMPLE_FRAME}/plan",
            "mpc_observation/state",
        ):
            self.assertIn(path, complete)


class ReadmeTest(unittest.TestCase):
    """README.md tabulates the contracts, so that the producer's author finds every path there."""

    @classmethod
    def setUpClass(cls) -> None:
        with open(README, encoding="utf-8") as readme:
            cls.text = readme.read()

    def assert_tabulated(self, path: str) -> None:
        self.assertIn(f"`{path}`", self.text, f"README.md does not list `{path}`")

    def test_every_static_group_is_listed(self) -> None:
        for group in telemetry_contract.STATIC_GROUPS:
            self.assert_tabulated(group.path)

    def test_the_frame_groups_are_listed(self) -> None:
        for kind in telemetry_contract.FRAME_KINDS:
            self.assert_tabulated(f"frames/{kind}/<frame>/<source>")

    def test_every_marker_instance_and_status_entity_is_listed(self) -> None:
        for marker in scene_contract.MARKERS:
            self.assert_tabulated(
                scene_contract.entity_path(scene_contract.WORLD_ROOT, marker.path)
            )
        for style in scene_contract.ROBOT_INSTANCES:
            self.assert_tabulated(scene_contract.instance_path(style.name))
        for series in status_contract.STATUS_SERIES:
            self.assert_tabulated(series.path)
        for path in (
            status_contract.FSM_STATE_LOG,
            status_contract.LOOP_TIMING_LOG,
            status_contract.MPC_STATUS_LOG,
            status_contract.BRIDGE_LOG,
        ):
            self.assert_tabulated(path)


class SceneContractTest(unittest.TestCase):
    def test_marker_paths_are_unique_and_valid(self) -> None:
        paths = [marker.path for marker in scene_contract.MARKERS]
        self.assertEqual(len(paths), len(set(paths)))
        for marker in scene_contract.MARKERS:
            self.assertTrue(scene_contract.is_valid_relative_path(marker.path))
            self.assertGreater(marker.default_radius, 0.0)
            self.assertTrue(marker.default_colors)

    def test_what_is_clutter_most_of_the_time_is_hidden_by_default(self) -> None:
        self.assertEqual(
            set(scene_contract.hidden_by_default()),
            {
                "world/robots/terminal_target",
                "world/markers/corner_forces",
                "world/markers/collision_spheres",
            },
        )

    def test_the_arrow_geometry_is_kept(self) -> None:
        # Rerun draws the shaft with half the radius and the head with all of it: a 0.01 m shaft and a 0.02 m head.
        self.assertAlmostEqual(scene_contract.ARROW_RADIUS, 0.01)
        self.assertAlmostEqual(2 * scene_contract.TRAJECTORY_RADIUS, 0.01)

    def test_unknown_instances_get_the_default_style_under_their_name(self) -> None:
        style = scene_contract.instance_style("ghost")
        self.assertEqual(style.name, "ghost")
        self.assertNotIn(style, scene_contract.ROBOT_INSTANCES)
        self.assertEqual(
            scene_contract.instance_style(scene_contract.MEASURED),
            scene_contract.ROBOT_INSTANCES[0],
        )

    def test_relative_paths(self) -> None:
        for path in ("markers/x", "a", "frames/pose/foot.l/measured", "a-b_c"):
            self.assertTrue(scene_contract.is_valid_relative_path(path), path)
        for path in ("", "/a", "a/", "a//b", "a/../b", "./a", "a b", "a\nb"):
            self.assertFalse(scene_contract.is_valid_relative_path(path), path)

    def test_status_series_are_under_the_status_root(self) -> None:
        paths = [series.path for series in status_contract.STATUS_SERIES]
        self.assertEqual(len(paths), len(set(paths)))
        for path in paths:
            self.assertTrue(path.startswith(scene_contract.STATUS_ROOT + "/"))


if __name__ == "__main__":
    unittest.main()
