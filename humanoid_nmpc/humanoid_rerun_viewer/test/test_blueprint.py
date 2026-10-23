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

"""Tests for blueprint.py.

The blueprint builds from the contracts, lays out every tab and panel, hides what is hidden by default, and survives a
round trip through an .rrd file.
"""

from collections.abc import Iterator
import os
import shutil
import tempfile
import unittest

import rerun.blueprint as rrb

from humanoid_rerun_viewer import blueprint
from humanoid_rerun_viewer import bridge
from humanoid_rerun_viewer import plot_config
from humanoid_rerun_viewer import scene_contract
from humanoid_rerun_viewer import telemetry_contract
import rrd_contents


def views(container) -> Iterator[rrb.View]:
    """Every view under a container, depth first."""
    for item in container.contents:
        if isinstance(item, rrb.View):
            yield item
        else:
            yield from views(item)


class BlueprintStructureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.blueprint = blueprint.build_blueprint(tracked_link="pelvis")
        root = self.blueprint.root_container
        self.scene_column, self.tabs = root.contents

    def test_the_scene_and_the_events_share_the_left_column(self) -> None:
        scene, events = self.scene_column.contents
        self.assertIsInstance(scene, rrb.Spatial3DView)
        self.assertEqual(scene.name, blueprint.SCENE_VIEW_NAME)
        self.assertEqual(scene.origin, "/world")
        self.assertIsInstance(events, rrb.TextLogView)
        self.assertEqual(events.origin, "/status")

    def test_blueprint_without_plots(self) -> None:
        bp = blueprint.build_blueprint(tracked_link="pelvis", plots=False)
        root = bp.root_container
        self.assertIsInstance(root, rrb.Vertical)
        scene, events = root.contents
        self.assertIsInstance(scene, rrb.Spatial3DView)
        self.assertIsInstance(events, rrb.TextLogView)

    def test_blueprint_with_plot_config(self) -> None:
        config = plot_config.PlotConfig(
            signals=("base_pose/position_x", "status/mpc/solve_time")
        )
        bp = blueprint.build_blueprint(tracked_link="pelvis", config=config)
        root = bp.root_container
        self.assertIsInstance(root, rrb.Horizontal)
        scene_col, tabs_container = root.contents
        self.assertIsInstance(scene_col, rrb.Vertical)
        self.assertIsInstance(tabs_container, rrb.Tabs)
        self.assertEqual(
            [tab.name for tab in tabs_container.contents],
            ["Base Pose & Euler", blueprint.STATUS_TAB_NAME],
        )

    def test_blueprint_hides_unallowed_robot_instances(self) -> None:
        bp = blueprint.build_blueprint(
            tracked_link="pelvis",
            allowed_instances=(scene_contract.MEASURED,),
        )
        root = bp.root_container
        self.assertIsInstance(root, rrb.Horizontal)
        scene_col, _ = root.contents
        scene, _ = scene_col.contents
        self.assertIsInstance(scene, rrb.Spatial3DView)
        terminal_path = "/" + scene_contract.instance_path(
            scene_contract.TERMINAL_STATE
        )
        self.assertIn(terminal_path, scene.visualizer_overrides)
        behavior = scene.visualizer_overrides[terminal_path]
        self.assertIsInstance(behavior, rrb.EntityBehavior)
        self.assertIn(f"- {terminal_path}/**", scene.contents)

    def test_the_tabs_are_the_panel_tabs_then_the_complete_and_status_tabs(
        self,
    ) -> None:
        self.assertIsInstance(self.tabs, rrb.Tabs)
        self.assertEqual(
            [tab.name for tab in self.tabs.contents],
            [tab.title for tab in telemetry_contract.TABS]
            + [blueprint.STATUS_TAB_NAME],
        )

    def test_every_panel_is_a_view_of_its_groups(self) -> None:
        for tab, container in zip(telemetry_contract.TABS, self.tabs.contents):
            plotted: list[rrb.View] = list(views(container))
            self.assertEqual(
                [view.name for view in plotted],
                [panel.title for panel in tab.panels()],
            )
            for panel, view in zip(tab.panels(), plotted):
                self.assertIsInstance(view, rrb.TimeSeriesView)
                self.assertEqual(
                    list(view.contents),
                    [f"+ /telemetry/{path}" for path in panel.paths],
                )

    def test_rows_follow_the_rows_of_the_panel_tabs(self) -> None:
        for tab, container in zip(telemetry_contract.PANEL_TABS, self.tabs.contents):
            self.assertEqual(
                [len(row.contents) for row in container.contents],
                [len(row) for row in tab.rows],
            )

    def test_what_is_hidden_by_default_is_hidden_in_the_scene(self) -> None:
        scene = self.scene_column.contents[0]
        self.assertEqual(
            set(scene.visualizer_overrides),
            {f"/{path}" for path in scene_contract.hidden_by_default()},
        )
        for behavior in scene.visualizer_overrides.values():
            self.assertIsInstance(behavior, rrb.EntityBehavior)

    def test_the_eye_starts_above_the_ground_looking_at_the_origin(self) -> None:
        x, y, z = blueprint.initial_eye_position()
        # About 2.45 m from a focal point near the origin, above the ground.
        self.assertGreater(z, 0.5)
        self.assertAlmostEqual(
            ((x - 0.6088) ** 2 + (y - 0.0910) ** 2 + (z - 0.0075) ** 2) ** 0.5,
            2.4503,
            places=3,
        )

    def test_without_a_tracked_link_the_blueprint_builds_too(self) -> None:
        self.assertIsNotNone(blueprint.build_blueprint(tracked_link=None))


class BlueprintRoundTripTest(unittest.TestCase):
    def test_the_blueprint_is_written_with_every_view(self) -> None:
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory)
        path = os.path.join(directory, "blueprint.rrd")
        recording = bridge.new_recording("test_blueprint")
        recording.save(path)
        recording.send_blueprint(blueprint.build_blueprint("pelvis"))
        recording.flush()
        recording.disconnect()
        contents = rrd_contents.RrdContents(path)
        self.assertTrue(contents.blueprints)
        expected = {blueprint.SCENE_VIEW_NAME, blueprint.EVENTS_VIEW_NAME}
        for tab in telemetry_contract.TABS:
            expected.update(panel.title for panel in tab.panels())
        self.assertTrue(
            expected <= contents.view_names(), expected - contents.view_names()
        )
        self.assertIn([False], contents.blueprint_values("EntityBehavior:visible"))


if __name__ == "__main__":
    unittest.main()
