"""The blueprint builds from the contracts, lays out every tab and panel, hides what is hidden by default, and survives a round trip
through an .rrd file."""

import os
import shutil
import tempfile
import unittest
from typing import Iterator, List

import rerun.blueprint as rrb

import rrd_contents
from humanoid_rerun_viewer import blueprint
from humanoid_rerun_viewer import bridge
from humanoid_rerun_viewer import scene_contract
from humanoid_rerun_viewer import telemetry_contract


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
            plotted: List[rrb.View] = list(views(container))
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
        expected = {blueprint.SCENE_VIEW_NAME, blueprint.EVENTS_VIEW_NAME} | {
            panel.title for tab in telemetry_contract.TABS for panel in tab.panels()
        }
        self.assertTrue(
            expected <= contents.view_names(), expected - contents.view_names()
        )
        self.assertIn([False], contents.blueprint_values("EntityBehavior:visible"))


if __name__ == "__main__":
    unittest.main()
