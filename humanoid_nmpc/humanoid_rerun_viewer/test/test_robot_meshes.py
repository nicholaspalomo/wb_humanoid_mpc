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

"""The static robot model of every instance: one Asset3D per mesh, the URDF primitives, tints and opacities."""

import os
import shutil
import tempfile
import unittest

import pyarrow as pa
import pyarrow.compute as pc

from humanoid_rerun_viewer import bridge
from humanoid_rerun_viewer import palette
from humanoid_rerun_viewer import robot_meshes
from humanoid_rerun_viewer import scene_contract
from humanoid_rerun_viewer import urdf_model
import rrd_contents
import synthetic_messages

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.normpath(os.path.join(PACKAGE_DIR, "..", ".."))
SA01_URDF = os.path.join(
    REPO_ROOT,
    "robot_models",
    "engineai_sa01",
    "engineai_sa01_description",
    "urdf",
    "zq_sa01.urdf",
)


def blob_size(contents: rrd_contents.RrdContents, path: str) -> int:
    """The size of the first Asset3D blob of `path`."""
    blobs = contents.entities[path].columns("Asset3D:blob")[0].flatten()
    if pa.types.is_list(blobs.type) or pa.types.is_large_list(blobs.type):
        return pc.list_value_length(blobs)[0].as_py()
    return pc.binary_length(blobs)[0].as_py()


def packed(rgba8) -> int:
    """Rerun's Rgba32: 0xRRGGBBAA."""
    red, green, blue, alpha = rgba8
    return (red << 24) | (green << 16) | (blue << 8) | alpha


class RecordingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory)
        self.rrd_path = os.path.join(self.directory, "recording.rrd")
        self.recording = bridge.new_recording("test_robot_meshes")
        self.recording.save(self.rrd_path)

    def contents(self) -> rrd_contents.RrdContents:
        self.recording.flush()
        self.recording.disconnect()
        return rrd_contents.RrdContents(self.rrd_path)


class SampleModelTest(RecordingTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.model = urdf_model.load_urdf(
            synthetic_messages.write_sample_package(self.directory)
        )
        self.meshes = robot_meshes.RobotMeshes(self.model)

    def test_every_visual_of_every_instance_is_static(self) -> None:
        for style in scene_contract.ROBOT_INSTANCES:
            result = self.meshes.log_instance(self.recording, style)
            self.assertEqual((result.meshes, result.primitives), (2, 3))
        contents = self.contents()
        for style in scene_contract.ROBOT_INSTANCES:
            for visual in self.model.visuals:
                path = scene_contract.visual_path(style.name, visual.link, visual.index)
                with self.subTest(path=path):
                    entity = contents.entities[path]
                    self.assertEqual(entity.temporal_components, set())
                    self.assertIn("Transform3D:translation", entity.static_components)
                    self.assertIn("Transform3D:quaternion", entity.static_components)
            # The link entities carry only the poses of viz/scene, never static data that would hide them.
            for link in self.model.links:
                self.assertNotIn(
                    scene_contract.link_path(style.name, link), contents.entities
                )
        base = contents.entities[
            scene_contract.visual_path(scene_contract.MEASURED, "base_link", 0)
        ]
        self.assertTrue(
            {"Asset3D:blob", "Asset3D:media_type", "Asset3D:albedo_factor"}
            <= base.static_components
        )
        self.assertEqual(base.values("Asset3D:media_type"), [["model/stl"]])
        ((scale,),) = base.values("Transform3D:scale")
        for value in scale:
            self.assertAlmostEqual(value, 0.001)
        thigh_box = contents.entities[
            scene_contract.visual_path(scene_contract.MEASURED, "thigh", 0)
        ]
        self.assertIn("Boxes3D:half_sizes", thigh_box.static_components)
        thigh_cylinder = contents.entities[
            scene_contract.visual_path(scene_contract.MEASURED, "thigh", 1)
        ]
        self.assertIn("Cylinders3D:lengths", thigh_cylinder.static_components)
        shin = contents.entities[
            scene_contract.visual_path(scene_contract.MEASURED, "shin", 0)
        ]
        self.assertIn("Ellipsoids3D:half_sizes", shin.static_components)

    def test_the_instances_are_tinted_at_their_opacity(self) -> None:
        for style in scene_contract.ROBOT_INSTANCES:
            self.meshes.log_instance(self.recording, style)
        contents = self.contents()
        for style in scene_contract.ROBOT_INSTANCES:
            base = contents.entities[
                scene_contract.visual_path(style.name, "base_link", 0)
            ]
            expected = robot_meshes.visual_color(self.model.visuals[0], style)
            self.assertEqual(base.values("Asset3D:albedo_factor"), [[packed(expected)]])
            self.assertEqual(expected[3], round(style.alpha * 255))

    def test_each_mesh_file_is_read_once(self) -> None:
        for style in scene_contract.ROBOT_INSTANCES:
            self.meshes.log_instance(self.recording, style)
        self.assertEqual(
            sorted(self.meshes._mesh_contents),
            sorted(self.model.mesh_paths()),
        )


class VisualColorTest(unittest.TestCase):
    def visual(self, geometry, color=None) -> urdf_model.Visual:
        return urdf_model.Visual(
            link="a",
            index=0,
            origin=urdf_model.Origin(),
            geometry=geometry,
            color=color,
        )

    def test_the_tint_wins_over_the_material(self) -> None:
        style = scene_contract.instance_style(scene_contract.TERMINAL_STATE)
        visual = self.visual(urdf_model.Sphere(radius=1.0), color=(1.0, 0.0, 0.0, 1.0))
        self.assertEqual(
            robot_meshes.visual_color(visual, style),
            palette.to_rgba8(palette.with_alpha(palette.BLUE, style.alpha)),
        )

    def test_the_material_at_the_instance_opacity(self) -> None:
        style = scene_contract.instance_style(scene_contract.MEASURED)
        visual = self.visual(urdf_model.Sphere(radius=1.0), color=(1.0, 0.0, 0.0, 0.5))
        self.assertEqual(
            robot_meshes.visual_color(visual, style),
            palette.to_rgba8((1.0, 0.0, 0.0, 0.5 * style.alpha)),
        )

    def test_without_a_material(self) -> None:
        style = scene_contract.instance_style(scene_contract.MEASURED)
        stl = urdf_model.Mesh(uri="a.stl", path="a.stl", media_type="model/stl")
        glb = urdf_model.Mesh(uri="a.glb", path="a.glb", media_type="model/gltf-binary")
        self.assertEqual(
            robot_meshes.visual_color(self.visual(stl), style),
            palette.to_rgba8(
                palette.with_alpha(palette.DEFAULT_MESH_COLOR, style.alpha)
            ),
        )
        # A glTF carries its own colors, which a gray albedo would darken.
        self.assertEqual(
            robot_meshes.visual_color(self.visual(glb), style)[:3], (255, 255, 255)
        )


class ShippedRobotTest(RecordingTestCase):
    """The EngineAI SA01 in full: every mesh of every instance is an Asset3D in the recording."""

    def test_every_mesh_of_every_instance(self) -> None:
        model = urdf_model.load_urdf(
            SA01_URDF, urdf_model.default_search_roots(SA01_URDF)
        )
        meshes = robot_meshes.RobotMeshes(model)
        # The mesh file of each mesh visual, by the end of its entity path.
        mesh_files: dict[str, str] = {}
        for visual in model.visuals:
            if isinstance(visual.geometry, urdf_model.Mesh):
                mesh_files[f"/{visual.link}/visual_{visual.index}"] = (
                    visual.geometry.path
                )
        self.assertTrue(mesh_files)
        for style in scene_contract.ROBOT_INSTANCES:
            result = meshes.log_instance(self.recording, style)
            self.assertEqual(result.meshes, len(mesh_files))
        contents = self.contents()
        assets = [
            path
            for path, entity in contents.entities.items()
            if "Asset3D:blob" in entity.static_components
        ]
        self.assertEqual(
            len(assets), len(mesh_files) * len(scene_contract.ROBOT_INSTANCES)
        )
        for path in assets:
            self.assertTrue(path.startswith(scene_contract.ROBOTS_ROOT + "/"), path)
            with open(
                next(
                    mesh_file_path
                    for suffix, mesh_file_path in mesh_files.items()
                    if path.endswith(suffix)
                ),
                "rb",
            ) as mesh_file:
                size = len(mesh_file.read())
            self.assertEqual(blob_size(contents, path), size)


if __name__ == "__main__":
    unittest.main()
