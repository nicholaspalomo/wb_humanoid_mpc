"""URDF parsing and package:// resolution: a sample package with every feature, and every URDF of robot_models/."""

import math
import os
import random
import shutil
import tempfile
import unittest
from typing import List

import synthetic_messages
from humanoid_rerun_viewer import urdf_model

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.normpath(os.path.join(PACKAGE_DIR, "..", ".."))
ROBOT_MODELS = os.path.join(REPO_ROOT, "robot_models")
SHIPPED_URDFS = (
    "drc_atlas/drc_atlas_description/urdf/atlas.urdf",
    "engineai_sa01/engineai_sa01_description/urdf/zq_sa01.urdf",
    "unitree_g1/g1_description/urdf/g1_29dof.urdf",
    "unitree_r1/unitree_r1_description/urdf/R1.urdf",
)


def repository_urdfs() -> List[str]:
    """Every robot_models/<robot>/<package>/urdf/*.urdf in the runfiles."""
    found = []
    for directory, _, filenames in os.walk(ROBOT_MODELS, followlinks=True):
        if os.path.basename(directory) != "urdf":
            continue
        found.extend(
            os.path.join(directory, name)
            for name in filenames
            if name.endswith(".urdf")
        )
    return sorted(found)


class TemporaryDirectoryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory)

    def write(self, relative_path: str, text: str) -> str:
        path = os.path.join(self.directory, relative_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as output:
            output.write(text)
        return path


class SampleUrdfTest(TemporaryDirectoryTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.urdf_path = synthetic_messages.write_sample_package(self.directory)
        self.model = urdf_model.load_urdf(self.urdf_path)

    def test_links_and_root(self) -> None:
        self.assertEqual(self.model.name, "sample_robot")
        self.assertEqual(self.model.links, synthetic_messages.SAMPLE_LINKS)
        self.assertEqual(self.model.root_link, "base_link")
        self.assertEqual(
            sorted(self.model.links_with_visuals()),
            sorted(synthetic_messages.link_names_with_visuals()),
        )

    def test_mesh_with_scale_origin_and_named_material(self) -> None:
        (visual,) = self.model.visuals_of("base_link")
        mesh = visual.geometry
        assert isinstance(mesh, urdf_model.Mesh)
        self.assertTrue(mesh.path.endswith(os.path.join("meshes", "base.STL")))
        self.assertTrue(os.path.isfile(mesh.path))
        self.assertEqual(mesh.media_type, "model/stl")
        self.assertEqual(mesh.scale, (0.001, 0.001, 0.001))
        self.assertEqual(visual.origin.translation, (0.0, 0.0, 0.1))
        for actual, expected in zip(
            visual.origin.quaternion_xyzw,
            (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)),
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(visual.color, (0.5, 0.5, 0.6, 1.0))

    def test_primitives_and_inline_material(self) -> None:
        box, cylinder = self.model.visuals_of("thigh")
        self.assertEqual(box.geometry, urdf_model.Box(size=(0.1, 0.2, 0.3)))
        self.assertEqual(box.color, (1.0, 0.0, 0.0, 1.0))
        self.assertEqual(
            cylinder.geometry, urdf_model.Cylinder(radius=0.05, length=0.4)
        )
        self.assertIsNone(cylinder.color)
        self.assertEqual((box.index, cylinder.index), (0, 1))
        (sphere,) = self.model.visuals_of("shin")
        self.assertEqual(sphere.geometry, urdf_model.Sphere(radius=0.03))

    def test_a_collada_mesh_is_drawn_from_its_stl_twin(self) -> None:
        (visual,) = self.model.visuals_of("foot")
        mesh = visual.geometry
        assert isinstance(mesh, urdf_model.Mesh)
        self.assertTrue(mesh.uri.endswith("foot.dae"))
        self.assertTrue(mesh.path.endswith("foot.stl"))
        self.assertEqual(mesh.media_type, "model/stl")
        self.assertEqual(self.model.unsupported_meshes, ())
        self.assertEqual(self.model.missing_meshes, ())

    def test_mesh_paths_are_distinct_and_in_first_use_order(self) -> None:
        self.assertEqual(
            [os.path.basename(path) for path in self.model.mesh_paths()],
            ["base.STL", "foot.stl"],
        )


class PackageResolutionTest(TemporaryDirectoryTestCase):
    def urdf_with_mesh(self, relative_path: str, uri: str) -> str:
        return self.write(
            relative_path,
            f'<robot name="r"><link name="a"><visual><geometry><mesh filename="{uri}"/></geometry></visual>'
            "</link></robot>",
        )

    def mesh(self, relative_path: str) -> str:
        path = os.path.join(self.directory, relative_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        synthetic_messages.write_binary_stl(path)
        return path

    def resolved(self, urdf_path: str, search_roots=()) -> str:
        model = urdf_model.load_urdf(urdf_path, search_roots)
        self.assertEqual(model.missing_meshes, ())
        (visual,) = model.visuals
        assert isinstance(visual.geometry, urdf_model.Mesh)
        return visual.geometry.path

    def test_the_package_the_urdf_lives_in(self) -> None:
        mesh = self.mesh("ws/pkg/meshes/a.stl")
        urdf = self.urdf_with_mesh("ws/pkg/urdf/r.urdf", "package://pkg/meshes/a.stl")
        self.assertEqual(self.resolved(urdf), mesh)

    def test_a_package_next_to_a_directory_above_the_urdf(self) -> None:
        mesh = self.mesh("ws/pkg/meshes/a.stl")
        urdf = self.urdf_with_mesh(
            "ws/elsewhere/urdf/r.urdf", "package://pkg/meshes/a.stl"
        )
        self.assertEqual(self.resolved(urdf), mesh)

    def test_a_package_under_a_search_root(self) -> None:
        mesh = self.mesh("models/robot/deep/pkg/meshes/a.stl")
        urdf = self.urdf_with_mesh("other/r.urdf", "package://pkg/meshes/a.stl")
        self.assertEqual(
            self.resolved(urdf, [os.path.join(self.directory, "models")]), mesh
        )

    def test_a_package_nowhere_is_a_missing_mesh(self) -> None:
        urdf = self.urdf_with_mesh("other/r.urdf", "package://nowhere/meshes/a.stl")
        model = urdf_model.load_urdf(urdf)
        self.assertEqual(model.missing_meshes, ("package://nowhere/meshes/a.stl",))
        self.assertEqual(model.visuals, ())

    def test_file_uris_and_relative_paths(self) -> None:
        mesh = self.mesh("ws/meshes/a.stl")
        self.assertEqual(
            self.resolved(self.urdf_with_mesh("ws/urdf/r1.urdf", f"file://{mesh}")),
            mesh,
        )
        self.assertEqual(
            self.resolved(self.urdf_with_mesh("ws/urdf/r2.urdf", "../meshes/a.stl")),
            mesh,
        )

    def test_a_format_rerun_cannot_draw_without_a_twin_is_skipped(self) -> None:
        self.write("ws/pkg/meshes/a.dae", "<COLLADA/>")
        urdf = self.urdf_with_mesh("ws/pkg/urdf/r.urdf", "package://pkg/meshes/a.dae")
        model = urdf_model.load_urdf(urdf)
        self.assertEqual(model.unsupported_meshes, ("package://pkg/meshes/a.dae",))
        self.assertEqual(model.visuals, ())

    def test_the_resolver_remembers_packages(self) -> None:
        self.mesh("ws/pkg/meshes/a.stl")
        urdf = self.urdf_with_mesh("ws/pkg/urdf/r.urdf", "package://pkg/meshes/a.stl")
        resolver = urdf_model.PackageResolver(urdf)
        first = resolver.package_directory("pkg")
        shutil.rmtree(os.path.join(self.directory, "ws", "pkg", "meshes"))
        self.assertEqual(resolver.package_directory("pkg"), first)
        self.assertIsNone(resolver.package_directory("other_pkg"))


class MalformedUrdfTest(TemporaryDirectoryTestCase):
    def assert_error(self, text: str, expected: str) -> None:
        path = self.write("r.urdf", text)
        with self.assertRaises(urdf_model.UrdfError) as context:
            urdf_model.load_urdf(path)
        self.assertIn(path, str(context.exception))
        self.assertIn(expected, str(context.exception))

    def test_errors_name_the_file_and_the_element(self) -> None:
        self.assert_error("<robot", "not XML")
        self.assert_error("<sdf/>", "not <robot>")
        self.assert_error('<robot name="r"/>', "no links")
        self.assert_error(
            '<robot><link name="a"/><link name="a"/></robot>', "defined twice"
        )
        self.assert_error(
            '<robot><link name="a"><visual/></link></robot>', "link 'a', visual 0"
        )
        self.assert_error(
            '<robot><link name="a"><visual><origin xyz="1 2"/><geometry><box size="1 1 1"/></geometry>'
            "</visual></link></robot>",
            'xyz="1 2" is not 3 numbers',
        )
        self.assert_error(
            '<robot><link name="a"><visual><geometry><capsule/></geometry></visual></link></robot>',
            "unknown geometry <capsule>",
        )
        self.assert_error(
            '<robot><link name="a"><visual><geometry><mesh/></geometry></visual></link></robot>',
            "<mesh> has no filename",
        )
        self.assert_error(
            '<robot><link name="a"><visual><geometry><sphere/></geometry></visual></link></robot>',
            "<sphere> has no radius",
        )
        self.assert_error(
            '<robot><link name="a"><visual><geometry><box size="1 nan 1"/></geometry></visual></link>'
            "</robot>",
            "is not 3 numbers",
        )
        self.assert_error(
            '<robot><link name="a"/><link name="b"/><joint name="j"><parent link="a"/><child link="b"/>'
            '</joint><joint name="k"><parent link="b"/><child link="a"/></joint></robot>',
            "no root link",
        )

    def test_a_missing_file(self) -> None:
        path = os.path.join(self.directory, "missing.urdf")
        with self.assertRaises(urdf_model.UrdfError) as context:
            urdf_model.load_urdf(path)
        self.assertIn(path, str(context.exception))


class QuaternionTest(unittest.TestCase):
    def test_identity_and_quarter_turns(self) -> None:
        self.assertEqual(urdf_model.quaternion_from_rpy(0.0, 0.0, 0.0), (0, 0, 0, 1))
        half = math.sqrt(0.5)
        for rpy, expected in (
            ((math.pi / 2, 0.0, 0.0), (half, 0.0, 0.0, half)),
            ((0.0, math.pi / 2, 0.0), (0.0, half, 0.0, half)),
            ((0.0, 0.0, math.pi / 2), (0.0, 0.0, half, half)),
        ):
            for actual, value in zip(urdf_model.quaternion_from_rpy(*rpy), expected):
                self.assertAlmostEqual(actual, value)

    def test_fixed_axis_order_rz_ry_rx(self) -> None:
        # Rotating the x axis by roll then pitch then yaw about the fixed axes.
        generator = random.Random(4)
        for _ in range(20):
            roll, pitch, yaw = (generator.uniform(-math.pi, math.pi) for _ in range(3))
            x, y, z, w = urdf_model.quaternion_from_rpy(roll, pitch, yaw)
            self.assertAlmostEqual(x * x + y * y + z * z + w * w, 1.0)
            # First column of the rotation matrix of the quaternion, against Rz Ry Rx's.
            column = (
                1 - 2 * (y * y + z * z),
                2 * (x * y + z * w),
                2 * (x * z - y * w),
            )
            expected = (
                math.cos(yaw) * math.cos(pitch),
                math.sin(yaw) * math.cos(pitch),
                -math.sin(pitch),
            )
            for actual, value in zip(column, expected):
                self.assertAlmostEqual(actual, value)


class RepositoryUrdfTest(unittest.TestCase):
    """Every URDF the repository ships parses, and every one of its meshes resolves to a file Rerun draws."""

    def test_the_shipped_urdfs_are_found(self) -> None:
        found = repository_urdfs()
        for relative_path in SHIPPED_URDFS:
            self.assertIn(os.path.join(ROBOT_MODELS, relative_path), found)

    def test_every_mesh_resolves(self) -> None:
        for path in repository_urdfs():
            with self.subTest(urdf=os.path.relpath(path, ROBOT_MODELS)):
                model = urdf_model.load_urdf(
                    path, urdf_model.default_search_roots(path)
                )
                self.assertEqual(model.missing_meshes, ())
                self.assertEqual(model.unsupported_meshes, ())
                self.assertIn(model.root_link, model.links)
                meshes = [
                    visual.geometry
                    for visual in model.visuals
                    if isinstance(visual.geometry, urdf_model.Mesh)
                ]
                self.assertTrue(meshes)
                for mesh in meshes:
                    self.assertTrue(os.path.isfile(mesh.path), mesh.uri)
                    self.assertIn(mesh.media_type, urdf_model.MESH_MEDIA_TYPES.values())

    def test_the_search_roots_hold_robot_models(self) -> None:
        path = os.path.join(ROBOT_MODELS, SHIPPED_URDFS[0])
        roots = urdf_model.default_search_roots(path)
        self.assertIn(os.path.abspath(ROBOT_MODELS), roots)


if __name__ == "__main__":
    unittest.main()
