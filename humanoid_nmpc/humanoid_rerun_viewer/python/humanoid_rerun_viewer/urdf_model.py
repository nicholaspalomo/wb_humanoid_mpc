"""The visual geometry of a URDF, with its mesh files found on disk: what the bridge draws for every robot instance.

    model = load_urdf("robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf",
                      search_roots=default_search_roots(urdf_path))

A mesh named "package://<package>/<path>" is looked up in the directory of that package: the nearest directory named
<package> above the URDF (the URDF itself usually lives in it), or one directly inside a directory above the URDF,
or else a directory named <package> anywhere under one of the search roots (the repository's robot_models/). A
"file://" URI or a plain path is taken as is, relative to the URDF's directory.

Rerun draws STL, OBJ, glTF and GLB meshes (rerun.Asset3D). A mesh in another format (such as COLLADA .dae) is drawn
from a file of the same name in a supported format next to it, when there is one, and is otherwise skipped and listed
in RobotModel.unsupported_meshes. Boxes, cylinders and spheres are drawn as Rerun primitives.
"""

import dataclasses
import math
import os
import urllib.parse
import xml.etree.ElementTree as element_tree
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple, Union

from humanoid_rerun_viewer import palette

# The mesh formats Rerun's Asset3D draws, by lower-case file extension.
# LINT.IfChange(mesh_media_types)
MESH_MEDIA_TYPES: Dict[str, str] = {
    ".stl": "model/stl",
    ".obj": "model/obj",
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
}
# LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/README.md:mesh_media_types)

_PACKAGE_SCHEME = "package://"
_FILE_SCHEME = "file://"


class UrdfError(ValueError):
    """A URDF that cannot be read, naming the file and the element."""


@dataclasses.dataclass(frozen=True)
class Origin:
    """A pose relative to the link frame: translation [m] and unit quaternion (x, y, z, w)."""

    translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion_xyzw: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


@dataclasses.dataclass(frozen=True)
class Mesh:
    """A mesh file.

    Attributes:
        uri: the filename attribute as the URDF writes it.
        path: the file drawn: the resolved URI, or a file in a supported format next to it.
        media_type: of `path`, one of MESH_MEDIA_TYPES.
        scale: of the mesh along its axes.
    """

    uri: str
    path: str
    media_type: str
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclasses.dataclass(frozen=True)
class Box:
    size: Tuple[float, float, float]


@dataclasses.dataclass(frozen=True)
class Cylinder:
    """Along the z axis of its origin, as in URDF."""

    radius: float
    length: float


@dataclasses.dataclass(frozen=True)
class Sphere:
    radius: float


Geometry = Union[Mesh, Box, Cylinder, Sphere]


@dataclasses.dataclass(frozen=True)
class Visual:
    """One <visual> of a link.

    Attributes:
        link: the link's name.
        index: the visual's position among the link's <visual> elements that the bridge draws.
        origin: of the geometry in the link frame.
        geometry: what is drawn.
        color: the material color, RGBA in [0, 1]; None when the URDF gives none.
    """

    link: str
    index: int
    origin: Origin
    geometry: Geometry
    color: Optional[palette.Rgba]


@dataclasses.dataclass(frozen=True)
class RobotModel:
    """The links and the visual geometry of a URDF.

    Attributes:
        name: the robot's name attribute.
        urdf_path: the file read.
        links: every link, in document order.
        root_link: the link that is no joint's child.
        visuals: every visual the bridge draws, link by link in document order.
        unsupported_meshes: mesh URIs in a format Rerun cannot draw and without a drawable twin; skipped.
        missing_meshes: mesh URIs whose file was not found; skipped.
    """

    name: str
    urdf_path: str
    links: Tuple[str, ...]
    root_link: str
    visuals: Tuple[Visual, ...]
    unsupported_meshes: Tuple[str, ...] = ()
    missing_meshes: Tuple[str, ...] = ()

    def visuals_of(self, link: str) -> Tuple[Visual, ...]:
        return tuple(visual for visual in self.visuals if visual.link == link)

    def links_with_visuals(self) -> FrozenSet[str]:
        return frozenset(visual.link for visual in self.visuals)

    def mesh_paths(self) -> Tuple[str, ...]:
        """The distinct mesh files the visuals draw, in first-use order."""
        paths: Dict[str, None] = {}
        for visual in self.visuals:
            if isinstance(visual.geometry, Mesh):
                paths.setdefault(visual.geometry.path, None)
        return tuple(paths)


# ======================================================================================================================
# Finding the mesh files
# ======================================================================================================================


def _ancestors(directory: str) -> List[str]:
    """`directory` and every directory above it, nearest first."""
    result = []
    current = os.path.abspath(directory)
    while True:
        result.append(current)
        parent = os.path.dirname(current)
        if parent == current:
            return result
        current = parent


class PackageResolver:
    """Finds the directory of a ROS-style package and resolves mesh URIs; remembers what it found.

    Args:
        urdf_path: the URDF the URIs come from; relative paths and the upward search start at its directory.
        search_roots: directories searched recursively for a directory named after the package when none is found
            above the URDF.
    """

    def __init__(self, urdf_path: str, search_roots: Sequence[str] = ()) -> None:
        self._urdf_directory = os.path.dirname(os.path.abspath(urdf_path))
        self._search_roots = tuple(
            os.path.abspath(root) for root in search_roots if os.path.isdir(root)
        )
        self._packages: Dict[str, Optional[str]] = {}
        self._indexes: Dict[str, Dict[str, str]] = {}

    def package_directory(self, package: str) -> Optional[str]:
        """The directory of `package`, or None when neither the URDF's ancestors nor the search roots have it."""
        if package not in self._packages:
            self._packages[package] = self._find_package(package)
        return self._packages[package]

    def _find_package(self, package: str) -> Optional[str]:
        for directory in _ancestors(self._urdf_directory):
            if os.path.basename(directory) == package:
                return directory
            candidate = os.path.join(directory, package)
            if os.path.isdir(candidate):
                return candidate
        for root in self._search_roots:
            found = self._index(root).get(package)
            if found is not None:
                return found
        return None

    def _index(self, root: str) -> Dict[str, str]:
        """Every directory under `root` by name; the first one in sorted walk order wins a name."""
        if root not in self._indexes:
            index: Dict[str, str] = {}
            for directory, subdirectories, _ in os.walk(root, followlinks=True):
                subdirectories.sort()
                for subdirectory in subdirectories:
                    index.setdefault(
                        subdirectory, os.path.join(directory, subdirectory)
                    )
            self._indexes[root] = index
        return self._indexes[root]

    def resolve(self, uri: str) -> Optional[str]:
        """The file `uri` names, or None when its package is not found. The file itself may not exist."""
        if uri.startswith(_PACKAGE_SCHEME):
            package, _, relative = uri[len(_PACKAGE_SCHEME) :].partition("/")
            directory = self.package_directory(package)
            if directory is None:
                return None
            return os.path.join(directory, urllib.parse.unquote(relative))
        if uri.startswith(_FILE_SCHEME):
            path = urllib.parse.unquote(urllib.parse.urlparse(uri).path)
        else:
            path = uri
        return os.path.normpath(os.path.join(self._urdf_directory, path))


def media_type_of(path: str) -> Optional[str]:
    """The Rerun media type of a mesh file, or None for a format Rerun does not draw."""
    return MESH_MEDIA_TYPES.get(os.path.splitext(path)[1].lower())


def drawable_twin(path: str) -> Optional[str]:
    """A file next to `path` with the same name in a format Rerun draws (e.g. head.stl for head.dae), or None."""
    directory, filename = os.path.split(path)
    stem = os.path.splitext(filename)[0]
    try:
        entries = sorted(os.listdir(directory))
    except OSError:
        return None
    for extension in MESH_MEDIA_TYPES:
        for entry in entries:
            entry_stem, entry_extension = os.path.splitext(entry)
            if entry_stem == stem and entry_extension.lower() == extension:
                return os.path.join(directory, entry)
    return None


def default_search_roots(urdf_path: str) -> Tuple[str, ...]:
    """The robot_models/ directories to search packages in: the repository's (under `bazel run`), and any above the
    URDF or the current directory."""
    candidates = []
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if workspace:
        candidates.append(os.path.join(workspace, "robot_models"))
    for directory in _ancestors(os.path.dirname(os.path.abspath(urdf_path))):
        candidates.append(os.path.join(directory, "robot_models"))
    candidates.append(os.path.join(os.getcwd(), "robot_models"))
    roots: Dict[str, None] = {}
    for candidate in candidates:
        if os.path.isdir(candidate):
            roots.setdefault(os.path.abspath(candidate), None)
    return tuple(roots)


# ======================================================================================================================
# Parsing
# ======================================================================================================================


def quaternion_from_rpy(
    roll: float, pitch: float, yaw: float
) -> Tuple[float, float, float, float]:
    """The unit quaternion (x, y, z, w) of URDF's fixed-axis roll, pitch, yaw: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


class _Parser:
    def __init__(self, urdf_path: str, resolver: PackageResolver) -> None:
        self._urdf_path = urdf_path
        self._resolver = resolver
        self._materials: Dict[str, palette.Rgba] = {}
        self.unsupported_meshes: List[str] = []
        self.missing_meshes: List[str] = []

    def error(self, where: str, text: str) -> UrdfError:
        return UrdfError(f"{self._urdf_path}: {where}: {text}")

    def floats(
        self, element: element_tree.Element, attribute: str, count: int, where: str
    ) -> Optional[Tuple[float, ...]]:
        text = element.get(attribute)
        if text is None:
            return None
        try:
            values = tuple(float(value) for value in text.split())
        except ValueError:
            raise self.error(
                where, f'{attribute}="{text}" is not {count} numbers'
            ) from None
        if len(values) != count or not all(math.isfinite(value) for value in values):
            raise self.error(where, f'{attribute}="{text}" is not {count} numbers')
        return values

    def number(
        self, element: element_tree.Element, attribute: str, where: str
    ) -> float:
        values = self.floats(element, attribute, 1, where)
        if values is None:
            raise self.error(where, f"<{element.tag}> has no {attribute}")
        return values[0]

    def color(
        self, material: Optional[element_tree.Element], where: str
    ) -> Optional[palette.Rgba]:
        if material is None:
            return None
        color = material.find("color")
        if color is not None:
            rgba = self.floats(color, "rgba", 4, where)
            if rgba is not None:
                return rgba  # type: ignore[return-value]
        return self._materials.get(material.get("name", ""))

    def read_materials(self, robot: element_tree.Element) -> None:
        for material in robot.findall("material"):
            name = material.get("name", "")
            color = self.color(material, f"material '{name}'")
            if name and color is not None:
                self._materials[name] = color

    def origin(self, element: element_tree.Element, where: str) -> Origin:
        origin = element.find("origin")
        if origin is None:
            return Origin()
        xyz = self.floats(origin, "xyz", 3, where) or (0.0, 0.0, 0.0)
        rpy = self.floats(origin, "rpy", 3, where) or (0.0, 0.0, 0.0)
        return Origin(translation=xyz, quaternion_xyzw=quaternion_from_rpy(*rpy))  # type: ignore[arg-type]

    def mesh(self, mesh: element_tree.Element, where: str) -> Optional[Mesh]:
        uri = mesh.get("filename", "")
        if not uri:
            raise self.error(where, "<mesh> has no filename")
        scale = self.floats(mesh, "scale", 3, where) or (1.0, 1.0, 1.0)
        path = self._resolver.resolve(uri)
        if path is None or not os.path.isfile(path):
            self.missing_meshes.append(uri)
            return None
        media_type = media_type_of(path)
        if media_type is None:
            twin = drawable_twin(path)
            if twin is None:
                self.unsupported_meshes.append(uri)
                return None
            path, media_type = twin, media_type_of(twin)
        return Mesh(uri=uri, path=path, media_type=media_type, scale=scale)  # type: ignore[arg-type]

    def geometry(
        self, geometry: element_tree.Element, where: str
    ) -> Optional[Geometry]:
        shapes = list(geometry)
        if len(shapes) != 1:
            raise self.error(where, "<geometry> must hold exactly one shape")
        shape = shapes[0]
        if shape.tag == "mesh":
            return self.mesh(shape, where)
        if shape.tag == "box":
            size = self.floats(shape, "size", 3, where)
            if size is None:
                raise self.error(where, "<box> has no size")
            return Box(size=size)  # type: ignore[arg-type]
        if shape.tag == "cylinder":
            return Cylinder(
                radius=self.number(shape, "radius", where),
                length=self.number(shape, "length", where),
            )
        if shape.tag == "sphere":
            return Sphere(radius=self.number(shape, "radius", where))
        raise self.error(where, f"unknown geometry <{shape.tag}>")

    def visuals(self, link: element_tree.Element, name: str) -> List[Visual]:
        visuals: List[Visual] = []
        for position, visual in enumerate(link.findall("visual")):
            where = f"link '{name}', visual {position}"
            geometry_element = visual.find("geometry")
            if geometry_element is None:
                raise self.error(where, "<visual> has no <geometry>")
            geometry = self.geometry(geometry_element, where)
            if geometry is None:
                continue
            visuals.append(
                Visual(
                    link=name,
                    index=len(visuals),
                    origin=self.origin(visual, where),
                    geometry=geometry,
                    color=self.color(visual.find("material"), where),
                )
            )
        return visuals


def parse_urdf(
    text: str, urdf_path: str, search_roots: Sequence[str] = ()
) -> RobotModel:
    """The model of the URDF document `text`, whose file is `urdf_path` (for relative mesh paths and messages).

    Raises:
        UrdfError: the document is not a URDF the bridge can draw; the message names the file and the element.
    """
    try:
        robot = element_tree.fromstring(text)
    except element_tree.ParseError as error:
        raise UrdfError(f"{urdf_path}: not XML: {error}") from None
    if robot.tag != "robot":
        raise UrdfError(f"{urdf_path}: the root element is <{robot.tag}>, not <robot>")
    parser = _Parser(urdf_path, PackageResolver(urdf_path, search_roots))
    parser.read_materials(robot)

    links: List[str] = []
    visuals: List[Visual] = []
    for link in robot.findall("link"):
        name = link.get("name", "")
        if not name:
            raise parser.error(f"link {len(links)}", "<link> has no name")
        if name in links:
            raise parser.error(f"link '{name}'", "the link is defined twice")
        links.append(name)
        visuals.extend(parser.visuals(link, name))
    if not links:
        raise UrdfError(f"{urdf_path}: the robot has no links")

    children = {
        child.get("link", "")
        for joint in robot.findall("joint")
        for child in joint.findall("child")
    }
    roots = [link for link in links if link not in children]
    if not roots:
        raise UrdfError(
            f"{urdf_path}: every link is a joint's child, so there is no root link"
        )
    return RobotModel(
        name=robot.get("name", ""),
        urdf_path=urdf_path,
        links=tuple(links),
        root_link=roots[0],
        visuals=tuple(visuals),
        unsupported_meshes=tuple(parser.unsupported_meshes),
        missing_meshes=tuple(parser.missing_meshes),
    )


def load_urdf(path: str, search_roots: Sequence[str] = ()) -> RobotModel:
    """The model of the URDF file `path`.

    Raises:
        UrdfError: the file cannot be read or is not a URDF the bridge can draw.
    """
    try:
        with open(path, encoding="utf-8") as urdf:
            text = urdf.read()
    except OSError as error:
        raise UrdfError(f"{path}: cannot read the URDF: {error.strerror}") from None
    return parse_urdf(text, path, search_roots)
