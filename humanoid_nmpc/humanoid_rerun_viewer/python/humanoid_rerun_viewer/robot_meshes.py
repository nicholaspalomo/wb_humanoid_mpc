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

"""Logs the URDF's visual geometry once per robot instance, as static data that the link poses of viz/scene then move.

For every visual of every link, the entity world/robots/<instance>/<link>/visual_<i> gets a static Transform3D (the
visual's origin in the link frame, and the mesh scale) and its geometry: a rerun.Asset3D for a mesh, a Boxes3D,
Cylinders3D or Ellipsoids3D for a URDF primitive. The link entity itself (world/robots/<instance>/<link>) only ever
receives the temporal link poses, never static data, which would hide them.
"""

import dataclasses

import rerun as rr

from humanoid_rerun_viewer import palette
from humanoid_rerun_viewer import scene_contract
from humanoid_rerun_viewer import urdf_model

# Mesh formats that carry their own colors; without a URDF material they are drawn untinted.
_SELF_COLORED_MEDIA_TYPES = frozenset(
    {"model/obj", "model/gltf-binary", "model/gltf+json"}
)
_WHITE: palette.Rgb = (1.0, 1.0, 1.0)


@dataclasses.dataclass
class InstanceLogResult:
    """What log_instance() logged."""

    meshes: int = 0
    primitives: int = 0

    @property
    def visuals(self) -> int:
        return self.meshes + self.primitives


def visual_color(
    visual: urdf_model.Visual, style: scene_contract.RobotInstanceStyle
) -> palette.Rgba8:
    """Picks the color a visual of the instance is drawn with, at the instance's opacity.

    Args:
        visual: the URDF visual.
        style: the robot instance it is drawn for.

    Returns:
        The instance's tint, else the URDF material's color, else white for a mesh that carries its own colors, else
        the default mesh color.
    """
    if style.tint is not None:
        rgb: palette.Rgb = style.tint
        alpha = style.alpha
    elif visual.color is not None:
        rgb = visual.color[:3]
        alpha = style.alpha * visual.color[3]
    elif (
        isinstance(visual.geometry, urdf_model.Mesh)
        and visual.geometry.media_type in _SELF_COLORED_MEDIA_TYPES
    ):
        rgb, alpha = _WHITE, style.alpha
    else:
        rgb, alpha = palette.DEFAULT_MESH_COLOR, style.alpha
    return palette.to_rgba8(palette.with_alpha(rgb, alpha))


class RobotMeshes:
    """The visual geometry of one URDF, ready to log for any number of instances; reads each mesh file once."""

    def __init__(self, model: urdf_model.RobotModel) -> None:
        self._model = model
        self._mesh_contents: dict[str, bytes] = {}

    @property
    def model(self) -> urdf_model.RobotModel:
        return self._model

    def _contents(self, path: str) -> bytes:
        contents = self._mesh_contents.get(path)
        if contents is None:
            with open(path, "rb") as mesh:
                contents = mesh.read()
            self._mesh_contents[path] = contents
        return contents

    def log_instance(
        self,
        recording: rr.RecordingStream,
        style: scene_contract.RobotInstanceStyle,
    ) -> InstanceLogResult:
        """Logs every visual of the model under world/robots/<style.name>, statically.

        Args:
            recording: where the visuals go.
            style: the robot instance, which names the entities and colors the visuals.

        Returns:
            How many meshes and primitives it logged.

        Raises:
            OSError: a mesh file can no longer be read.
        """
        result = InstanceLogResult()
        for visual in self._model.visuals:
            path = scene_contract.visual_path(style.name, visual.link, visual.index)
            color = visual_color(visual, style)
            geometry = visual.geometry
            scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
            if isinstance(geometry, urdf_model.Mesh):
                scale = geometry.scale
            recording.log(
                path,
                rr.Transform3D(
                    translation=list(visual.origin.translation),
                    quaternion=rr.Quaternion(xyzw=list(visual.origin.quaternion_xyzw)),
                    scale=list(scale),
                ),
                static=True,
            )
            if isinstance(geometry, urdf_model.Mesh):
                recording.log(
                    path,
                    rr.Asset3D(
                        contents=self._contents(geometry.path),
                        media_type=geometry.media_type,
                        albedo_factor=color,
                    ),
                    static=True,
                )
                result.meshes += 1
                continue
            if isinstance(geometry, urdf_model.Box):
                archetype = rr.Boxes3D(
                    half_sizes=[[size / 2.0 for size in geometry.size]],
                    colors=[color],
                    fill_mode=rr.components.FillMode.Solid,
                )
            elif isinstance(geometry, urdf_model.Cylinder):
                archetype = rr.Cylinders3D(
                    lengths=[geometry.length],
                    radii=[geometry.radius],
                    colors=[color],
                    fill_mode=rr.components.FillMode.Solid,
                )
            else:
                archetype = rr.Ellipsoids3D(
                    half_sizes=[[geometry.radius] * 3],
                    colors=[color],
                    fill_mode=rr.components.FillMode.Solid,
                )
            recording.log(path, archetype, static=True)
            result.primitives += 1
        return result
