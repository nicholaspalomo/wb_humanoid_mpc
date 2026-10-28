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

"""Draws the terrain of a multi-contact scenario in Rerun (humanoid_nmpc/docs/multi_contact_planning/README.md, 3.2).

The boxes and capsules of a TerrainScene are logged statically under TERRAIN_ROOT, one entity per primitive by its
name: what the plan playback (humanoid_multi_contact/app) and the bridge of a multi-contact dummy sim
(`--multi_contact_scenario`) draw the scene with.
"""

from collections.abc import Sequence
import importlib
import math
from typing import TYPE_CHECKING

import numpy as np
import rerun as rr

from humanoid_rerun_viewer import palette
import nproto_textproto

if TYPE_CHECKING:
    from humanoid_multi_contact.terrain import terrain_scene_pb2

# The generated module of the scenario schema, imported when a scenario is drawn: a runfiles tree whose generated
# __init__.py files make humanoid_multi_contact a regular package (that of the launch files' test, whose GUI binary
# creates them) cannot import it, and the bridge starts there too, for its flags.
_SCENARIO_MODULE = "humanoid_multi_contact.scenarios.scenario_pb2"

TERRAIN_ROOT = "world/terrain"
_TERRAIN_COLOR = palette.to_rgba8(
    palette.with_alpha(palette.DEFAULT_MESH_COLOR, 0.6)
)
# Below this sine two directions count as parallel.
_PARALLEL_TOLERANCE = 1e-9
_IDENTITY_XYZW = (0.0, 0.0, 0.0, 1.0)


def quaternion_between(
    source: Sequence[float] | np.ndarray, target: Sequence[float] | np.ndarray
) -> list[float]:
    """The shortest rotation from the direction `source` to `target`, as (x, y, z, w).

    Args:
        source: A nonzero vector.
        target: A nonzero vector.

    Returns:
        The unit quaternion's four coefficients.
    """
    a = np.asarray(source, dtype=float)
    b = np.asarray(target, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    axis = np.cross(a, b)
    sine = float(np.linalg.norm(axis))
    cosine = float(np.dot(a, b))
    if sine < _PARALLEL_TOLERANCE:
        if cosine > 0.0:
            return list(_IDENTITY_XYZW)
        # Opposite: half a turn about an axis orthogonal to both.
        helper = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(a, helper))) > 1.0 - _PARALLEL_TOLERANCE:
            helper = np.array([0.0, 1.0, 0.0])
        orthogonal = np.cross(a, helper)
        orthogonal /= np.linalg.norm(orthogonal)
        return [
            float(orthogonal[0]),
            float(orthogonal[1]),
            float(orthogonal[2]),
            0.0,
        ]
    angle = math.atan2(sine, cosine)
    half_sine = math.sin(angle / 2.0) / sine
    return [
        float(axis[0] * half_sine),
        float(axis[1] * half_sine),
        float(axis[2] * half_sine),
        math.cos(angle / 2.0),
    ]


def log_terrain(
    recording: rr.RecordingStream, scene: "terrain_scene_pb2.TerrainScene"
) -> int:
    """Logs every primitive of `scene` under TERRAIN_ROOT, statically.

    Args:
        recording: Where the primitives go.
        scene: The terrain.

    Returns:
        How many primitives it logged.
    """
    for primitive in scene.primitives:
        path = f"{TERRAIN_ROOT}/{primitive.name}"
        if primitive.HasField("box"):
            box = primitive.box
            orientation = list(box.orientation_xyzw) or list(_IDENTITY_XYZW)
            recording.log(
                path,
                rr.Boxes3D(
                    centers=[list(box.center)],
                    half_sizes=[list(box.half_extents)],
                    quaternions=[rr.Quaternion(xyzw=orientation)],
                    colors=[_TERRAIN_COLOR],
                    fill_mode=rr.components.FillMode.Solid,
                ),
                static=True,
            )
        else:
            capsule = primitive.capsule
            start = np.asarray(capsule.start, dtype=float)
            axis = np.asarray(capsule.end, dtype=float) - start
            length = float(np.linalg.norm(axis))
            orientation = list(_IDENTITY_XYZW)
            if length > 0.0:
                orientation = quaternion_between([0.0, 0.0, 1.0], axis)
            recording.log(
                path,
                rr.Capsules3D(
                    lengths=[length],
                    radii=[capsule.radius],
                    translations=[start.tolist()],
                    quaternions=[rr.Quaternion(xyzw=orientation)],
                    colors=[_TERRAIN_COLOR],
                ),
                static=True,
            )
    return len(scene.primitives)


def load_scenario_terrain(path: str) -> "terrain_scene_pb2.TerrainScene":
    """The terrain of the scenario textproto at `path` (humanoid_multi_contact/scenarios/scenario.proto).

    Args:
        path: The scenario file.

    Returns:
        Its terrain_scene.

    Raises:
        OSError: the file cannot be read.
        ValueError: it does not parse.
    """
    scenario_pb2 = importlib.import_module(_SCENARIO_MODULE)
    scenario = nproto_textproto.load_textproto(path, scenario_pb2.Scenario)
    return scenario.terrain_scene
