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


"""Tests that the DRC Atlas's collision model copies the collision geoms of atlas.xml (design section 5.4).

Every link names the MJCF body its geometry comes from; the body's joint must be the link's parent joint (the pelvis's
free joint is "root_joint"), and the link's shape must be one of the body's collision geoms, number for number.
"""

import unittest
from xml.etree import ElementTree

from humanoid_multi_contact.collision import collision_model_pb2

import nproto_textproto

_MJCF = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.xml"
_MODEL = (
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/multi_contact/"
    "collision_model.textproto"
)
# [m] How far a copied number may be from the MJCF's.
_TOLERANCE = 1e-9


def _numbers(text: str) -> list[float]:
    """The numbers of an MJCF attribute such as "0 0 0.1"."""
    return [float(value) for value in text.split()]


def _close(first: list[float], second: list[float]) -> bool:
    """Whether two lists of numbers are equal within _TOLERANCE."""
    return len(first) == len(second) and all(
        abs(a - b) <= _TOLERANCE for a, b in zip(first, second)
    )


def _bodies() -> dict[str, ElementTree.Element]:
    """The MJCF's bodies by name."""
    root = ElementTree.parse(_MJCF).getroot()
    return {body.get("name", ""): body for body in root.iter("body")}


def _collision_geoms(body: ElementTree.Element) -> list[ElementTree.Element]:
    """The collision geoms of `body` itself, not of its children."""
    return [
        geom
        for geom in body.findall("geom")
        if geom.get("class") == "collision"
    ]


def _joint_name(body: ElementTree.Element) -> str:
    """The name of the body's joint, "root_joint" for the free joint."""
    if body.find("freejoint") is not None:
        return "root_joint"
    joint = body.find("joint")
    return "" if joint is None else joint.get("name", "")


def _matches(
    link: collision_model_pb2.CollisionModel.Link,
    geoms: list[ElementTree.Element],
) -> bool:
    """Whether the link's shape is the body's collision geometry."""
    shape = link.WhichOneof("shape")
    if shape == "capsule":
        expected = list(link.capsule.start) + list(link.capsule.end)
        return any(
            geom.get("type") == "capsule"
            and _close(_numbers(geom.get("fromto", "")), expected)
            and _close(_numbers(geom.get("size", "")), [link.capsule.radius])
            for geom in geoms
        )
    if shape == "box":
        return any(
            geom.get("type") == "box"
            and _close(
                _numbers(geom.get("pos", "0 0 0")), list(link.box.center)
            )
            and _close(
                _numbers(geom.get("size", "")), list(link.box.half_extents)
            )
            for geom in geoms
        )
    spheres = [geom for geom in geoms if geom.get("type") == "sphere"]
    if len(spheres) != len(link.spheres.spheres):
        return False
    return all(
        any(
            _close(_numbers(geom.get("pos", "0 0 0")), list(sphere.center))
            and _close(_numbers(geom.get("size", "")), [sphere.radius])
            for geom in spheres
        )
        for sphere in link.spheres.spheres
    )


class CollisionModelMjcfTest(unittest.TestCase):
    """The collision model against atlas.xml."""

    def test_every_link_copies_its_body(self) -> None:
        model = nproto_textproto.load_textproto(
            _MODEL, collision_model_pb2.CollisionModel
        )
        bodies = _bodies()
        self.assertGreater(len(model.links), 0)
        for link in model.links:
            with self.subTest(link=link.name):
                self.assertIn(link.mjcf_body, bodies)
                body = bodies[link.mjcf_body]
                self.assertEqual(_joint_name(body), link.parent_joint)
                self.assertTrue(
                    _matches(link, _collision_geoms(body)),
                    f"{link.name} is not a geom of {link.mjcf_body}",
                )

    def test_a_changed_number_is_caught(self) -> None:
        model = nproto_textproto.load_textproto(
            _MODEL, collision_model_pb2.CollisionModel
        )
        bodies = _bodies()
        link = next(
            link
            for link in model.links
            if link.WhichOneof("shape") == "capsule"
        )
        link.capsule.radius += 0.01
        self.assertFalse(
            _matches(link, _collision_geoms(bodies[link.mjcf_body]))
        )


if __name__ == "__main__":
    unittest.main()
