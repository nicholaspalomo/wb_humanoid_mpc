/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************************************************************/

#pragma once

#include <optional>
#include <vector>

#include "Eigen/Core"

namespace ocs2::humanoid::multi_contact {

/** The six faces of a box, by the axis of the box frame their outward normal
 * points along. */
enum class BoxFace {
  kPositiveX,
  kNegativeX,
  kPositiveY,
  kNegativeY,
  kPositiveZ,
  kNegativeZ,
};

/** The number of faces of a box. */
inline constexpr int kBoxFaceCount = 6;

/**
 * The frame of a surface patch of the terrain, in which the friction cones and
 * the slip terms of a contact on it are written (design section 3.2): the
 * normal is the patch frame's z axis. Passive data.
 */
struct PatchFrame {
  // The primitive of the TerrainModel the patch lies on.
  int primitiveIndex = 0;
  // The box face, or std::nullopt for the tangent plane of a capsule.
  std::optional<BoxFace> face;
  // [m] The face's center, or the tangent point on a capsule, in the world.
  Eigen::Vector3d origin = Eigen::Vector3d::Zero();
  // world_R_patch: the columns are the two tangents and the outward normal, a
  // right-handed frame.
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  // [m] The face's polygon in the patch's (x, y) coordinates about origin,
  // counterclockwise seen from outside; empty for a capsule's tangent plane,
  // which has no edge.
  std::vector<Eigen::Vector2d> polygon;
  // The Coulomb friction coefficient of the primitive.
  double friction = 0.0;

  Eigen::Vector3d normal() const { return rotation.col(kNormalColumn); }

  // The column of `rotation` that holds the normal.
  static constexpr int kNormalColumn = 2;
};

}  // namespace ocs2::humanoid::multi_contact
