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

#include "humanoid_multi_contact/contact_set/ContactSet.h"
#include "humanoid_multi_contact/terrain/PatchFrame.h"
#include "humanoid_multi_contact/terrain/TerrainModel.h"

namespace ocs2::humanoid::multi_contact {

/** Where a candidate may rest: a point on a face of the terrain. Passive
 * data. */
struct ContactSample {
  int candidate = 0;
  // The primitive, and the box face, or std::nullopt on a capsule.
  int primitive = 0;
  std::optional<BoxFace> face;
  // [m] The pad's center in the world, on the surface.
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  // The surface's outward unit normal there.
  Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
  // A surface pad's x axis on the face (its yaw), orthogonal to `normal`.
  Eigen::Vector3d xAxis = Eigen::Vector3d::UnitX();
};

/** The defaults of ContactSamplerOptions; its fields say what each is. */
inline constexpr double kDefaultMinPointNormalZ = -0.2;
inline constexpr double kDefaultGridSpacing = 0.05;
inline constexpr double kDefaultEdgeMargin = 0.015;
inline constexpr double kDefaultPenetrationTolerance = 0.004;
inline constexpr double kDefaultMinSurfaceNormalZ = 0.7071067811865476;

/** The settings of sampleContacts(). */
struct ContactSamplerOptions {
  // [m] The grid on every face and along every capsule (design section 5.1).
  double gridSpacing = kDefaultGridSpacing;
  // [m] How far inside a face's edges a pad stays, beyond its own radius (a
  // point pad) or its rectangle (a surface pad): a 0.24 m sole fits a 0.28 m
  // tread with 1.5 cm at its toe and heel.
  double edgeMargin = kDefaultEdgeMargin;
  // [m] How far a sample's pad may reach into another primitive: none, but for
  // the 3.5 mm by which the smooth minimum lowers the distance at an inside
  // corner (TerrainModel).
  double penetrationTolerance = kDefaultPenetrationTolerance;
  // The smallest z component of the normal of a face a point pad may rest on
  // (no ceilings), and of one a surface pad may rest on (no walls: a foot,
  // the seat or the torso rests on a surface within 45 degrees of level).
  double minPointNormalZ = kDefaultMinPointNormalZ;
  double minSurfaceNormalZ = kDefaultMinSurfaceNormalZ;
};

/**
 * The samples of `candidate` on the faces of `terrain` within `reach` of
 * `center` (design section 5.1): on each box face whose normal the candidate's
 * kind accepts, the points of a grid of the face's frame, `gridSpacing` apart,
 * whose pad lies inside the face, `edgeMargin` from its edges (a point pad's
 * disc of its geom radius, a surface pad's rectangle at the yaw of `xAxisHint`
 * projected on the face); on each capsule, the points along its axis at the
 * same spacing, on its top and on its side facing `center`, for point pads
 * only. A sample whose pad would lie inside another primitive (a point under a
 * stair, a sole against a riser) is dropped. The order is deterministic: by
 * primitive, face, then grid.
 */
std::vector<ContactSample> sampleContacts(const TerrainModel& terrain,
                                          const ContactCandidate& candidate,
                                          int candidateIndex,
                                          const Eigen::Vector3d& center,
                                          double reach,
                                          const Eigen::Vector3d& xAxisHint,
                                          const ContactSamplerOptions& options);

/** `hint` projected on the plane of `normal` and normalized, or a tangent of
 * the plane when it is nearly the normal. */
Eigen::Vector3d tangentAxis(const Eigen::Vector3d& hint,
                            const Eigen::Vector3d& normal);

}  // namespace ocs2::humanoid::multi_contact
