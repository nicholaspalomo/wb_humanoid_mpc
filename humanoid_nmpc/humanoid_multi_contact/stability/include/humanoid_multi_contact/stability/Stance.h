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
#include <string>
#include <vector>

#include "Eigen/Core"
#include "absl/status/statusor.h"
#include "ocs2_core/Dimensions.h"

#include "humanoid_multi_contact/contact_set/ContactSet.h"

namespace ocs2::humanoid::multi_contact {

/** The edges of a linearized friction cone by default: a pyramid of eight
 * faces inscribed in the Coulomb cone (design section 2). */
inline constexpr int kDefaultFrictionConeEdges = 8;

/** The fewest edges a linearized friction cone may have. */
inline constexpr int kMinFrictionConeEdges = 3;

/** The number of corners of a surface contact's rectangle, each of which
 * carries a friction cone of its own. */
inline constexpr int kRectangleCorners = 4;

/** One active contact of a stance, in the world frame. Passive data. */
struct StanceContact {
  // The candidate's name, and its index in the ContactSet (for the Jacobians).
  std::string name;
  int candidateIndex = 0;
  ContactKind kind = ContactKind::kPoint;
  // [m] The pad's center.
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  // world_R_cone: z is the terrain's outward normal at the contact; a surface
  // contact's rectangle lies in the frame's (x, y) plane.
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  // The Coulomb friction coefficient, positive.
  double friction = 0.0;
  // Set exactly for a surface contact.
  std::optional<ContactRectangle> rectangle;
};

/** A set of active contacts. */
using Stance = std::vector<StanceContact>;

/** A wrench as a 6-vector: the force, then the moment. */
using Wrench = Eigen::Matrix<double, kWrenchDim, 1>;

/**
 * The generators of the linearized contact wrench cone of a stance: every
 * contact wrench the stance can apply is a non-negative combination of their
 * columns (Caron, Pham and Nakamura). A point contact contributes the edges of
 * its friction pyramid at its position; a surface contact the edges of a
 * pyramid at each corner of its rectangle, which spans the wrench cone of the
 * rectangle exactly: the friction, center-of-pressure and torsional limits.
 *
 * The edge j of a pyramid of n edges is n + mu cos(pi / n) (cos(2 pi j / n) t1
 * + sin(2 pi j / n) t2), inscribed in the Coulomb cone of friction mu, with a
 * normal component of one, so the sum of a contact's scalings is its normal
 * force. Passive data.
 */
struct WrenchGenerators {
  // [m] The point the moments are taken about: the mean of the contact
  // positions, so that the moments are of the size of the stance.
  Eigen::Vector3d reference = Eigen::Vector3d::Zero();
  // One column per generator: its unit force, then its moment about
  // `reference`.
  Eigen::Matrix<double, kWrenchDim, Eigen::Dynamic> wrenches;
  // Per generator: the index of its contact in the stance, its point of
  // application and its force direction, in the world frame.
  std::vector<int> contactIndices;
  std::vector<Eigen::Vector3d> points;
  std::vector<Eigen::Vector3d> directions;

  int size() const { return static_cast<int>(wrenches.cols()); }
};

/**
 * Builds the generators of `stance` with friction pyramids of `edges` edges.
 * InvalidArgument for fewer than kMinFrictionConeEdges edges, an empty stance,
 * a friction coefficient that is not positive, a rotation that is not one, or
 * a surface contact without a rectangle.
 */
absl::StatusOr<WrenchGenerators> buildWrenchGenerators(const Stance& stance,
                                                       int edges);

}  // namespace ocs2::humanoid::multi_contact
