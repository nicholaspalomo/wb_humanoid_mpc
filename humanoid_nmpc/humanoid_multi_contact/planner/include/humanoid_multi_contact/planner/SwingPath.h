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

#include <vector>

#include "Eigen/Core"

#include "humanoid_multi_contact/terrain/TerrainModel.h"

namespace ocs2::humanoid::multi_contact {

/** Where a swinging pad starts and lands. Passive data. */
struct SwingEnds {
  // [m] The pad's position when the swing starts, and the outward normal of the
  // surface it rests on, if `lifts`.
  Eigen::Vector3d start = Eigen::Vector3d::Zero();
  Eigen::Vector3d startNormal = Eigen::Vector3d::UnitZ();
  // Whether the pad starts on a surface, so that it lifts off along its
  // normal first.
  bool lifts = false;
  // [m] Its target on the surface of outward normal `targetNormal`.
  Eigen::Vector3d target = Eigen::Vector3d::Zero();
  Eigen::Vector3d targetNormal = Eigen::Vector3d::UnitZ();
  // [m] How far the pad's geometry reaches from its center (a sole's half
  // diagonal), which the traverse keeps off the terrain too.
  double extent = 0.0;
};

/**
 * The path of a swinging pad (design section 7.2's generalized swing, here for
 * the planner): lift-off along the start surface's normal by the clearance,
 * a traverse to the point the clearance above the target, and touch-down along
 * the target's normal. When the straight traverse passes closer than half the
 * clearance plus the pad's extent to the terrain, the traverse is an arc
 * instead, the straight line
 * raised by 4 u (1 - u) H in z, with the lowest H in steps of the clearance
 * that keeps that distance (at most 0.6 m): it rises while it advances,
 * as a foot stepping onto a box does. Parameterized by the fraction s in [0, 1]
 * of its length. A value type.
 */
class SwingPath {
 public:
  /** The path between `ends` with `clearance` [m] above the surfaces, on
   * `terrain`. */
  static SwingPath Create(const SwingEnds& ends,
                          double clearance,
                          const TerrainModel& terrain);

  /** [m] The point at fraction `s` of the length, s clamped into [0, 1]. */
  Eigen::Vector3d pointAt(double s) const;
  /** [m] */
  double length() const { return cumulative_.back(); }
  /** The fraction at which the lift-off ends (0 without one) and the one at
   * which the touch-down starts. */
  double liftEnd() const { return liftEnd_; }
  double approachStart() const { return approachStart_; }
  const std::vector<Eigen::Vector3d>& waypoints() const { return waypoints_; }

 private:
  SwingPath() = default;

  std::vector<Eigen::Vector3d> waypoints_;
  // [m] The arc length at each waypoint; the first is 0.
  std::vector<double> cumulative_;
  double liftEnd_ = 0.0;
  double approachStart_ = 1.0;
};

}  // namespace ocs2::humanoid::multi_contact
