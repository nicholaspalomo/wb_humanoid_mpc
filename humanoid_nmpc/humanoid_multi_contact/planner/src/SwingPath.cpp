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

#include "humanoid_multi_contact/planner/SwingPath.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "ocs2_core/Dimensions.h"

namespace ocs2::humanoid::multi_contact {
namespace {

// [m] Waypoints closer than this are one.
constexpr double kSamePoint = 1.0e-6;
// [m] The spacing at which the straight traverse is checked against the
// terrain.
constexpr double kTraverseCheckSpacing = 0.02;
// An arc over the terrain is drawn with this many segments.
constexpr int kArcPoints = 12;
// [m] The highest an arc rises above its straight line.
constexpr double kHighestArc = 0.6;

/** Whether the segment from `from` to `to` keeps `clearance` from the
 * terrain. */
bool keepsClear(const Eigen::Vector3d& from,
                const Eigen::Vector3d& to,
                double clearance,
                const TerrainModel& terrain) {
  const int steps = std::max(
      1,
      static_cast<int>(std::ceil((to - from).norm() / kTraverseCheckSpacing)));
  for (int k = 0; k <= steps; ++k) {
    const Eigen::Vector3d point =
        from + (to - from) * static_cast<double>(k) / steps;
    if (terrain.signedDistance(point) < clearance) return false;
  }
  return true;
}

/** The point at `u` of the arc from `from` to `to` raised by `height` in its
 * middle. */
Eigen::Vector3d arcPoint(const Eigen::Vector3d& from,
                         const Eigen::Vector3d& to,
                         double height,
                         double u) {
  // The arc's bump 4 u (1 - u) is 1 in its middle.
  constexpr double kBumpScale = 4.0;
  return from + u * (to - from) +
         kBumpScale * u * (1.0 - u) * height * Eigen::Vector3d::UnitZ();
}

/** Whether the arc from `from` to `to` raised by `height` keeps `clearance`
 * from the terrain. */
bool arcKeepsClear(const Eigen::Vector3d& from,
                   const Eigen::Vector3d& to,
                   double height,
                   double clearance,
                   const TerrainModel& terrain) {
  const int steps = std::max(
      kArcPoints,
      static_cast<int>(std::ceil((to - from).norm() / kTraverseCheckSpacing)));
  for (int k = 0; k <= steps; ++k) {
    if (terrain.signedDistance(arcPoint(
            from, to, height, static_cast<double>(k) / steps)) < clearance) {
      return false;
    }
  }
  return true;
}

}  // namespace

SwingPath SwingPath::Create(const SwingEnds& ends,
                            double clearance,
                            const TerrainModel& terrain) {
  const Eigen::Vector3d lifted =
      ends.lifts ? Eigen::Vector3d(ends.start + clearance * ends.startNormal)
                 : ends.start;
  const Eigen::Vector3d above = ends.target + clearance * ends.targetNormal;
  std::vector<Eigen::Vector3d> points = {ends.start, lifted};
  const double keep = 0.5 * clearance + ends.extent;
  if (!keepsClear(lifted, above, keep, terrain)) {
    // The lowest arc lerp(lifted, above, u) + 4 u (1 - u) H e_z that keeps
    // the clearance, H rising by the clearance at a time.
    double height = clearance;
    for (; height < kHighestArc; height += clearance) {
      if (arcKeepsClear(lifted, above, height, keep, terrain)) {
        break;
      }
    }
    for (int k = 1; k < kArcPoints; ++k) {
      points.push_back(
          arcPoint(lifted, above, height, static_cast<double>(k) / kArcPoints));
    }
  }
  points.push_back(above);
  points.push_back(ends.target);

  SwingPath path;
  for (const Eigen::Vector3d& point : points) {
    if (!path.waypoints_.empty() &&
        (point - path.waypoints_.back()).norm() < kSamePoint) {
      continue;
    }
    path.cumulative_.push_back(
        path.waypoints_.empty() ? 0.0
                                : path.cumulative_.back() +
                                      (point - path.waypoints_.back()).norm());
    path.waypoints_.push_back(point);
  }
  const double length = path.cumulative_.back();
  if (length < kSamePoint) return path;
  path.liftEnd_ = (lifted - ends.start).norm() / length;
  path.approachStart_ = 1.0 - (ends.target - above).norm() / length;
  return path;
}

Eigen::Vector3d SwingPath::pointAt(double s) const {
  const double distance = std::clamp(s, 0.0, 1.0) * length();
  for (size_t i = 1; i < waypoints_.size(); ++i) {
    if (distance <= cumulative_[i]) {
      const double span = cumulative_[i] - cumulative_[i - 1];
      const double fraction =
          span < kSamePoint ? 1.0 : (distance - cumulative_[i - 1]) / span;
      return waypoints_[i - 1] + fraction * (waypoints_[i] - waypoints_[i - 1]);
    }
  }
  return waypoints_.back();
}

}  // namespace ocs2::humanoid::multi_contact
