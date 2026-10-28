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
#include "absl/types/span.h"

namespace ocs2::humanoid::multi_contact {

/**
 * The convex hull of `points`, counterclockwise from the lowest-x (then
 * lowest-y) vertex, without collinear vertices; points within `tolerance` [m]
 * of each other count as one. One vertex for coinciding points, two for
 * collinear ones, none for no points.
 */
std::vector<Eigen::Vector2d> convexHull(
    absl::Span<const Eigen::Vector2d> points, double tolerance);

/**
 * The signed distance of `point` to the boundary of the convex polygon `hull`
 * (counterclockwise, as convexHull() returns it), positive inside. A hull of
 * one or two vertices (a point or a segment) has no inside: the result is minus
 * the distance to it. Requires a non-empty hull.
 */
double signedDistanceToConvexPolygon(absl::Span<const Eigen::Vector2d> hull,
                                     const Eigen::Vector2d& point);

}  // namespace ocs2::humanoid::multi_contact
