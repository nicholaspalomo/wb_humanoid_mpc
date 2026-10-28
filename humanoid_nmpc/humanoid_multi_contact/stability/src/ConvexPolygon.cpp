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

#include "humanoid_multi_contact/stability/ConvexPolygon.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "absl/types/span.h"

namespace ocs2::humanoid::multi_contact {
namespace {

/** (b - a) x (c - a): positive when a, b, c turn counterclockwise. */
double turn(const Eigen::Vector2d& a,
            const Eigen::Vector2d& b,
            const Eigen::Vector2d& c) {
  const Eigen::Vector2d ab = b - a;
  const Eigen::Vector2d ac = c - a;
  return ab.x() * ac.y() - ab.y() * ac.x();
}

double distanceToSegment(const Eigen::Vector2d& a,
                         const Eigen::Vector2d& b,
                         const Eigen::Vector2d& point) {
  const Eigen::Vector2d ab = b - a;
  const double squaredLength = ab.squaredNorm();
  if (squaredLength == 0.0) return (point - a).norm();
  const double fraction =
      std::clamp((point - a).dot(ab) / squaredLength, 0.0, 1.0);
  return (point - (a + fraction * ab)).norm();
}

// The vertices of the chain a turn is taken from, besides the next point.
constexpr size_t kTurnVertices = 2;
// The fewest vertices of a polygon with an inside.
constexpr size_t kPolygonMinVertices = 3;

/** `polygon` (convex, counterclockwise) without the vertices within
 * `tolerance` of the line through their two neighbors. */
std::vector<Eigen::Vector2d> withoutCollinearVertices(
    std::vector<Eigen::Vector2d> polygon, double tolerance) {
  bool removed = true;
  while (removed && polygon.size() > kTurnVertices) {
    removed = false;
    for (size_t i = 0; i < polygon.size(); ++i) {
      const Eigen::Vector2d& previous =
          polygon[(i + polygon.size() - 1) % polygon.size()];
      const Eigen::Vector2d& next = polygon[(i + 1) % polygon.size()];
      const double base = (next - previous).norm();
      if (turn(previous, polygon[i], next) <= tolerance * base) {
        polygon.erase(polygon.begin() + static_cast<ptrdiff_t>(i));
        removed = true;
        break;
      }
    }
  }
  return polygon;
}

bool lexicographicLess(const Eigen::Vector2d& left,
                       const Eigen::Vector2d& right) {
  return left.x() < right.x() ||
         (left.x() == right.x() && left.y() < right.y());
}

}  // namespace

std::vector<Eigen::Vector2d> convexHull(
    absl::Span<const Eigen::Vector2d> points, double tolerance) {
  std::vector<Eigen::Vector2d> sorted(points.begin(), points.end());
  std::sort(sorted.begin(), sorted.end(), lexicographicLess);
  std::vector<Eigen::Vector2d> unique;
  for (const Eigen::Vector2d& point : sorted) {
    const bool duplicate =
        std::any_of(unique.begin(), unique.end(),
                    [&point, tolerance](const Eigen::Vector2d& kept) {
                      return (kept - point).norm() <= tolerance;
                    });
    if (!duplicate) unique.push_back(point);
  }
  if (unique.size() <= 1) return unique;

  // Andrew's monotone chain, with the exact turn test, which is consistent
  // with the lexicographic order of the points as they are; then the vertices
  // within `tolerance` of the line through their neighbors are dropped.
  std::vector<Eigen::Vector2d> chain;
  chain.reserve(unique.size() + unique.size());
  for (const Eigen::Vector2d& point : unique) {
    while (chain.size() >= kTurnVertices &&
           !(turn(chain[chain.size() - kTurnVertices], chain.back(), point) >
             0.0)) {
      chain.pop_back();
    }
    chain.push_back(point);
  }
  const size_t lowerSize = chain.size() + 1;
  for (size_t i = unique.size() - 1; i-- > 0;) {
    while (chain.size() >= lowerSize &&
           !(turn(chain[chain.size() - kTurnVertices], chain.back(),
                  unique[i]) > 0.0)) {
      chain.pop_back();
    }
    chain.push_back(unique[i]);
  }
  // The last vertex is the first one again.
  chain.pop_back();
  return withoutCollinearVertices(chain, tolerance);
}

double signedDistanceToConvexPolygon(absl::Span<const Eigen::Vector2d> hull,
                                     const Eigen::Vector2d& point) {
  if (hull.size() == 1) return -(point - hull[0]).norm();
  double boundaryDistance = std::numeric_limits<double>::infinity();
  bool inside = hull.size() >= kPolygonMinVertices;
  for (size_t i = 0; i < hull.size(); ++i) {
    const Eigen::Vector2d& a = hull[i];
    const Eigen::Vector2d& b = hull[(i + 1) % hull.size()];
    boundaryDistance =
        std::min(boundaryDistance, distanceToSegment(a, b, point));
    if (turn(a, b, point) < 0.0) inside = false;
  }
  return inside ? boundaryDistance : -boundaryDistance;
}

}  // namespace ocs2::humanoid::multi_contact
