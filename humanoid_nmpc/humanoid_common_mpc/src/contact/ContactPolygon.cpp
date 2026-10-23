/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

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

#include "humanoid_common_mpc/contact/ContactPolygon.h"

#include <algorithm>
#include <string>
#include <vector>

#include "ocs2_core/Dimensions.h"

namespace ocs2::humanoid {

namespace {

/** `points` scaled by `scaleFactor` about the frame's origin. */
std::vector<vector3_t> scaledPoints(const std::vector<vector3_t>& points, scalar_t scaleFactor) {
  std::vector<vector3_t> scaled;
  scaled.reserve(points.size());
  for (const vector3_t& point : points) scaled.push_back(scaleFactor * point);
  return scaled;
}

/** The x and y extent of `points`; all zero for no points. */
PolygonBounds boundsOf(const std::vector<vector3_t>& points) {
  if (points.empty()) return PolygonBounds(/*x_min=*/0.0, /*x_max=*/0.0, /*y_min=*/0.0, /*y_max=*/0.0);
  PolygonBounds bounds(points.front()(kXAxisIndex), points.front()(kXAxisIndex), points.front()(kYAxisIndex), points.front()(kYAxisIndex));
  for (const vector3_t& point : points) {
    bounds.x_min = std::min(bounds.x_min, point(kXAxisIndex));
    bounds.x_max = std::max(bounds.x_max, point(kXAxisIndex));
    bounds.y_min = std::min(bounds.y_min, point(kYAxisIndex));
    bounds.y_max = std::max(bounds.y_max, point(kYAxisIndex));
  }
  return bounds;
}

}  // namespace

ContactPolygon::ContactPolygon(const std::vector<vector3_t>& polygonPoints,
                               const ContactCenterPoint& contactCenterPoint,
                               scalar_t scaleFactor)
    : ContactPolygon(scaledPoints(polygonPoints, scaleFactor), boundsOf(scaledPoints(polygonPoints, scaleFactor)), contactCenterPoint) {}

ContactPolygon::ContactPolygon(const std::vector<vector3_t>& polygonPoints,
                               const PolygonBounds& polygonBounds,
                               const ContactCenterPoint& contactCenterPoint)
    : polygonLimits_(polygonBounds), polygonPoints_(polygonPoints), contactCenterPoint_(contactCenterPoint) {
  polygonPointFrameNames_.reserve(polygonPoints_.size());

  for (int i = 0; i < static_cast<int>(polygonPoints_.size()); ++i) {
    polygonPointFrameNames_.emplace_back(contactCenterPoint.frameName + "_p_" + std::to_string(i));
  }
}

matrix3_t ContactPolygon::getContactPointTranslationCrossProductMatrix(int index) const {
  vector3_t current_point = polygonPoints_[index];
  matrix3_t crossprodMat = matrix3_t::Zero();
  // [p]x of the point's projection (x, y, 0) on the sole's plane.
  crossprodMat(kXAxisIndex, kZAxisIndex) = current_point[kYAxisIndex];
  crossprodMat(kYAxisIndex, kZAxisIndex) = -current_point[kXAxisIndex];
  crossprodMat(kZAxisIndex, kXAxisIndex) = -current_point[kYAxisIndex];
  crossprodMat(kZAxisIndex, kYAxisIndex) = current_point[kXAxisIndex];
  return crossprodMat;
}

}  // namespace ocs2::humanoid
