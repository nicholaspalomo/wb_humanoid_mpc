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
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/Types.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact/ContactCenterPoint.h"

namespace ocs2::humanoid {

namespace {

/** `posCornersRtContactEwrtContact` scaled by `scaleFactor` about the contact
 * frame's origin. */
std::vector<vector3_t> computeScaledPosCornersRtContactEwrtContact(
    absl::Span<const vector3_t> posCornersRtContactEwrtContact,
    scalar_t scaleFactor) {
  std::vector<vector3_t> scaledPosCornersRtContactEwrtContact;
  scaledPosCornersRtContactEwrtContact.reserve(
      posCornersRtContactEwrtContact.size());
  for (const vector3_t& posCornerRtContactEwrtContact :
       posCornersRtContactEwrtContact) {
    scaledPosCornersRtContactEwrtContact.push_back(
        scaleFactor * posCornerRtContactEwrtContact);
  }
  return scaledPosCornersRtContactEwrtContact;
}

/** The x and y extent of `posCornersRtContactEwrtContact`; all zero for no
 * corners. */
PolygonBounds boundsOf(
    absl::Span<const vector3_t> posCornersRtContactEwrtContact) {
  if (posCornersRtContactEwrtContact.empty()) {
    return PolygonBounds(/*min_pos_corners_rt_contact_ewrt_contact_x=*/0.0,
                         /*max_pos_corners_rt_contact_ewrt_contact_x=*/0.0,
                         /*min_pos_corners_rt_contact_ewrt_contact_y=*/0.0,
                         /*max_pos_corners_rt_contact_ewrt_contact_y=*/0.0);
  }
  PolygonBounds bounds(posCornersRtContactEwrtContact.front()(kXAxisIndex),
                       posCornersRtContactEwrtContact.front()(kXAxisIndex),
                       posCornersRtContactEwrtContact.front()(kYAxisIndex),
                       posCornersRtContactEwrtContact.front()(kYAxisIndex));
  for (const vector3_t& posCornerRtContactEwrtContact :
       posCornersRtContactEwrtContact) {
    bounds.min_pos_corners_rt_contact_ewrt_contact_x =
        std::min(bounds.min_pos_corners_rt_contact_ewrt_contact_x,
                 posCornerRtContactEwrtContact(kXAxisIndex));
    bounds.max_pos_corners_rt_contact_ewrt_contact_x =
        std::max(bounds.max_pos_corners_rt_contact_ewrt_contact_x,
                 posCornerRtContactEwrtContact(kXAxisIndex));
    bounds.min_pos_corners_rt_contact_ewrt_contact_y =
        std::min(bounds.min_pos_corners_rt_contact_ewrt_contact_y,
                 posCornerRtContactEwrtContact(kYAxisIndex));
    bounds.max_pos_corners_rt_contact_ewrt_contact_y =
        std::max(bounds.max_pos_corners_rt_contact_ewrt_contact_y,
                 posCornerRtContactEwrtContact(kYAxisIndex));
  }
  return bounds;
}

}  // namespace

ContactPolygon::ContactPolygon(
    absl::Span<const vector3_t> posCornersRtContactEwrtContact,
    const ContactCenterPoint& contactCenterPoint,
    scalar_t scaleFactor)
    : ContactPolygon(computeScaledPosCornersRtContactEwrtContact(
                         posCornersRtContactEwrtContact, scaleFactor),
                     boundsOf(computeScaledPosCornersRtContactEwrtContact(
                         posCornersRtContactEwrtContact, scaleFactor)),
                     contactCenterPoint) {}

ContactPolygon::ContactPolygon(
    std::vector<vector3_t> posCornersRtContactEwrtContact,
    const PolygonBounds& polygonBounds,
    const ContactCenterPoint& contactCenterPoint)
    : polygonLimits_(polygonBounds),
      posCornersRtContactEwrtContact_(
          std::move(posCornersRtContactEwrtContact)),
      contactCenterPoint_(contactCenterPoint) {
  polygonPointFrameNames_.reserve(posCornersRtContactEwrtContact_.size());

  for (int i = 0; i < static_cast<int>(posCornersRtContactEwrtContact_.size());
       ++i) {
    polygonPointFrameNames_.push_back(
        absl::StrCat(contactCenterPoint.frameName, "_p_", i));
  }
}

matrix3_t ContactPolygon::getCrossProductMatrixPosCornerRtContactEwrtContact(
    int index) const {
  vector3_t posCornerRtContactEwrtContact =
      posCornersRtContactEwrtContact_[index];
  matrix3_t crossprodMat = matrix3_t::Zero();
  // [p]x of the point's projection (x, y, 0) on the sole's plane.
  crossprodMat(kXAxisIndex, kZAxisIndex) =
      posCornerRtContactEwrtContact[kYAxisIndex];
  crossprodMat(kYAxisIndex, kZAxisIndex) =
      -posCornerRtContactEwrtContact[kXAxisIndex];
  crossprodMat(kZAxisIndex, kXAxisIndex) =
      -posCornerRtContactEwrtContact[kYAxisIndex];
  crossprodMat(kZAxisIndex, kYAxisIndex) =
      posCornerRtContactEwrtContact[kXAxisIndex];
  return crossprodMat;
}

}  // namespace ocs2::humanoid
