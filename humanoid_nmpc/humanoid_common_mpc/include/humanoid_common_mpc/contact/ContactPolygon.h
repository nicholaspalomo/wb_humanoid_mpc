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

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "ocs2_core/Types.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact/ContactCenterPoint.h"

namespace ocs2::humanoid {

/// \brief the maximum extension of the polygon with respect to the specified
/// frame
struct PolygonBounds {
  PolygonBounds(scalar_t min_pos_corners_rt_contact_ewrt_contact_x,
                scalar_t max_pos_corners_rt_contact_ewrt_contact_x,
                scalar_t min_pos_corners_rt_contact_ewrt_contact_y,
                scalar_t max_pos_corners_rt_contact_ewrt_contact_y,
                scalar_t scaleFactor = 1.0)
      : min_pos_corners_rt_contact_ewrt_contact_x(
            min_pos_corners_rt_contact_ewrt_contact_x * scaleFactor),
        max_pos_corners_rt_contact_ewrt_contact_x(
            max_pos_corners_rt_contact_ewrt_contact_x * scaleFactor),
        min_pos_corners_rt_contact_ewrt_contact_y(
            min_pos_corners_rt_contact_ewrt_contact_y * scaleFactor),
        max_pos_corners_rt_contact_ewrt_contact_y(
            max_pos_corners_rt_contact_ewrt_contact_y * scaleFactor) {}

  scalar_t min_pos_corners_rt_contact_ewrt_contact_x;
  scalar_t max_pos_corners_rt_contact_ewrt_contact_x;
  scalar_t min_pos_corners_rt_contact_ewrt_contact_y;
  scalar_t max_pos_corners_rt_contact_ewrt_contact_y;
};

///
/// \brief A planar contact polygon defined by the convex hull spanned up by a
/// set of corner points expressed in the frame name specified
///
/// \param[in] posCornersRtContactEwrtContact A vector of 3D points on the xy
/// plane (z = 0)
/// \param[in] contactCenterPoint the contact center point whose frame the
/// corners are given in
/// \param[in] scaleFactor A factor to shrink or extend the polygon
///
/// An immutable value once constructed, so concurrent reads are safe.
class ContactPolygon {
 public:
  /** The polygon of `posCornersRtContactEwrtContact` scaled by `scaleFactor`,
   * with the bounds of the scaled points. */
  ContactPolygon(absl::Span<const vector3_t> posCornersRtContactEwrtContact,
                 const ContactCenterPoint& contactCenterPoint,
                 scalar_t scaleFactor = 1.0);

  // Copyable, but not assignable: the contact center point is const. A move
  // copies the const contact center point, and only an allocation failure could
  // make that copy throw.
  ContactPolygon(const ContactPolygon&) = default;
  ContactPolygon& operator=(const ContactPolygon&) = delete;
  ContactPolygon(ContactPolygon&&) noexcept = default;
  ContactPolygon& operator=(ContactPolygon&&) = delete;
  ~ContactPolygon() = default;

  size_t getNumberOfContactPoints() const {
    return posCornersRtContactEwrtContact_.size();
  }
  vector3_t getPosCornerRtContactEwrtContact(int index) const {
    return vector3_t(posCornersRtContactEwrtContact_[index][0],
                     posCornersRtContactEwrtContact_[index][1], 0.0);
  }
  matrix3_t getCrossProductMatrixPosCornerRtContactEwrtContact(int index) const;
  const std::string& getParentJointName() const {
    return contactCenterPoint_.parentJointName;
  }
  const std::string& getPolygonPointFrameName(int i) const {
    return polygonPointFrameNames_[i];
  }
  const ContactCenterPoint& getContactCenterPoint() const {
    return contactCenterPoint_;
  }
  const PolygonBounds& getBounds() const { return polygonLimits_; }

 protected:
  // constructor can be used by child class. Child class has to ensure that the
  // polygonLimits are correct
  ContactPolygon(std::vector<vector3_t> posCornersRtContactEwrtContact,
                 const PolygonBounds& polygonBounds,
                 const ContactCenterPoint& contactCenterPoint);

 private:
  // contact dimensions relative to frame attached at contact surface
  PolygonBounds polygonLimits_;
  std::vector<vector3_t> posCornersRtContactEwrtContact_;
  std::vector<std::string> polygonPointFrameNames_;
  const ContactCenterPoint contactCenterPoint_;
};  // namespace ContactPolygon

}  // namespace ocs2::humanoid
