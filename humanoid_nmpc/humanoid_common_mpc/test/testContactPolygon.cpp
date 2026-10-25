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

#include <algorithm>
#include <random>
#include <vector>

#include "gtest/gtest.h"

#include "humanoid_common_mpc/contact/ContactCenterPoint.h"
#include "humanoid_common_mpc/contact/ContactPolygon.h"

/*
 * The bounds of a ContactPolygon built from its corner points: the extent of
 * the scaled points, at every scale factor (1 included, where they used to stay
 * zero) and for a polygon that does not contain the frame's origin.
 */
namespace ocs2::humanoid {

TEST(ContactPolygon, BoundsAreTheExtentOfTheScaledPoints) {
  std::mt19937 generator(29);
  std::uniform_real_distribution<scalar_t> coordinate(-0.2, 0.3);
  std::uniform_real_distribution<scalar_t> offset(-0.5, 0.5);
  const ContactCenterPoint center("sole", "ankle", vector3_t::Zero());
  for (int trial = 0; trial < 200; ++trial) {
    // An offset polygon, often entirely on one side of the origin.
    const vector3_t shift(offset(generator), offset(generator), 0.0);
    std::vector<vector3_t> points;
    for (int i = 0; i < 3 + trial % 4; ++i)
      points.push_back(
          shift + vector3_t(coordinate(generator), coordinate(generator), 0.0));
    for (const scalar_t scale : {1.0, 0.8, 1.3}) {
      const ContactPolygon polygon(points, center, scale);
      scalar_t xMin = scale * points.front()(0);
      scalar_t xMax = xMin;
      scalar_t yMin = scale * points.front()(1);
      scalar_t yMax = yMin;
      for (const vector3_t& point : points) {
        xMin = std::min(xMin, scale * point(0));
        xMax = std::max(xMax, scale * point(0));
        yMin = std::min(yMin, scale * point(1));
        yMax = std::max(yMax, scale * point(1));
      }
      EXPECT_DOUBLE_EQ(polygon.getBounds().x_min, xMin)
          << "trial " << trial << " scale " << scale;
      EXPECT_DOUBLE_EQ(polygon.getBounds().x_max, xMax);
      EXPECT_DOUBLE_EQ(polygon.getBounds().y_min, yMin);
      EXPECT_DOUBLE_EQ(polygon.getBounds().y_max, yMax);
      ASSERT_EQ(polygon.getNumberOfContactPoints(), points.size());
      for (size_t i = 0; i < points.size(); ++i) {
        EXPECT_DOUBLE_EQ(
            polygon.getContactPointTranslation(static_cast<int>(i))(0),
            scale * points[i](0));
        EXPECT_DOUBLE_EQ(
            polygon.getContactPointTranslation(static_cast<int>(i))(1),
            scale * points[i](1));
      }
    }
  }
}

}  // namespace ocs2::humanoid
