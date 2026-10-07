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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <cmath>
#include <random>

#include "gtest/gtest.h"
#include "ocs2_robotic_tools/common/RotationTransforms.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"

/**
 * quaternionToEulerZYX(), which the MRTs use to write the measured base orientation into the MPC state. The sine of
 * the pitch is 2(wy - zx), which is exactly 1 at a pitch of +90 degrees but can round to just above it, and asin was
 * then NaN. It is clamped to [-1, 1] now; inside that range the conversion is the one it always was, to the bit.
 */
namespace ocs2::humanoid {
namespace {

/** The conversion as it was before the clamp: the reference the clamped one must reproduce wherever it was defined. */
vector3_t unclampedQuaternionToEulerZyx(const quaternion_t& quat) {
  const scalar_t w = quat.w();
  const scalar_t x = quat.x();
  const scalar_t y = quat.y();
  const scalar_t z = quat.z();
  const scalar_t yaw = std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
  const scalar_t pitch = std::asin(2.0 * (w * y - z * x));
  const scalar_t roll = std::atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y));
  return vector3_t(yaw, pitch, roll);
}

/** The sine of the pitch the conversion takes the asin of. */
scalar_t sinPitchOf(const quaternion_t& quat) {
  return 2.0 * (quat.w() * quat.y() - quat.z() * quat.x());
}

/** From coefficients in Eigen's (x, y, z, w) storage order: Eigen's four-scalar constructor takes (w, x, y, z). */
quaternion_t quaternionFromCoeffs(scalar_t x, scalar_t y, scalar_t z, scalar_t w) {
  return quaternion_t(vector4_t(x, y, z, w));
}

/** The rotation the Euler angles (yaw, pitch, roll) describe, R = Rz(yaw) Ry(pitch) Rx(roll). */
matrix3_t rotationOf(const vector3_t& eulerZyx) {
  return getRotationMatrixFromZyxEulerAngles<scalar_t>(eulerZyx);
}

TEST(QuaternionToEulerClamp, IsFiniteWhereTheSineOfThePitchRoundsPastOne) {
  // A pitch of exactly +90 degrees, written as the measured quaternion would carry it: sqrt(0.5)^2 rounds to
  // 0.5 + 2^-53, so the sine of the pitch is 1 + 2^-52.
  const scalar_t halfSqrt2 = std::sqrt(0.5);
  for (const scalar_t sign : {1.0, -1.0}) {
    const quaternion_t quat = quaternionFromCoeffs(/*x=*/0.0, /*y=*/sign * halfSqrt2, /*z=*/0.0, /*w=*/halfSqrt2);
    // The case under test, and the failure it was: the argument is past +-1, and the unclamped conversion was NaN.
    ASSERT_GT(std::abs(sinPitchOf(quat)), 1.0) << "sign " << sign;
    ASSERT_TRUE(std::isnan(unclampedQuaternionToEulerZyx(quat)(1))) << "sign " << sign;

    const vector3_t euler = quaternionToEulerZYX(quat);
    EXPECT_TRUE(euler.allFinite()) << "sign " << sign << ": " << euler.transpose();
    EXPECT_EQ(euler(1), sign * M_PI_2) << "sign " << sign;
    // At the singularity only yaw - roll (pitch +90) or yaw + roll (pitch -90) is defined, so the angles are checked
    // through the rotation they describe rather than one by one.
    EXPECT_TRUE(rotationOf(euler).isApprox(quat.toRotationMatrix(), /*prec=*/1.0e-12)) << "sign " << sign << "\n"
                                                                                       << rotationOf(euler) << "\nvs\n"
                                                                                       << quat.toRotationMatrix();
  }
}

TEST(QuaternionToEulerClamp, IsFiniteForAMeasuredQuaternionSlightlyLongerThanOne) {
  // A simulator or an estimator hands over quaternions normalized only to round-off. Scaled up by a part in 1e12 near
  // a pitch of 90 degrees, the sine of the pitch exceeds 1 by far more than one rounding step.
  const scalar_t halfSqrt2 = std::sqrt(0.5);
  for (const scalar_t scale : {1.0 + 1.0e-12, 1.0 + 1.0e-9}) {
    const quaternion_t quat = quaternionFromCoeffs(/*x=*/0.0, /*y=*/scale * halfSqrt2, /*z=*/0.0, /*w=*/scale * halfSqrt2);
    ASSERT_GT(sinPitchOf(quat), 1.0) << "scale " << scale;
    const vector3_t euler = quaternionToEulerZYX(quat);
    EXPECT_TRUE(euler.allFinite()) << "scale " << scale << ": " << euler.transpose();
    EXPECT_EQ(euler(1), M_PI_2) << "scale " << scale;
  }
}

TEST(QuaternionToEulerClamp, IsTheUnclampedConversionToTheBitWhereverThatWasDefined) {
  // Random unit quaternions, normalized in double precision as the measured ones are: the sine of the pitch lies in
  // [-1, 1] for all of them but the rare rounding at +-90 degrees, and there the conversion is unchanged, bit for bit.
  std::mt19937 generator(11);
  std::normal_distribution<scalar_t> coefficient(0.0, 1.0);
  int compared = 0;
  for (int sample = 0; sample < 20000; ++sample) {
    const vector4_t coeffs(coefficient(generator), coefficient(generator), coefficient(generator), coefficient(generator));
    const quaternion_t quat(vector4_t(coeffs.normalized()));
    if (std::abs(sinPitchOf(quat)) > 1.0) continue;  // the case the clamp exists for
    const vector3_t clamped = quaternionToEulerZYX(quat);
    const vector3_t unclamped = unclampedQuaternionToEulerZyx(quat);
    for (int i = 0; i < 3; ++i) {
      ASSERT_EQ(clamped(i), unclamped(i)) << "sample " << sample << ", angle " << i;
    }
    ++compared;
  }
  EXPECT_GT(compared, 19000);
}

TEST(QuaternionToEulerClamp, DescribesTheRotationOfTheQuaternion) {
  // The angles are the ZYX Euler angles of the quaternion's rotation, at every attitude including the tilts the
  // singularity is approached from.
  std::mt19937 generator(5);
  std::uniform_real_distribution<scalar_t> yawDistribution(-M_PI, M_PI);
  std::uniform_real_distribution<scalar_t> pitchDistribution(-M_PI_2, M_PI_2);
  std::uniform_real_distribution<scalar_t> rollDistribution(-M_PI, M_PI);
  for (int sample = 0; sample < 2000; ++sample) {
    const vector3_t euler(yawDistribution(generator), pitchDistribution(generator), rollDistribution(generator));
    const quaternion_t quat(rotationOf(euler));
    const vector3_t converted = quaternionToEulerZYX(quat);
    ASSERT_TRUE(converted.allFinite()) << "sample " << sample;
    EXPECT_TRUE(rotationOf(converted).isApprox(quat.toRotationMatrix(), /*prec=*/1.0e-9)) << "sample " << sample;
    EXPECT_LE(std::abs(converted(1)), M_PI_2) << "sample " << sample;
  }
}

}  // namespace
}  // namespace ocs2::humanoid
