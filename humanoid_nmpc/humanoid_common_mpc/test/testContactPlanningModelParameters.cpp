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

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <gtest/gtest.h>

#include <cmath>
#include <string>

#include <pinocchio/multibody/joint/joint-generic.hpp>
#include <pinocchio/multibody/model.hpp>

#include "humanoid_common_mpc/contact_planning/ContactPlanningModelParameters.h"

/**
 * deriveHipYawRange() walks a leg up from its contact parent joint to the first revolute joint whose axis is vertical
 * in the world. An axis along a coordinate axis is a property of the joint's type (JointModelRZ, JointModelRUBZ), but
 * any other axis - including -z, which pinocchio's URDF parser turns into an unaligned joint - is data of the joint
 * model, and has to be read from it for the bounded and the unbounded unaligned joint alike. These tests build a leg in
 * which only the hip yaw can be vertical, so that an axis that is not read sends the walk past it to the root and the
 * leg back to the symmetric fallback.
 */
namespace ocs2::humanoid {
namespace {

constexpr pinocchio::JointIndex kUniverse = 0;
constexpr char kHipYawJoint[] = "hip_yaw";
constexpr char kHipRollJoint[] = "hip_roll";
constexpr char kKneeJoint[] = "knee";

// The hip yaw joint's position limits: asymmetric, so that a mirrored range differs from the unmirrored one.
constexpr scalar_t kLowerLimit = -0.3;
constexpr scalar_t kUpperLimit = 0.6;

/**
 * A leg as a URDF hangs it - the hip yaw under test at the root, then a hip roll and a knee about horizontal axes -
 * with every placement the identity, so that each joint's axis in the world is its local axis. Every configuration
 * entry of the hip yaw gets [kLowerLimit, kUpperLimit], the (cos q, sin q) entries of an unbounded joint included.
 */
PinocchioInterface::Model makeLeg(const PinocchioInterface::JointModel& hipYaw) {
  PinocchioInterface::Model model;
  const pinocchio::JointIndex hipYawIndex = model.addJoint(kUniverse, hipYaw, pinocchio::SE3::Identity(), kHipYawJoint);
  const int idxQ = model.joints[hipYawIndex].idx_q();
  const int nq = model.joints[hipYawIndex].nq();
  model.lowerPositionLimit.segment(idxQ, nq).setConstant(kLowerLimit);
  model.upperPositionLimit.segment(idxQ, nq).setConstant(kUpperLimit);
  const pinocchio::JointIndex hipRollIndex =
      model.addJoint(hipYawIndex, pinocchio::JointModelRX(), pinocchio::SE3::Identity(), kHipRollJoint);
  model.addJoint(hipRollIndex, pinocchio::JointModelRY(), pinocchio::SE3::Identity(), kKneeJoint);
  return model;
}

/** The symmetric fallback a leg without a hip yaw joint gets. */
void expectFallback(const HipYawRange& range) {
  const HipYawRange fallback;
  EXPECT_TRUE(range.joint.empty()) << "found '" << range.joint << "' where the leg has no vertical revolute joint";
  EXPECT_EQ(range.lower, fallback.lower);
  EXPECT_EQ(range.upper, fallback.upper);
}

TEST(DeriveHipYawRange, AnAlignedUpwardHipYawContributesItsLimits) {
  const HipYawRange range = deriveHipYawRange(makeLeg(pinocchio::JointModelRZ()), kKneeJoint);
  EXPECT_EQ(range.joint, kHipYawJoint);
  EXPECT_EQ(range.lower, kLowerLimit);
  EXPECT_EQ(range.upper, kUpperLimit);
}

TEST(DeriveHipYawRange, AnUnalignedUpwardHipYawIsFoundAndReadLikeAnAlignedOne) {
  const HipYawRange aligned = deriveHipYawRange(makeLeg(pinocchio::JointModelRZ()), kKneeJoint);
  const HipYawRange unaligned = deriveHipYawRange(makeLeg(pinocchio::JointModelRevoluteUnaligned(vector3_t::UnitZ())), kKneeJoint);
  EXPECT_EQ(unaligned.joint, kHipYawJoint) << "the axis of an unaligned joint is data of the joint model and must be read from it";
  EXPECT_EQ(unaligned.lower, aligned.lower);
  EXPECT_EQ(unaligned.upper, aligned.upper);
}

TEST(DeriveHipYawRange, AnUnalignedDownwardHipYawMirrorsItsLimits) {
  // The axis a URDF writes as "0 0 -1": the foot yaw is -q, so the joint limits [lower, upper] are the foot yaw range
  // [-upper, -lower].
  const HipYawRange range = deriveHipYawRange(makeLeg(pinocchio::JointModelRevoluteUnaligned(-vector3_t::UnitZ())), kKneeJoint);
  EXPECT_EQ(range.joint, kHipYawJoint);
  EXPECT_EQ(range.lower, -kUpperLimit);
  EXPECT_EQ(range.upper, -kLowerLimit);
}

TEST(DeriveHipYawRange, AnUnboundedUnalignedHipYawIsFoundAndTurnsAllTheWayRound) {
  // The unbounded unaligned joint is a variant alternative of its own, not a kind of the bounded one; reading only the
  // bounded alternative's axis once made this joint invisible. Its (cos q, sin q) limits are not angles.
  for (const vector3_t& axis : {vector3_t(vector3_t::UnitZ()), vector3_t(-vector3_t::UnitZ())}) {
    const HipYawRange range = deriveHipYawRange(makeLeg(pinocchio::JointModelRevoluteUnboundedUnaligned(axis)), kKneeJoint);
    EXPECT_EQ(range.joint, kHipYawJoint) << "axis " << axis.transpose();
    EXPECT_EQ(range.lower, -M_PI) << "axis " << axis.transpose();
    EXPECT_EQ(range.upper, M_PI) << "axis " << axis.transpose();
  }
}

TEST(DeriveHipYawRange, AnUnboundedAlignedHipYawTurnsAllTheWayRound) {
  const HipYawRange range = deriveHipYawRange(makeLeg(pinocchio::JointModelRUBZ()), kKneeJoint);
  EXPECT_EQ(range.joint, kHipYawJoint);
  EXPECT_EQ(range.lower, -M_PI);
  EXPECT_EQ(range.upper, M_PI);
}

TEST(DeriveHipYawRange, AnUnalignedHorizontalJointIsNotAHipYaw) {
  // Read correctly, the axis of an unaligned joint can also rule it out.
  expectFallback(deriveHipYawRange(makeLeg(pinocchio::JointModelRevoluteUnaligned(vector3_t::UnitX())), kKneeJoint));
  expectFallback(deriveHipYawRange(makeLeg(pinocchio::JointModelRevoluteUnboundedUnaligned(vector3_t::UnitY())), kKneeJoint));
}

TEST(DeriveHipYawRange, AJointTheModelDoesNotHaveGivesTheFallback) {
  expectFallback(deriveHipYawRange(makeLeg(pinocchio::JointModelRZ()), "no_such_joint"));
}

}  // namespace
}  // namespace ocs2::humanoid
