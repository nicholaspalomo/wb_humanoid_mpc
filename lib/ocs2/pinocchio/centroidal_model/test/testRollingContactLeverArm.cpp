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

// The lever arm of a rolling contact in the centroidal dynamics: with
// CentroidalModelInfo::rollingContactWheelRadii a wheel's force acts a radius
// below its frame, the wheel center; without it (the default) nothing changes.

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <string>
#include <vector>

#include "Eigen/Core"
#include "gtest/gtest.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"
#include "ocs2_centroidal_model/FactoryFunctions.h"
#include "ocs2_centroidal_model/ModelHelperFunctions.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/PhysicalConstants.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/automatic_differentiation/Types.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/multibody/frame.hpp"
#include "pinocchio/multibody/joint/joint-composite.hpp"
#include "pinocchio/multibody/joint/joint-spherical.hpp"
#include "pinocchio/multibody/joint/joint-translation.hpp"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/spatial/se3.hpp"

namespace ocs2 {
namespace {

using Vector6 = Eigen::Matrix<scalar_t, centroidal_model::kCentroidalMomentumDim, 1>;

/** A base and a leg on a knee hinge, whose wheel center carries the contact. */
constexpr char kLegUrdf[] = R"(<?xml version="1.0"?>
<robot name="leg">
  <link name="base">
    <inertial><origin xyz="0 0 0.05"/><mass value="10"/><inertia ixx="0.1" iyy="0.2" izz="0.3" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="shank">
    <inertial><origin xyz="0.02 0 -0.15"/><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="knee" type="revolute">
    <parent link="base"/><child link="shank"/><origin xyz="0.1 0.05 0"/><axis xyz="0 1 0"/>
    <limit lower="-2" upper="2" effort="80" velocity="10"/>
  </joint>
</robot>)";

constexpr char kWheelFrame[] = "wheel_contact";
constexpr scalar_t kWheelRadius = 0.098;
constexpr scalar_t kTolerance = 1.0e-12;

/** The model of kLegUrdf on the MPC's Composite(Translation, Spherical) root,
 * with the wheel frame 0.3 m below the knee. */
PinocchioInterface legInterface() {
  pinocchio::JointModelComposite root(/*nj=*/2);
  root.addJoint(pinocchio::JointModelTranslation());
  root.addJoint(pinocchio::JointModelSpherical());
  pinocchio::Model model;
  pinocchio::urdf::buildModelFromXML(kLegUrdf, root, model);
  const pinocchio::JointIndex knee = model.getJointId("knee");
  model.addFrame(pinocchio::Frame(
      kWheelFrame, knee, model.getFrameId("knee"),
      pinocchio::SE3(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0.01, 0.0, -0.3)),
      pinocchio::FIXED_JOINT));
  return PinocchioInterface(model);
}

/** The base 0.5 m up, pitched by a quarter turn of its quaternion's y, the knee
 * at 0.4 rad. */
vector_t legConfiguration(const PinocchioInterface& interface) {
  vector_t q = pinocchio::neutral(interface.getModel());
  q.head<kPositionDim>() = Eigen::Vector3d(0.2, -0.1, 0.5);
  q.segment<kQuaternionCoefficients>(kPositionDim) =
      Eigen::Vector4d(0.0, 0.0998334, 0.0, 0.9950042).normalized();
  q.tail(1)[0] = 0.4;
  return q;
}

/** The input: the wheel's force, then the knee velocity. */
vector_t legInput() {
  vector_t input(kForceDim + 1);
  input << 3.0, -2.0, 60.0, 0.0;
  return input;
}

CentroidalModelInfo legInfo(const PinocchioInterface& interface) {
  return centroidal_model::createCentroidalModelInfo(
      interface, CentroidalModelType::FullCentroidalDynamics,
      vector_t::Zero(1), /*threeDofContactNames=*/{kWheelFrame},
      /*sixDofContactNames=*/{});
}

/** (p - r e_z - c) x f + m g, f over the mass, by hand. */
Vector6 expectedRate(PinocchioInterface& interface,
                       const CentroidalModelInfo& info, scalar_t wheelRadius) {
  const pinocchio::Data& data = interface.getData();
  const Eigen::Vector3d posWheelRtWorldEwrtWorld =
      data.oMf[interface.getModel().getFrameId(kWheelFrame)].translation();
  const Eigen::Vector3d posContactRtComEwrtWorld =
      posWheelRtWorldEwrtWorld - wheelRadius * Eigen::Vector3d::UnitZ() -
      data.com[0];
  const Eigen::Vector3d forceContactEwrtWorld = legInput().head<kForceDim>();
  Vector6 rate;
  rate << info.robotMass * Eigen::Vector3d(0.0, 0.0, -kGravityMagnitude) +
              forceContactEwrtWorld,
      posContactRtComEwrtWorld.cross(forceContactEwrtWorld);
  return rate / info.robotMass;
}

TEST(RollingContactLeverArm, WithoutWheelRadiiTheForceActsAtTheFrame) {
  PinocchioInterface interface = legInterface();
  const CentroidalModelInfo info = legInfo(interface);
  ASSERT_TRUE(info.rollingContactWheelRadii.empty()) << "the default";
  updateCentroidalDynamics(interface, info, legConfiguration(interface));
  const Vector6 rate =
      getMnormRateCmomComRtWorldEwrtWorld(interface, info, legInput());
  EXPECT_TRUE(rate.isApprox(expectedRate(interface, info, /*wheelRadius=*/0.0),
                            kTolerance));

  // A zero radius, a contact fixed on its link, is the same expression.
  CentroidalModelInfo fixed = info;
  fixed.rollingContactWheelRadii = {0.0};
  EXPECT_EQ(getMnormRateCmomComRtWorldEwrtWorld(interface, fixed, legInput()),
            rate);
  EXPECT_EQ(getPosContactRtComEwrtWorld(interface, fixed, /*contactIndex=*/0),
            getPosContactRtComEwrtWorld(interface, info, /*contactIndex=*/0));
}

TEST(RollingContactLeverArm, AWheelsForceActsAtItsLowestPoint) {
  PinocchioInterface interface = legInterface();
  CentroidalModelInfo info = legInfo(interface);
  info.rollingContactWheelRadii = {kWheelRadius};
  updateCentroidalDynamics(interface, info, legConfiguration(interface));
  const Vector6 rate =
      getMnormRateCmomComRtWorldEwrtWorld(interface, info, legInput());
  EXPECT_TRUE(
      rate.isApprox(expectedRate(interface, info, kWheelRadius), kTolerance));
  const Vector6 rateAtTheCenter =
      expectedRate(interface, info, /*wheelRadius=*/0.0);
  EXPECT_GT((rate - rateAtTheCenter).norm(), 1.0e-3)
      << "the radius changes the moment";
  EXPECT_EQ(rate.head<kForceDim>(), rateAtTheCenter.head<kForceDim>());
}

TEST(RollingContactLeverArm, TheRadiiReachTheAutomaticDifferentiationModel) {
  PinocchioInterface interface = legInterface();
  CentroidalModelInfo info = legInfo(interface);
  info.rollingContactWheelRadii = {kWheelRadius};
  const vector_t q = legConfiguration(interface);
  updateCentroidalDynamics(interface, info, q);
  const Eigen::Vector3d posContactRtComEwrtWorld =
      getPosContactRtComEwrtWorld(interface, info, /*contactIndex=*/0);

  const CentroidalModelInfoCppAd infoAd = info.toCppAd();
  ASSERT_EQ(infoAd.rollingContactWheelRadii.size(), 1U);
  PinocchioInterfaceCppAd interfaceAd = interface.toCppAd();
  const ad_vector_t qAd = q.cast<ad_scalar_t>();
  updateCentroidalDynamics(interfaceAd, infoAd, qAd);
  const Eigen::Matrix<ad_scalar_t, kPositionDim, 1> adPosContactRtComEwrtWorld =
      getPosContactRtComEwrtWorld(interfaceAd, infoAd, /*contactIndex=*/0);
  for (int axis = 0; axis < kPositionDim; ++axis) {
    EXPECT_NEAR(CppAD::Value(adPosContactRtComEwrtWorld[axis]).getValue(),
                posContactRtComEwrtWorld[axis], kTolerance)
        << axis;
  }
}

}  // namespace
}  // namespace ocs2
