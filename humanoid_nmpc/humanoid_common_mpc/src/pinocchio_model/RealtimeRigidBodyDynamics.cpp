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

#include "humanoid_common_mpc/pinocchio_model/RealtimeRigidBodyDynamics.h"

#include <array>
#include <cstddef>
#include <string>

#include "Eigen/Cholesky"
#include "absl/base/nullability.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"
#include "pinocchio/algorithm/centroidal.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/multibody/joint/joint-free-flyer.hpp"

#include "humanoid_common_mpc/common/StatusMacros.h"

namespace ocs2::humanoid {
namespace {

/** The root joint of the MPC models, the first joint after the universe. */
constexpr pinocchio::JointIndex kRootJoint = 1;
/** The configuration and velocity sizes of the root, Composite(Translation,
 * Spherical) or FreeFlyer alike. */
constexpr int kRootConfigurationDim = kPositionDim + kQuaternionCoefficients;
constexpr int kRootVelocityDim = kPoseTangentDim;
/** [m] and [rad] The base pose the twin's root is compared with the MPC model's
 * at, away from the identity. */
constexpr double kProbeTranslation = 0.3;
constexpr double kProbeAngle = 0.7;
/** The poses the two roots are compared at: the neutral one and the displaced
 * one. */
constexpr size_t kNumProbePoses = 2;
/** How far the two roots' placements may differ at the probe pose: round-off.
 */
constexpr double kPlacementTolerance = 1.0e-12;

/**
 * Checks that the FreeFlyer root of `twin` puts its child bodies where the
 * composite root of `mpcModel` does, at the neutral configuration and at a pose
 * with a translation and a rotation about a skew axis.
 */
absl::Status checkSameRootPlacement(const PinocchioInterface::Model& mpcModel,
                                    const PinocchioInterface::Model& twin) {
  PinocchioInterface::Data mpcData(mpcModel);
  PinocchioInterface::Data twinData(twin);
  const vector_t neutral = pinocchio::neutral(mpcModel);
  vector_t displaced = neutral;
  displaced.head<kPositionDim>().setConstant(kProbeTranslation);
  const vector3_t axis =
      // NOLINTNEXTLINE(magic-number): an arbitrary skew axis
      vector3_t(1.0, 2.0, 3.0).normalized();
  displaced.segment<kQuaternionCoefficients>(kPositionDim) =
      quaternionExp<scalar_t>(vector3_t(kProbeAngle * axis));
  const std::array<const vector_t* absl_nonnull, kNumProbePoses> probes = {
      &neutral, &displaced};
  for (const vector_t* absl_nonnull q : probes) {
    pinocchio::forwardKinematics(mpcModel, mpcData, *q);
    pinocchio::forwardKinematics(twin, twinData, *q);
    if (!mpcData.oMi[kRootJoint].isApprox(twinData.oMi[kRootJoint],
                                          kPlacementTolerance)) {
      return absl::InvalidArgumentError(
          "[RealtimeRigidBodyDynamics] the root joint of the model does not "
          "place the base as a FreeFlyer of the same configuration "
          "does: it must be Composite(Translation, Spherical) with nothing "
          "between the two");
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<RealtimeRigidBodyDynamics>>
RealtimeRigidBodyDynamics::Create(
    const PinocchioInterface::Model& mpcModel,
    const feet_array_t<pinocchio::FrameIndex>& contactFrames) {
  if (mpcModel.njoints <= static_cast<int>(kRootJoint) ||
      mpcModel.parents[kRootJoint] != 0 ||
      mpcModel.joints[kRootJoint].shortname() != "JointModelComposite" ||
      mpcModel.joints[kRootJoint].nq() != kRootConfigurationDim ||
      mpcModel.joints[kRootJoint].nv() != kRootVelocityDim) {
    return absl::InvalidArgumentError(
        absl::StrCat("[RealtimeRigidBodyDynamics] the model's root joint is ",
                     mpcModel.njoints > static_cast<int>(kRootJoint)
                         ? mpcModel.joints[kRootJoint].shortname()
                         : std::string("missing"),
                     ", not the MPC models' Composite(Translation, Spherical) "
                     "on the universe"));
  }
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    if (contactFrames[contact] >= mpcModel.frames.size()) {
      return absl::NotFoundError(absl::StrCat(
          "[RealtimeRigidBodyDynamics] contact ", contact,
          " has the frame index ", contactFrames[contact],
          ", but the model has ", mpcModel.frames.size(), " frames"));
    }
  }
  // The same model with the root's joint model replaced: the root keeps its
  // indices, sizes and placement, so every index, support and sparsity pattern
  // of the model stays as it is.
  PinocchioInterface::Model twin = mpcModel;
  pinocchio::JointModelFreeFlyer root;
  root.setIndexes(kRootJoint, mpcModel.idx_qs[kRootJoint],
                  mpcModel.idx_vs[kRootJoint],
                  mpcModel.idx_vExtendeds[kRootJoint]);
  twin.joints[kRootJoint] = root;
  RETURN_IF_ERROR(checkSameRootPlacement(mpcModel, twin));
  // The constructor is private, so std::make_unique cannot reach it.
  return absl::WrapUnique(new RealtimeRigidBodyDynamics(twin, contactFrames));
}

RealtimeRigidBodyDynamics::RealtimeRigidBodyDynamics(
    const PinocchioInterface::Model& twin,
    const feet_array_t<pinocchio::FrameIndex>& contactFrames)
    : model_(twin),
      data_(model_),
      contactFrames_(contactFrames),
      configuration_(pinocchio::neutral(model_)),
      generalizedVelocity_(vector_t::Zero(model_.nv)),
      generalizedAcceleration_(vector_t::Zero(model_.nv)),
      soleJacobian_(matrix_t::Zero(kTwistDim, model_.nv)),
      generalizedContactForce_(vector_t::Zero(model_.nv)),
      jointTorques_(vector_t::Zero(model_.nv - kRootVelocityDim)),
      gravityJointTorques_(vector_t::Zero(model_.nv - kRootVelocityDim)) {
  rotmatContactsRtWorld_.fill(matrix3_t::Identity());
}

matrix3_t RealtimeRigidBodyDynamics::rotmatBaseRtWorld() const {
  return quaternionRotationMatrix<scalar_t>(quaternion_coeffs_t<scalar_t>(
      configuration_.segment<kQuaternionCoefficients>(kPositionDim)));
}

void RealtimeRigidBodyDynamics::setState(const Eigen::Ref<const vector_t>& q,
                                         const Eigen::Ref<const vector_t>& v) {
  configuration_ = q;
  // The FreeFlyer's linear velocity is the base's in its own axes, R' dp_W; the
  // angular and joint velocities are shared.
  generalizedVelocity_ = v;
  generalizedVelocity_.head<kPositionDim>().noalias() =
      rotmatBaseRtWorld().transpose() * v.head<kPositionDim>();
}

void RealtimeRigidBodyDynamics::computeGeneralizedContactForce(
    const feet_array_t<vector6_t>& wrenchFeetEwrtWorld) {
  // The LOCAL_WORLD_ALIGNED Jacobians take the world-frame wrenches at the sole
  // origins. Their joint columns are those of the MPC model; only the base
  // columns follow the twin's base velocity.
  pinocchio::computeJointJacobians(model_, data_, configuration_);
  generalizedContactForce_.setZero();
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    soleJacobian_.setZero();
    pinocchio::getFrameJacobian(model_, data_, contactFrames_[contact],
                                pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
                                soleJacobian_);
    generalizedContactForce_.noalias() +=
        soleJacobian_.transpose() * wrenchFeetEwrtWorld[contact];
  }
}

const vector_t& RealtimeRigidBodyDynamics::gravityJointTorques(
    const Eigen::Ref<const vector_t>& q) {
  configuration_ = q;
  pinocchio::computeGeneralizedGravity(model_, data_, configuration_);
  gravityJointTorques_ = data_.g.tail(gravityJointTorques_.size());
  return gravityJointTorques_;
}

const vector_t& RealtimeRigidBodyDynamics::freeBaseJointTorques(
    const Eigen::Ref<const vector_t>& q,
    const Eigen::Ref<const vector_t>& v,
    const Eigen::Ref<const vector_t>& qddJoints,
    const feet_array_t<vector6_t>& wrenchFeetEwrtWorld) {
  setState(q, v);
  // M with both triangles (crba() writes the upper one), and nle(q, v), in the
  // twin's coordinates.
  pinocchio::crba(model_, data_, configuration_);
  data_.M.triangularView<Eigen::StrictlyLower>() =
      data_.M.transpose().triangularView<Eigen::StrictlyLower>();
  pinocchio::nonLinearEffects(model_, data_, configuration_,
                              generalizedVelocity_);
  computeGeneralizedContactForce(wrenchFeetEwrtWorld);

  // The base rows M_bb a_b = (J^T W)_b - nle_b - M_bj qdd_j. M_bb is the
  // robot's composite inertia, positive definite.
  const Eigen::Index numJoints = qddJoints.size();
  vector6_t baseRows = generalizedContactForce_.head<kRootVelocityDim>() -
                       data_.nle.head<kRootVelocityDim>();
  baseRows.noalias() -=
      data_.M.topRightCorner(kRootVelocityDim, numJoints) * qddJoints;
  const Eigen::LLT<Eigen::Matrix<scalar_t, kRootVelocityDim, kRootVelocityDim>>
      baseInertia(data_.M.topLeftCorner<kRootVelocityDim, kRootVelocityDim>());
  generalizedAcceleration_.head<kRootVelocityDim>() =
      baseInertia.solve(baseRows);
  generalizedAcceleration_.tail(numJoints) = qddJoints;
  jointTorques_.noalias() =
      data_.M.bottomRows(numJoints) * generalizedAcceleration_;
  jointTorques_ += data_.nle.tail(numJoints);
  jointTorques_ -= generalizedContactForce_.tail(numJoints);

  // The base acceleration in the MPC model's coordinates: v_B = R' dp_W, so
  // ddp_W = R (dv_B + w_B x v_B).
  const vector3_t linvelBaseRtWorldEwrtBase =
      generalizedVelocity_.head<kPositionDim>();
  const vector3_t angvelBaseRtWorldEwrtBase =
      generalizedVelocity_.segment<kRotationTangentDim>(kPositionDim);
  const vector3_t rateLinvelBaseRtWorldEwrtBase =
      generalizedAcceleration_.head<kPositionDim>();
  baseGeneralizedAcceleration_.head<kPositionDim>() =
      rotmatBaseRtWorld() *
      (rateLinvelBaseRtWorldEwrtBase +
       angvelBaseRtWorldEwrtBase.cross(linvelBaseRtWorldEwrtBase));
  baseGeneralizedAcceleration_.tail<kRotationTangentDim>() =
      generalizedAcceleration_.segment<kRotationTangentDim>(kPositionDim);
  return jointTorques_;
}

const vector_t& RealtimeRigidBodyDynamics::heldBaseJointTorques(
    const Eigen::Ref<const vector_t>& q,
    const Eigen::Ref<const vector_t>& v,
    const Eigen::Ref<const vector_t>& qddJoints,
    const feet_array_t<vector6_t>& wrenchFeetEwrtWorld) {
  setState(q, v);
  // ddp_W = 0 and dw_B = 0 are, in the twin's coordinates, dv_B = -w_B x v_B
  // and dw_B = 0.
  const vector3_t linvelBaseRtWorldEwrtBase =
      generalizedVelocity_.head<kPositionDim>();
  const vector3_t angvelBaseRtWorldEwrtBase =
      generalizedVelocity_.segment<kRotationTangentDim>(kPositionDim);
  generalizedAcceleration_.head<kPositionDim>() =
      -angvelBaseRtWorldEwrtBase.cross(linvelBaseRtWorldEwrtBase);
  generalizedAcceleration_.segment<kRotationTangentDim>(kPositionDim).setZero();
  const Eigen::Index numJoints = qddJoints.size();
  generalizedAcceleration_.tail(numJoints) = qddJoints;
  pinocchio::rnea(model_, data_, configuration_, generalizedVelocity_,
                  generalizedAcceleration_);
  computeGeneralizedContactForce(wrenchFeetEwrtWorld);
  jointTorques_ = data_.tau.tail(numJoints);
  jointTorques_ -= generalizedContactForce_.tail(numJoints);
  return jointTorques_;
}

const vector6_t& RealtimeRigidBodyDynamics::computeCmomComRtWorldEwrtWorld(
    const Eigen::Ref<const vector_t>& q, const Eigen::Ref<const vector_t>& v) {
  setState(q, v);
  pinocchio::ccrba(model_, data_, configuration_, generalizedVelocity_);
  cmomComRtWorldEwrtWorld_ << data_.hg.linear(), data_.hg.angular();
  return cmomComRtWorldEwrtWorld_;
}

const feet_array_t<matrix3_t>&
RealtimeRigidBodyDynamics::computeRotmatContactsRtWorld(
    const Eigen::Ref<const vector_t>& q) {
  configuration_ = q;
  pinocchio::forwardKinematics(model_, data_, configuration_);
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    rotmatContactsRtWorld_[contact] =
        pinocchio::updateFramePlacement(model_, data_, contactFrames_[contact])
            .rotation();
  }
  return rotmatContactsRtWorld_;
}

}  // namespace ocs2::humanoid
