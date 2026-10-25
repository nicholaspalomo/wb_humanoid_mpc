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

#include "humanoid_common_mpc/cost/ComAndAcomTrackingCost.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/die_if_null.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/Dimensions.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/orientation/BaseOrientation.h"

namespace ocs2::humanoid {

namespace {

/// The base pose's rows in the centroidal tuning layout, which Q is written in
/// (tuningLayoutOf(): the normalized linear and angular momentum, then
/// base_position and base_orientation): the rows this cost replaces.
constexpr Eigen::Index kBasePoseTuningRow = 6;
constexpr Eigen::Index kBasePoseTuningDim = 6;

/**
 * Loads the aCOM network for `robotName` and checks it against the model the
 * cost evaluates: the network's joints, name by name, against the model's
 * actuated joints, which also pins the count to info.actuatedDofNum.
 */
absl::StatusOr<std::unique_ptr<AngularCenterOfMass>> createCheckedAcom(
    const PinocchioInterface& pinocchioInterface,
    const CentroidalModelInfo& info,
    absl::string_view robotName) {
  ASSIGN_OR_RETURN(
      const std::vector<std::string> modelJointNames,
      ComAndAcomTrackingCost::actuatedJointNames(pinocchioInterface, info));
  return AngularCenterOfMass::Create(robotName, modelJointNames);
}

/**
 * Checks that the robot model reads the state of the Pinocchio model the cost
 * evaluates: its configuration (getConfigurationDim(), the size of
 * getGeneralizedCoordinates()) and velocity (getVelocityDim(), the rows of
 * getPinocchioTangentJacobian()) are Pinocchio's nq and nv, and its joints are
 * the actuated joints of `info`, which the network is fed.
 */
absl::Status checkRobotModelDimensions(
    const PinocchioInterface& pinocchioInterface,
    const CentroidalModelInfo& info,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel) {
  const pinocchio::Model& model = pinocchioInterface.getModel();
  if (mpcRobotModel.getConfigurationDim() != static_cast<size_t>(model.nq) ||
      mpcRobotModel.getVelocityDim() != static_cast<size_t>(model.nv) ||
      mpcRobotModel.getJointDim() != info.actuatedDofNum) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[ComAndAcomTrackingCost] the MPC robot model does not read the "
        "Pinocchio model's state: its configuration, velocity and "
        "joint dimensions are ",
        mpcRobotModel.getConfigurationDim(), ", ",
        mpcRobotModel.getVelocityDim(), " and ", mpcRobotModel.getJointDim(),
        ", the Pinocchio model's nq and nv are ", model.nq, " and ", model.nv,
        ", and CentroidalModelInfo.actuatedDofNum is ", info.actuatedDofNum,
        "."));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::vector<std::string>>
ComAndAcomTrackingCost::actuatedJointNames(
    const PinocchioInterface& pinocchioInterface,
    const CentroidalModelInfo& info) {
  const pinocchio::Model& model = pinocchioInterface.getModel();
  // The floating base is the model's first joint after the universe, with as
  // many velocities as its root joint has (the Euler root's six coordinates or
  // a quaternion root's seven both move in six); the actuated joints' follow
  // them.
  const int floatingBaseVelocityDim =
      model.njoints > 1 ? model.joints[1].nv() : 0;
  if (model.nv - floatingBaseVelocityDim !=
      static_cast<int>(info.actuatedDofNum)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[ComAndAcomTrackingCost] CentroidalModelInfo.actuatedDofNum (",
        info.actuatedDofNum,
        ") does not describe the Pinocchio model, which has ",
        model.nv - floatingBaseVelocityDim,
        " joint velocities after its floating base."));
  }
  // model.names starts with the universe and the floating-base joint; the
  // actuated joints are the rest, in the order the state stores them. Counting
  // from the end keeps this independent of how many joints the base is built
  // from.
  const size_t numActuated = static_cast<size_t>(info.actuatedDofNum);
  if (model.names.size() < numActuated) {
    return absl::InvalidArgumentError(
        absl::StrCat("[ComAndAcomTrackingCost] the Pinocchio model names ",
                     model.names.size(), " joints, fewer than actuatedDofNum (",
                     numActuated, ")."));
  }
  return std::vector<std::string>(
      model.names.end() - static_cast<ptrdiff_t>(numActuated),
      model.names.end());
}

absl::Status ComAndAcomTrackingCost::validateWeights(const matrix_t& Q_com,
                                                     const matrix_t& Q_acom) {
  if (Q_com.rows() != kPositionDim || Q_com.cols() != kPositionDim) {
    return absl::InvalidArgumentError(
        absl::StrCat("[ComAndAcomTrackingCost] Q_com, the weight of "
                     "com_weights, must be 3x3, got ",
                     Q_com.rows(), "x", Q_com.cols(), "."));
  }
  if (Q_acom.rows() != kRotationTangentDim ||
      Q_acom.cols() != kRotationTangentDim) {
    return absl::InvalidArgumentError(
        absl::StrCat("[ComAndAcomTrackingCost] Q_acom, the weight of "
                     "acom_weights, must be 3x3, got ",
                     Q_acom.rows(), "x", Q_acom.cols(), "."));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<ComAndAcomTrackingCost>>
ComAndAcomTrackingCost::Create(
    matrix_t Q_com,
    matrix_t Q_acom,
    PinocchioInterface pinocchioInterface,
    const CentroidalModelInfo& info,
    const MpcRobotModelBase<scalar_t>* absl_nonnull mpcRobotModel,
    absl::string_view robotName) {
  RETURN_IF_ERROR(validateWeights(Q_com, Q_acom));
  ASSIGN_OR_RETURN(std::unique_ptr<AngularCenterOfMass> acom,
                   createCheckedAcom(pinocchioInterface, info, robotName));
  RETURN_IF_ERROR(checkRobotModelDimensions(pinocchioInterface, info,
                                            *ABSL_DIE_IF_NULL(mpcRobotModel)));
  // The constructor is private, so std::make_unique cannot reach it.
  return absl::WrapUnique(new ComAndAcomTrackingCost(
      std::move(Q_com), std::move(Q_acom), std::move(pinocchioInterface),
      mpcRobotModel, std::move(acom)));
}

ComAndAcomTrackingCost::ComAndAcomTrackingCost(
    matrix_t Q_com,
    matrix_t Q_acom,
    PinocchioInterface pinocchioInterface,
    const MpcRobotModelBase<scalar_t>* absl_nonnull mpcRobotModel,
    std::unique_ptr<AngularCenterOfMass> acom)
    : Q_com_(std::move(Q_com)),
      Q_acom_(std::move(Q_acom)),
      pinocchioInterface_(std::move(pinocchioInterface)),
      mpcRobotModelPtr_(ABSL_DIE_IF_NULL(mpcRobotModel)),
      acom_(std::move(acom)) {}

ComAndAcomTrackingCost::ComAndAcomTrackingCost(
    const ComAndAcomTrackingCost& rhs)
    : StateCost(rhs),
      Q_com_(rhs.Q_com_),
      Q_acom_(rhs.Q_acom_),
      pinocchioInterface_(rhs.pinocchioInterface_),
      mpcRobotModelPtr_(rhs.mpcRobotModelPtr_),
      // A deep copy of the already-checked evaluator: the clone runs the same
      // network without re-resolving it.
      acom_(std::make_unique<AngularCenterOfMass>(*rhs.acom_)) {}

ComAndAcomTrackingCost* absl_nonnull ComAndAcomTrackingCost::clone() const {
  return new ComAndAcomTrackingCost(*this);
}

vector4_t ComAndAcomTrackingCost::acomAttitudeOf(const vector_t& state) const {
  return acom_->acomAttitude(mpcRobotModelPtr_->getBaseOrientation(state),
                             mpcRobotModelPtr_->getJointAngles(state));
}

scalar_t ComAndAcomTrackingCost::getValue(
    scalar_t time,
    const vector_t& state,
    const TargetTrajectories& targetTrajectories,
    const PreComputation& /*preComp*/) const {
  const vector_t stateRef =
      mpcRobotModelPtr_->getDesiredState(targetTrajectories, time);
  const vector_t q = mpcRobotModelPtr_->getGeneralizedCoordinates(state);
  const vector_t qRef = mpcRobotModelPtr_->getGeneralizedCoordinates(stateRef);

  pinocchio::Data& pinocchioData = pinocchioInterface_.getData();
  const pinocchio::Model& pinocchioModel = pinocchioInterface_.getModel();

  // centerOfMass returns a reference into pinocchioData, so p_com must be
  // copied out before the reference configuration overwrites it.
  const vector3_t p_com =
      pinocchio::centerOfMass(pinocchioModel, pinocchioData, q);
  const vector3_t p_com_ref =
      pinocchio::centerOfMass(pinocchioModel, pinocchioData, qRef);
  const vector3_t p_err = p_com - p_com_ref;
  const vector3_t theta_err =
      headingTiltResidual(acomAttitudeOf(state), acomAttitudeOf(stateRef));

  return 0.5 * p_err.dot(Q_com_ * p_err) +
         0.5 * theta_err.dot(Q_acom_ * theta_err);
}

ScalarFunctionQuadraticApproximation
ComAndAcomTrackingCost::getQuadraticApproximation(
    scalar_t time,
    const vector_t& state,
    const TargetTrajectories& targetTrajectories,
    const PreComputation& /*preComp*/) const {
  ScalarFunctionQuadraticApproximation approx;

  const vector_t stateRef =
      mpcRobotModelPtr_->getDesiredState(targetTrajectories, time);
  const vector_t q = mpcRobotModelPtr_->getGeneralizedCoordinates(state);
  const vector_t qRef = mpcRobotModelPtr_->getGeneralizedCoordinates(stateRef);

  pinocchio::Data& pinocchioData = pinocchioInterface_.getData();
  const pinocchio::Model& pinocchioModel = pinocchioInterface_.getModel();

  // ---- CoM position and Jacobian ----
  // p_com and J_com_v are deep copies, because the reference CoM evaluation
  // below overwrites the same pinocchioData fields.
  pinocchio::jacobianCenterOfMass(pinocchioModel, pinocchioData, q);
  const vector3_t p_com = pinocchioData.com[0];
  const matrix_t J_com_v =
      pinocchioData
          .Jcom;  // 3 x nv, columns in Pinocchio's configuration tangent

  const vector3_t p_com_ref =
      pinocchio::centerOfMass(pinocchioModel, pinocchioData, qRef);
  const vector3_t p_err = p_com - p_com_ref;

  // ---- CoM Jacobian w.r.t. the state ----
  // Pinocchio's CoM Jacobian is with respect to its configuration tangent v;
  // the robot model's getPinocchioTangentJacobian() is how a state step moves q
  // in that tangent (nv x nx). Whatever coordinates the root joint has, that
  // map is the whole story - the selection of the configuration slots where v
  // is the time derivative of the stored coordinates, the lift of the body
  // rotation to the stored coefficients where it is not - so no rate mapping of
  // the base orientation is chained on here by hand:
  // testComAndAcomTrackingCost.cpp compares dfdx with central differences of
  // getValue at a tilted base.
  const matrix_t J_com_dx =
      J_com_v * mpcRobotModelPtr_->getPinocchioTangentJacobian(state);

  // ---- ACoM heading-tilt residual and its Jacobian w.r.t. the state ----
  //
  // xi_a = xi_hat (x) Exp(Delta_theta(q_j)) moves by the body tangent dphi_a =
  // R(Delta_theta)^T dphi + J_r(Delta_theta) J_Delta_theta dq_j
  // (AcomAttitudeTangentJacobian), and e_HT by J_HT(xi_a) dphi_a; dphi is the
  // base rotation's body tangent, getBaseRotationTangentJacobian() times the
  // step of the orientation slots. Nothing else in the state moves the ACoM.
  const vector_t qJoints = mpcRobotModelPtr_->getJointAngles(state);
  const HeadingTiltError acomError = headingTiltError(
      acom_->acomAttitude(mpcRobotModelPtr_->getBaseOrientation(state),
                          qJoints),
      acomAttitudeOf(stateRef));
  const AcomAttitudeTangentJacobian acomTangent =
      acom_->acomAttitudeTangentJacobian(qJoints);
  const matrix_t baseRotationJacobian =
      mpcRobotModelPtr_->getBaseRotationTangentJacobian(
          state);  // 3 x orientation slots
  matrix_t J_acom_dx = matrix_t::Zero(
      kRotationTangentDim,
      static_cast<Eigen::Index>(mpcRobotModelPtr_->getStateDim()));
  J_acom_dx.middleCols(static_cast<Eigen::Index>(
                           mpcRobotModelPtr_->getBaseOrientationStartIndex()),
                       baseRotationJacobian.cols()) =
      acomError.jacobianBodyTangent * acomTangent.baseOrientation *
      baseRotationJacobian;
  J_acom_dx.middleCols(
      static_cast<Eigen::Index>(mpcRobotModelPtr_->getJointStartindex()),
      acomTangent.jointAngles.cols()) =
      acomError.jacobianBodyTangent * acomTangent.jointAngles;
  const vector3_t& theta_err = acomError.residual;

  // ---- Gauss-Newton quadratic approximation ----
  approx.f = 0.5 * p_err.dot(Q_com_ * p_err) +
             0.5 * theta_err.dot(Q_acom_ * theta_err);
  approx.dfdx = J_com_dx.transpose() * Q_com_ * p_err +
                J_acom_dx.transpose() * Q_acom_ * theta_err;
  approx.dfdxx = J_com_dx.transpose() * Q_com_ * J_com_dx +
                 J_acom_dx.transpose() * Q_acom_ * J_acom_dx;
  return approx;
}

void ComAndAcomTrackingCost::zeroBasePoseWeights(matrix_t& Q) {
  Q.block(kBasePoseTuningRow, kBasePoseTuningRow, kBasePoseTuningDim,
          kBasePoseTuningDim)
      .setZero();
}

absl::Status ComAndAcomTrackingCost::setWeights(matrix_t Q_com,
                                                matrix_t Q_acom) {
  RETURN_IF_ERROR(validateWeights(Q_com, Q_acom));
  Q_com_ = std::move(Q_com);
  Q_acom_ = std::move(Q_acom);
  return absl::OkStatus();
}

}  // namespace ocs2::humanoid
