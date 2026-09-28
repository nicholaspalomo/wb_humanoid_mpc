#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include "humanoid_common_mpc/cost/ComAndAcomTrackingCost.h"

#include <ocs2_centroidal_model/AccessHelperFunctions.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/StatusMacros.h"

namespace ocs2::humanoid {

namespace {

/// The centroidal state layout is
///   x = [h_norm(6), p_base(3), euler_zyx_base(3), q_joints(nj)]
/// so the generalized coordinates always start at index 6, and the base
/// orientation at index 9. These constants keep the magic numbers out of the
/// Jacobian assembly below.
constexpr std::size_t kGeneralizedCoordinatesStartIndex = 6;
constexpr std::size_t kBasePositionStateIndex = kGeneralizedCoordinatesStartIndex;
constexpr std::size_t kBaseOrientationStateIndex = kGeneralizedCoordinatesStartIndex + 3;
constexpr std::size_t kJointStateIndex = kGeneralizedCoordinatesStartIndex + 6;

/// Generalized coordinates of the floating base: 3 translations and 3 ZYX Euler angles.
constexpr int kGeneralizedBaseDim = 6;

/// The base pose in the state: p_base followed by the ZYX Euler angles, which this cost replaces.
constexpr Eigen::Index kBasePoseStateIndex = kBasePositionStateIndex;
constexpr Eigen::Index kBasePoseDim = kGeneralizedBaseDim;

/**
 * Loads the aCOM network for `robotName` and checks it against the model the cost evaluates: the network's joints,
 * name by name, against the model's actuated joints, which also pins the count to info.actuatedDofNum.
 */
absl::StatusOr<std::unique_ptr<AngularCenterOfMass>> createCheckedAcom(const PinocchioInterface& pinocchioInterface,
                                                                       const CentroidalModelInfo& info,
                                                                       absl::string_view robotName) {
  ASSIGN_OR_RETURN(const std::vector<std::string> modelJointNames, ComAndAcomTrackingCost::actuatedJointNames(pinocchioInterface, info));
  return AngularCenterOfMass::Create(robotName, modelJointNames);
}

}  // namespace

absl::StatusOr<std::vector<std::string>> ComAndAcomTrackingCost::actuatedJointNames(const PinocchioInterface& pinocchioInterface,
                                                                                    const CentroidalModelInfo& info) {
  const pinocchio::Model& model = pinocchioInterface.getModel();
  if (model.nq != kGeneralizedBaseDim + static_cast<int>(info.actuatedDofNum)) {
    return absl::InvalidArgumentError(absl::StrCat("[ComAndAcomTrackingCost] CentroidalModelInfo.actuatedDofNum (", info.actuatedDofNum,
                                                   ") does not describe the Pinocchio model, which has ", model.nq - kGeneralizedBaseDim,
                                                   " joint coordinates after its floating base."));
  }
  // model.names starts with the universe and the floating-base joint; the actuated joints are the rest, in the order
  // the state stores them. Counting from the end keeps this independent of how many joints the base is built from.
  const std::size_t numActuated = static_cast<std::size_t>(info.actuatedDofNum);
  if (model.names.size() < numActuated) {
    return absl::InvalidArgumentError(absl::StrCat("[ComAndAcomTrackingCost] the Pinocchio model names ", model.names.size(),
                                                   " joints, fewer than actuatedDofNum (", numActuated, ")."));
  }
  return std::vector<std::string>(model.names.end() - static_cast<std::ptrdiff_t>(numActuated), model.names.end());
}

absl::Status ComAndAcomTrackingCost::validateWeights(const matrix_t& Q_com, const matrix_t& Q_acom) {
  if (Q_com.rows() != 3 || Q_com.cols() != 3) {
    return absl::InvalidArgumentError(
        absl::StrCat("[ComAndAcomTrackingCost] Q_com in task.yaml must be 3x3, got ", Q_com.rows(), "x", Q_com.cols(), "."));
  }
  if (Q_acom.rows() != 3 || Q_acom.cols() != 3) {
    return absl::InvalidArgumentError(
        absl::StrCat("[ComAndAcomTrackingCost] Q_acom in task.yaml must be 3x3, got ", Q_acom.rows(), "x", Q_acom.cols(), "."));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<ComAndAcomTrackingCost>> ComAndAcomTrackingCost::Create(
    matrix_t Q_com, matrix_t Q_acom, PinocchioInterface pinocchioInterface, CentroidalModelInfo info, absl::string_view robotName) {
  RETURN_IF_ERROR(validateWeights(Q_com, Q_acom));
  ASSIGN_OR_RETURN(std::unique_ptr<AngularCenterOfMass> acom, createCheckedAcom(pinocchioInterface, info, robotName));
  // The constructor is private, so std::make_unique cannot reach it.
  return std::unique_ptr<ComAndAcomTrackingCost>(
      new ComAndAcomTrackingCost(std::move(Q_com), std::move(Q_acom), std::move(pinocchioInterface), std::move(info), std::move(acom)));
}

ComAndAcomTrackingCost::ComAndAcomTrackingCost(
    matrix_t Q_com, matrix_t Q_acom, PinocchioInterface pinocchioInterface, CentroidalModelInfo info, const std::string& robotName)
    : Q_com_(std::move(Q_com)), Q_acom_(std::move(Q_acom)), pinocchioInterface_(std::move(pinocchioInterface)), info_(std::move(info)) {
  const absl::Status weights = validateWeights(Q_com_, Q_acom_);
  if (!weights.ok()) {
    throw std::runtime_error(std::string(weights.message()));
  }
  absl::StatusOr<std::unique_ptr<AngularCenterOfMass>> acom = createCheckedAcom(pinocchioInterface_, info_, robotName);
  if (!acom.ok()) {
    throw std::runtime_error(std::string(acom.status().message()));
  }
  acom_ = *std::move(acom);
}

ComAndAcomTrackingCost::ComAndAcomTrackingCost(matrix_t Q_com,
                                               matrix_t Q_acom,
                                               PinocchioInterface pinocchioInterface,
                                               CentroidalModelInfo info,
                                               std::unique_ptr<AngularCenterOfMass> acom)
    : Q_com_(std::move(Q_com)),
      Q_acom_(std::move(Q_acom)),
      pinocchioInterface_(std::move(pinocchioInterface)),
      info_(std::move(info)),
      acom_(std::move(acom)) {}

ComAndAcomTrackingCost::ComAndAcomTrackingCost(const ComAndAcomTrackingCost& rhs)
    : StateCost(rhs),
      Q_com_(rhs.Q_com_),
      Q_acom_(rhs.Q_acom_),
      pinocchioInterface_(rhs.pinocchioInterface_),
      info_(rhs.info_),
      // A deep copy of the already-checked evaluator: the clone runs the same network without re-resolving it.
      acom_(std::make_unique<AngularCenterOfMass>(*rhs.acom_)) {}

ComAndAcomTrackingCost* ComAndAcomTrackingCost::clone() const {
  return new ComAndAcomTrackingCost(*this);
}

vector3_t ComAndAcomTrackingCost::computeAcomError(const vector_t& state, const vector_t& stateRef) const {
  // The joint angles sit directly in the state, so there is no need to assemble
  // the full generalized configuration just to read them.
  const vector_t qJoints = state.segment(kJointStateIndex, info_.actuatedDofNum);
  const vector_t qJointsRef = stateRef.segment(kJointStateIndex, info_.actuatedDofNum);

  // State Euler angles are in ZYX order: [Yaw, Pitch, Roll]. The aCOM network
  // emits XYZ order [Roll, Pitch, Yaw], so its output is reordered to match.
  const vector3_t eulerAnglesZYX = state.segment<3>(kBaseOrientationStateIndex);
  const vector3_t eulerAnglesZYXRef = stateRef.segment<3>(kBaseOrientationStateIndex);
  const vector3_t thetaAcom = eulerAnglesZYX + acomXyzToZyx(acom_->computeJointOrientationOffset(qJoints));
  const vector3_t thetaAcomRef = eulerAnglesZYXRef + acomXyzToZyx(acom_->computeJointOrientationOffset(qJointsRef));

  vector3_t thetaError = thetaAcom - thetaAcomRef;
  // Wrap the yaw error to [-pi, pi] so that crossing +-pi is not a discontinuity.
  thetaError[0] = moduloAngleWithReference(thetaAcom[0], thetaAcomRef[0]) - thetaAcomRef[0];
  return thetaError;
}

scalar_t ComAndAcomTrackingCost::getValue(scalar_t time,
                                          const vector_t& state,
                                          const TargetTrajectories& targetTrajectories,
                                          const PreComputation& preComp) const {
  const vector_t stateRef = targetTrajectories.getDesiredState(time);
  const vector_t q = centroidal_model::getGeneralizedCoordinates(state, info_);
  const vector_t qRef = centroidal_model::getGeneralizedCoordinates(stateRef, info_);

  pinocchio::Data& pinocchioData = pinocchioInterface_.getData();
  const pinocchio::Model& pinocchioModel = pinocchioInterface_.getModel();

  // centerOfMass returns a reference into pinocchioData, so p_com must be copied
  // out before the reference configuration overwrites it.
  const vector3_t p_com = pinocchio::centerOfMass(pinocchioModel, pinocchioData, q);
  const vector3_t p_com_ref = pinocchio::centerOfMass(pinocchioModel, pinocchioData, qRef);
  const vector3_t p_err = p_com - p_com_ref;
  const vector3_t theta_err = computeAcomError(state, stateRef);

  return 0.5 * p_err.dot(Q_com_ * p_err) + 0.5 * theta_err.dot(Q_acom_ * theta_err);
}

ScalarFunctionQuadraticApproximation ComAndAcomTrackingCost::getQuadraticApproximation(scalar_t time,
                                                                                       const vector_t& state,
                                                                                       const TargetTrajectories& targetTrajectories,
                                                                                       const PreComputation& preComp) const {
  ScalarFunctionQuadraticApproximation approx;

  const vector_t stateRef = targetTrajectories.getDesiredState(time);
  const vector_t q = centroidal_model::getGeneralizedCoordinates(state, info_);
  const vector_t qRef = centroidal_model::getGeneralizedCoordinates(stateRef, info_);

  pinocchio::Data& pinocchioData = pinocchioInterface_.getData();
  const pinocchio::Model& pinocchioModel = pinocchioInterface_.getModel();

  // ---- CoM position and Jacobian ----
  // p_com and J_com_v are deep copies, because the reference CoM evaluation below
  // overwrites the same pinocchioData fields.
  pinocchio::jacobianCenterOfMass(pinocchioModel, pinocchioData, q);
  const vector3_t p_com = pinocchioData.com[0];
  const matrix_t J_com_v = pinocchioData.Jcom;  // 3 x nv

  const vector3_t p_com_ref = pinocchio::centerOfMass(pinocchioModel, pinocchioData, qRef);
  const vector3_t p_err = p_com - p_com_ref;
  const vector3_t theta_err = computeAcomError(state, stateRef);

  // ---- CoM Jacobian w.r.t. the centroidal state ----
  //
  // The floating base of this Pinocchio model is a composite of a translation
  // joint and a SphericalZYX joint (see getBaseJointcomposite in
  // createPinocchioModel.cpp), so the tangent vector v IS the time derivative of
  // the generalized coordinates:
  //   v = [dp_base/dt, d(euler_zyx)/dt, dq_j/dt]
  // In particular v[3:6] is the Euler angle rate, NOT the angular velocity omega.
  // Pinocchio's CoM Jacobian is therefore already dp_com/dq, and no Euler-rate to
  // angular-velocity mapping may be chained onto its base orientation columns.
  // testComAndAcomTrackingCost.cpp pins this convention by comparing dfdx with
  // central differences of getValue at a tilted base.
  //
  // Build dp_com/dx by hand, since CentroidalModelPinocchioMapping does not
  // expose a configuration Jacobian for the centroidal formulation.
  matrix_t J_com_dx = matrix_t::Zero(3, info_.stateDim);
  J_com_dx.block(0, kBasePositionStateIndex, 3, 3) = J_com_v.leftCols(3);
  J_com_dx.block(0, kBaseOrientationStateIndex, 3, 3) = J_com_v.middleCols(3, 3);
  J_com_dx.block(0, kJointStateIndex, 3, info_.actuatedDofNum) = J_com_v.rightCols(info_.actuatedDofNum);

  // ---- aCOM Jacobian w.r.t. the centroidal state ----
  //
  // theta_acom = euler_zyx_base + P * Delta_theta(q_j), so
  //   d(theta_acom)/dx = [0_{3x6} | 0_{3x3} | I_{3x3} | P * J_Delta_theta]
  //                       h_norm   p_base    euler     q_joints
  matrix_t J_acom_dx = matrix_t::Zero(3, info_.stateDim);
  J_acom_dx.block<3, 3>(0, kBaseOrientationStateIndex).setIdentity();
  J_acom_dx.block(0, kJointStateIndex, 3, info_.actuatedDofNum) =
      acomJacobianXyzToZyx(acom_->computeJointOffsetJacobian(q.tail(info_.actuatedDofNum)));

  // ---- Gauss-Newton quadratic approximation ----
  approx.f = 0.5 * p_err.dot(Q_com_ * p_err) + 0.5 * theta_err.dot(Q_acom_ * theta_err);
  approx.dfdx = J_com_dx.transpose() * Q_com_ * p_err + J_acom_dx.transpose() * Q_acom_ * theta_err;
  approx.dfdxx = J_com_dx.transpose() * Q_com_ * J_com_dx + J_acom_dx.transpose() * Q_acom_ * J_acom_dx;
  return approx;
}

void ComAndAcomTrackingCost::zeroBasePoseWeights(matrix_t& Q) {
  Q.block(kBasePoseStateIndex, kBasePoseStateIndex, kBasePoseDim, kBasePoseDim).setZero();
}

void ComAndAcomTrackingCost::setWeights(matrix_t Q_com, matrix_t Q_acom) {
  const absl::Status status = validateWeights(Q_com, Q_acom);
  if (!status.ok()) {
    throw std::invalid_argument(std::string(status.message()));
  }
  Q_com_ = std::move(Q_com);
  Q_acom_ = std::move(Q_acom);
}

}  // namespace ocs2::humanoid
