#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include "humanoid_common_mpc/cost/ComAndAcomTrackingCost.h"

#include <ocs2_centroidal_model/CentroidalModelPinocchioMapping.h>
#include <ocs2_robotic_tools/common/RotationDerivativesTransforms.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>
#include <pinocchio/algorithm/center-of-mass.hpp>

#include <stdexcept>

namespace ocs2::humanoid {

namespace {

/// The centroidal state layout is [h_norm(6), q_generalized(nq)], so
/// generalized coordinates always start at index 6. This constant avoids
/// scattering magic numbers through the Jacobian assembly code.
constexpr size_t kGeneralizedCoordinatesStartIndex = 6;

/**
 * Reorder an XYZ (Roll, Pitch, Yaw) 3-vector to ZYX (Yaw, Pitch, Roll).
 *
 * The ACoM SIREN network outputs delta_theta in XYZ order [roll, pitch, yaw],
 * but OCS2 centroidal state uses ZYX Euler angles [yaw, pitch, roll].
 */
inline vector3_t xyzToZyx(const vector3_t& v) {
  return vector3_t(v[2], v[1], v[0]);
}

/**
 * Reorder the rows of a (3 x n) Jacobian from XYZ output convention to ZYX.
 * Swaps row 0 (Roll) ↔ row 2 (Yaw), keeping row 1 (Pitch) in place.
 */
inline matrix_t reorderJacobianRowsXyzToZyx(const matrix_t& J) {
  matrix_t J_zyx(J.rows(), J.cols());
  J_zyx.row(0) = J.row(2);  // Yaw
  J_zyx.row(1) = J.row(1);  // Pitch
  J_zyx.row(2) = J.row(0);  // Roll
  return J_zyx;
}

}  // namespace

ComAndAcomTrackingCost::ComAndAcomTrackingCost(matrix_t Q_com,
                                               matrix_t Q_acom,
                                               PinocchioInterface pinocchioInterface,
                                               CentroidalModelInfo info,
                                               const SwitchedModelReferenceManager& referenceManager,
                                               const std::string& robotName)
    : Q_com_(std::move(Q_com)),
      Q_acom_(std::move(Q_acom)),
      pinocchioInterface_(std::move(pinocchioInterface)),
      info_(std::move(info)),
      referenceManagerPtr_(&referenceManager),
      robotName_(robotName),
      acom_(AngularCenterOfMass::createForRobot(robotName)) {
  // Runtime dimension check: the SIREN network must have been trained with the
  // same number of joints as the centroidal model expects.
  if (acom_->getInputDim() != static_cast<size_t>(info_.actuatedDofNum)) {
    throw std::runtime_error("[ComAndAcomTrackingCost] ACoM SIREN input_dim (" + std::to_string(acom_->getInputDim()) +
                             ") != actuatedDofNum (" + std::to_string(info_.actuatedDofNum) + "). Regenerate AcomSirenWeights" + robotName +
                             ".h with matching robot model.");
  }
}

ComAndAcomTrackingCost::ComAndAcomTrackingCost(const ComAndAcomTrackingCost& rhs)
    : StateCost(rhs),
      Q_com_(rhs.Q_com_),
      Q_acom_(rhs.Q_acom_),
      pinocchioInterface_(rhs.pinocchioInterface_),
      info_(rhs.info_),
      referenceManagerPtr_(rhs.referenceManagerPtr_),
      robotName_(rhs.robotName_),
      acom_(AngularCenterOfMass::createForRobot(rhs.robotName_)) {}

ComAndAcomTrackingCost* ComAndAcomTrackingCost::clone() const {
  return new ComAndAcomTrackingCost(*this);
}

scalar_t ComAndAcomTrackingCost::getValue(scalar_t time,
                                          const vector_t& state,
                                          const TargetTrajectories& targetTrajectories,
                                          const PreComputation& preComp) const {
  CentroidalModelPinocchioMapping mapping(info_);
  const vector_t q = mapping.getPinocchioJointPosition(state);

  // Interpolate reference
  const vector_t stateRef = targetTrajectories.getDesiredState(time);
  const vector_t qRef = mapping.getPinocchioJointPosition(stateRef);

  auto& pinocchioData = pinocchioInterface_.getData();
  const auto& pinocchioModel = pinocchioInterface_.getModel();

  // Compute current and ref CoM
  const vector3_t p_com = pinocchio::centerOfMass(pinocchioModel, pinocchioData, q);
  const vector3_t p_com_ref = pinocchio::centerOfMass(pinocchioModel, pinocchioData, qRef);
  const vector3_t p_err = p_com - p_com_ref;

  // ---- ACoM orientation error ----
  const vector_t qJoints = q.tail(info_.actuatedDofNum);
  const vector_t qJointsRef = qRef.tail(info_.actuatedDofNum);

  // State Euler angles are in ZYX order: [Yaw, Pitch, Roll]
  const vector3_t eulerAnglesZYX = state.segment<3>(kGeneralizedCoordinatesStartIndex + 3);
  const vector3_t eulerAnglesZYXRef = stateRef.segment<3>(kGeneralizedCoordinatesStartIndex + 3);

  // ACoM SIREN output is in XYZ order [Roll, Pitch, Yaw]; convert to ZYX
  // to match the centroidal state's Euler angle convention.
  const vector3_t deltaThetaZYX = xyzToZyx(acom_->computeJointOrientationOffset(qJoints));
  const vector3_t deltaThetaZYXRef = xyzToZyx(acom_->computeJointOrientationOffset(qJointsRef));

  const vector3_t theta_acom = eulerAnglesZYX + deltaThetaZYX;
  const vector3_t theta_acom_ref = eulerAnglesZYXRef + deltaThetaZYXRef;

  vector3_t theta_err = theta_acom - theta_acom_ref;
  // Wrap yaw error to [-π, π] to avoid discontinuities at ±π.
  theta_err[0] = moduloAngleWithReference(theta_acom[0], theta_acom_ref[0]) - theta_acom_ref[0];

  scalar_t cost = 0.5 * p_err.dot(Q_com_ * p_err) + 0.5 * theta_err.dot(Q_acom_ * theta_err);
  return cost;
}

ScalarFunctionQuadraticApproximation ComAndAcomTrackingCost::getQuadraticApproximation(scalar_t time,
                                                                                       const vector_t& state,
                                                                                       const TargetTrajectories& targetTrajectories,
                                                                                       const PreComputation& preComp) const {
  ScalarFunctionQuadraticApproximation approx;
  approx.dfdx = vector_t::Zero(info_.stateDim);
  approx.dfdxx = matrix_t::Zero(info_.stateDim, info_.stateDim);

  CentroidalModelPinocchioMapping mapping(info_);
  const vector_t q = mapping.getPinocchioJointPosition(state);
  const vector_t stateRef = targetTrajectories.getDesiredState(time);
  const vector_t qRef = mapping.getPinocchioJointPosition(stateRef);

  auto& pinocchioData = pinocchioInterface_.getData();
  const auto& pinocchioModel = pinocchioInterface_.getModel();

  // ---- CoM position + Jacobian ----
  // IMPORTANT: Deep-copy p_com and J_com before computing the reference,
  // because the next pinocchio call overwrites pinocchioData.
  pinocchio::jacobianCenterOfMass(pinocchioModel, pinocchioData, q);
  const vector3_t p_com = pinocchioData.com[0];
  const matrix_t J_com_v = pinocchioData.Jcom;  // 3 × nv, deep copy

  // Reference CoM (overwrites pinocchioData.com[0] — safe because p_com was copied)
  const vector3_t p_com_ref = pinocchio::centerOfMass(pinocchioModel, pinocchioData, qRef);
  const vector3_t p_err = p_com - p_com_ref;

  // ---- ACoM orientation error ----
  const vector_t qJoints = q.tail(info_.actuatedDofNum);
  const vector_t qJointsRef = qRef.tail(info_.actuatedDofNum);

  const vector3_t eulerAnglesZYX = state.segment<3>(kGeneralizedCoordinatesStartIndex + 3);
  const vector3_t eulerAnglesZYXRef = stateRef.segment<3>(kGeneralizedCoordinatesStartIndex + 3);

  const vector3_t deltaThetaZYX = xyzToZyx(acom_->computeJointOrientationOffset(qJoints));
  const vector3_t deltaThetaZYXRef = xyzToZyx(acom_->computeJointOrientationOffset(qJointsRef));

  const vector3_t theta_acom = eulerAnglesZYX + deltaThetaZYX;
  const vector3_t theta_acom_ref = eulerAnglesZYXRef + deltaThetaZYXRef;

  vector3_t theta_err = theta_acom - theta_acom_ref;
  theta_err[0] = moduloAngleWithReference(theta_acom[0], theta_acom_ref[0]) - theta_acom_ref[0];

  // ---- CoM Jacobian w.r.t. centroidal state ----
  //
  // Centroidal state:  x = [h_norm(6), p_base(3), θ_base_zyx(3), q_joints(nj)]
  //   Index ranges:         0..5        6..8        9..11          12..12+nj
  //
  // J_com_v maps generalized velocity v → CoM velocity:
  //   v_com = J_com_v * v,   where v = [v_base(3), ω_base(3), dq_j(nj)]
  //
  // For base translation: v[0:3] = dp_base/dt, so J_com_v[:, 0:3] maps directly.
  // For base rotation:    v[3:6] = ω (angular velocity), NOT dθ_zyx/dt.
  //                       ω = T(θ_zyx) * dθ_zyx/dt, so dp_com/dθ = J_com_v[:, 3:6] * T(θ).
  // For joints:           v[6:] = dq_j/dt, so J_com_v[:, 6:] maps directly.
  //
  // Manually construct dp_com/dx since CentroidalModelPinocchioMapping does
  // not provide a velocity-to-state Jacobian for the centroidal formulation.

  const auto T_zyx = getMappingFromEulerAnglesZyxDerivativeToGlobalAngularVelocity<scalar_t>(eulerAnglesZYX);

  matrix_t J_com_dx = matrix_t::Zero(3, info_.stateDim);
  // Base position block: state indices [6, 7, 8]
  J_com_dx.block(0, kGeneralizedCoordinatesStartIndex, 3, 3) = J_com_v.leftCols(3);
  // Base orientation block: state indices [9, 10, 11], chain with T(θ)
  J_com_dx.block(0, kGeneralizedCoordinatesStartIndex + 3, 3, 3) = J_com_v.middleCols(3, 3) * T_zyx;
  // Joint block: state indices [12, ...]
  J_com_dx.block(0, kGeneralizedCoordinatesStartIndex + 6, 3, info_.actuatedDofNum) = J_com_v.rightCols(info_.actuatedDofNum);

  // ---- ACoM Jacobian w.r.t. centroidal state ----
  //
  // θ_acom = θ_base_zyx + P * Δθ(q_j)
  // where P is the XYZ→ZYX permutation (reorderJacobianRowsXyzToZyx).
  //
  // dθ_acom/dx = [0_{3×6} | 0_{3×3} | I_{3×3} | P * J_Δθ]
  //               h_norm   p_base    θ_base     q_joints

  const matrix_t J_acom_joints_zyx = reorderJacobianRowsXyzToZyx(acom_->computeJointOffsetJacobian(qJoints));

  matrix_t J_acom_dx = matrix_t::Zero(3, info_.stateDim);
  J_acom_dx.block<3, 3>(0, kGeneralizedCoordinatesStartIndex + 3).setIdentity();
  J_acom_dx.block(0, kGeneralizedCoordinatesStartIndex + 6, 3, info_.actuatedDofNum) = J_acom_joints_zyx;

  // ---- Gradient and Hessian (Gauss-Newton approximation) ----
  approx.dfdx = J_com_dx.transpose() * Q_com_ * p_err + J_acom_dx.transpose() * Q_acom_ * theta_err;
  approx.dfdxx = J_com_dx.transpose() * Q_com_ * J_com_dx + J_acom_dx.transpose() * Q_acom_ * J_acom_dx;
  approx.f = 0.5 * p_err.dot(Q_com_ * p_err) + 0.5 * theta_err.dot(Q_acom_ * theta_err);

  return approx;
}

}  // namespace ocs2::humanoid
