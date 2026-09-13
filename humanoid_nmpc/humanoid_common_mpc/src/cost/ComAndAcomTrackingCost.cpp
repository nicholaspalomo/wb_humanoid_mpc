#include "humanoid_common_mpc/cost/ComAndAcomTrackingCost.h"
#include <ocs2_centroidal_model/CentroidalModelPinocchioMapping.h>
#include <pinocchio/algorithm/center-of-mass.hpp>

namespace ocs2::humanoid {

ComAndAcomTrackingCost::ComAndAcomTrackingCost(matrix_t Q_com,
                                               matrix_t Q_acom,
                                               PinocchioInterface pinocchioInterface,
                                               CentroidalModelInfo info,
                                               const SwitchedModelReferenceManager& referenceManager)
    : Q_com_(std::move(Q_com)),
      Q_acom_(std::move(Q_acom)),
      pinocchioInterface_(std::move(pinocchioInterface)),
      info_(std::move(info)),
      referenceManagerPtr_(&referenceManager),
      acom_(AngularCenterOfMass::createFromStaticWeights()) {}

ComAndAcomTrackingCost::ComAndAcomTrackingCost(const ComAndAcomTrackingCost& rhs)
    : StateCost(rhs),
      Q_com_(rhs.Q_com_),
      Q_acom_(rhs.Q_acom_),
      pinocchioInterface_(rhs.pinocchioInterface_),
      info_(rhs.info_),
      referenceManagerPtr_(rhs.referenceManagerPtr_),
      acom_(AngularCenterOfMass::createFromStaticWeights()) {}

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

  // Need a separate data object for ref to avoid modifying the same one?
  // pinocchio::centerOfMass has an overload without data, but it allocates.
  // Actually, we can just use the same data object sequentially.
  const vector3_t p_com_ref = pinocchio::centerOfMass(pinocchioModel, pinocchioData, qRef);

  const vector3_t p_err = p_com - p_com_ref;

  // Compute ACoM
  // acom_->computeAcomOrientation takes qJoints, but wait!
  // q in pinocchio mapping includes [p_base, quat_base, q_joints].
  // qJoints starts at index 7.
  const vector_t qJoints = q.tail(info_.actuatedDofNum);
  const vector_t qJointsRef = qRef.tail(info_.actuatedDofNum);

  // theta_acom = theta_base + Delta_theta(q_joints)
  // Wait, in Centroidal dynamics, the state already has euler angles (ZYX) for the base!
  // It's at index info_.generalizedCoordinatesIndex + 3.
  const vector3_t eulerAngles = state.segment<3>(info_.generalizedCoordinatesIndex + 3);
  const vector3_t eulerAnglesRef = stateRef.segment<3>(info_.generalizedCoordinatesIndex + 3);

  const vector3_t deltaTheta = acom_->computeJointOrientationOffset(qJoints);
  const vector3_t deltaThetaRef = acom_->computeJointOrientationOffset(qJointsRef);

  const vector3_t theta_acom = eulerAngles + deltaTheta;
  const vector3_t theta_acom_ref = eulerAnglesRef + deltaThetaRef;

  const vector3_t theta_err = theta_acom - theta_acom_ref;

  scalar_t cost = 0.5 * p_err.dot(Q_com_ * p_err) + 0.5 * theta_err.dot(Q_acom_ * theta_err);
  return cost;
}

ScalarFunctionQuadraticApproximation ComAndAcomTrackingCost::getQuadraticApproximation(scalar_t time,
                                                                                       const vector_t& state,
                                                                                       const TargetTrajectories& targetTrajectories,
                                                                                       const PreComputation& preComp) const {
  ScalarFunctionQuadraticApproximation approx;
  approx.f = getValue(time, state, targetTrajectories, preComp);

  approx.dfdx = vector_t::Zero(info_.stateDim);
  approx.dfdxx = matrix_t::Zero(info_.stateDim, info_.stateDim);

  CentroidalModelPinocchioMapping mapping(info_);
  const vector_t q = mapping.getPinocchioJointPosition(state);

  const vector_t stateRef = targetTrajectories.getDesiredState(time);
  const vector_t qRef = mapping.getPinocchioJointPosition(stateRef);

  auto& pinocchioData = pinocchioInterface_.getData();
  const auto& pinocchioModel = pinocchioInterface_.getModel();

  const vector3_t p_com = pinocchio::centerOfMass(pinocchioModel, pinocchioData, q);
  pinocchio::jacobianCenterOfMass(pinocchioModel, pinocchioData, q);  // updates Jcom
  matrix_t J_com = pinocchioData.Jcom;                                // 3 x (6 + nj)

  const vector3_t p_com_ref = pinocchio::centerOfMass(pinocchioModel, pinocchioData, qRef);
  const vector3_t p_err = p_com - p_com_ref;

  // ACoM
  const vector_t qJoints = q.tail(info_.actuatedDofNum);
  const vector_t qJointsRef = qRef.tail(info_.actuatedDofNum);

  const vector3_t eulerAngles = state.segment<3>(info_.generalizedCoordinatesIndex + 3);
  const vector3_t eulerAnglesRef = stateRef.segment<3>(info_.generalizedCoordinatesIndex + 3);

  const vector3_t deltaTheta = acom_->computeJointOrientationOffset(qJoints);
  const vector3_t deltaThetaRef = acom_->computeJointOrientationOffset(qJointsRef);

  const vector3_t theta_acom = eulerAngles + deltaTheta;
  const vector3_t theta_acom_ref = eulerAnglesRef + deltaThetaRef;
  const vector3_t theta_err = theta_acom - theta_acom_ref;

  matrix_t J_acom_joints = acom_->computeJointOrientationOffsetJacobian(qJoints);  // 3 x nj

  // We need to map Jacobians from pinocchio (v) to state (dx)
  matrix_t dq_dx = mapping.getPinocchioJointVelocityToStateMapping(state);

  // For CoM: J_com maps v to dp_com.
  // dp_com / dx = J_com * dq_dx
  matrix_t J_com_dx = J_com * dq_dx;

  // For ACoM:
  // theta_acom = eulerAngles + deltaTheta(q_joints)
  // d(theta_acom)/dx = d(eulerAngles)/dx + d(deltaTheta)/d(q_joints) * d(q_joints)/dx
  matrix_t J_acom_dx = matrix_t::Zero(3, info_.stateDim);
  // eulerAngles are at info_.generalizedCoordinatesIndex + 3
  J_acom_dx.block<3, 3>(0, info_.generalizedCoordinatesIndex + 3).setIdentity();

  // d(q_joints)/dx is just a selection matrix in Centroidal state.
  // In centroidal state x = [h, p_base, euler_base, q_joints]
  // q_joints starts at info_.generalizedCoordinatesIndex + 6
  J_acom_dx.block(0, info_.generalizedCoordinatesIndex + 6, 3, info_.actuatedDofNum) += J_acom_joints;

  // First order derivative: J^T * Q * err
  approx.dfdx = J_com_dx.transpose() * Q_com_ * p_err + J_acom_dx.transpose() * Q_acom_ * theta_err;

  // Second order (Gauss-Newton approx): J^T * Q * J
  approx.dfdxx = J_com_dx.transpose() * Q_com_ * J_com_dx + J_acom_dx.transpose() * Q_acom_ * J_acom_dx;

  return approx;
}

}  // namespace ocs2::humanoid
