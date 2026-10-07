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

#include "pinocchio/fwd.hpp"

#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/pinocchio_model/PinocchioFrameConversions.h"

namespace ocs2::humanoid {

namespace {

constexpr size_t kExtraConstraintsCount = 7;
constexpr size_t kWrenchForceDim = 3;
constexpr size_t kWrenchMomentDim = 3;

constexpr size_t kForceXIdx = 0;
constexpr size_t kForceYIdx = 1;
constexpr size_t kForceZIdx = 2;

constexpr size_t kMomentXIdx = 0;
constexpr size_t kMomentYIdx = 1;
constexpr size_t kMomentZIdx = 2;

constexpr scalar_t kPatchOffsetZeroThreshold = 1.0e-9;
constexpr scalar_t kHalf = 0.5;

/** The cross-product matrix [v]x, with [v]x w = v x w. */
matrix3_t skewSymmetric(const vector3_t& v) {
  matrix3_t skew;
  skew << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return skew;
}

/**
 * d(getGeneralizedCoordinates(state)) / d(state), read off the model by probing it with unit states. Every model in
 * this repository selects the generalized coordinates out of the state, so this is a constant selection matrix and one
 * probe per state entry, at construction, is all it costs.
 */
matrix_t generalizedCoordinatesStateJacobian(const MpcRobotModelBase<scalar_t>& mpcRobotModel) {
  const Eigen::Index stateDim = static_cast<Eigen::Index>(mpcRobotModel.getStateDim());
  const vector_t atZero = mpcRobotModel.getGeneralizedCoordinates(vector_t::Zero(stateDim));
  matrix_t jacobian(atZero.size(), stateDim);
  for (Eigen::Index index = 0; index < stateDim; ++index) {
    jacobian.col(index) = mpcRobotModel.getGeneralizedCoordinates(vector_t::Unit(stateDim, index)) - atZero;
  }
  return jacobian;
}

}  // namespace

absl::Status ContactWrenchConeConstraint::checkWrenchSpaceInput(const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                                                size_t contactPointIndex) {
  const size_t wrenchStart = mpcRobotModel.getContactWrenchStartIndices(contactPointIndex);
  const size_t forceStart = mpcRobotModel.getContactForceStartIndices(contactPointIndex);
  const size_t momentStart = mpcRobotModel.getContactMomentStartIndices(contactPointIndex);
  const size_t inputDim = mpcRobotModel.getContactInputDim(contactPointIndex);
  if (inputDim != kContactWrenchDim || forceStart != wrenchStart || momentStart != wrenchStart + kWrenchForceDim) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[ContactWrenchConeConstraint] contact ", contactPointIndex, " is parameterized by ", inputDim, " inputs starting at ", wrenchStart,
        " (force at ", forceStart, ", moment at ", momentStart,
        "), not by its six-entry wrench. This term reads the wrench in the world frame and writes its Jacobian at the force and moment "
        "start indices, which is only right for a wrench-space model; with ",
        R"(contact_input_parameterization: "basis_vectors")",
        " the cone is enforced structurally by the non-negativity of the basis scalings instead "
        "(humanoid_nmpc/docs/contact_basis_vectors/README.md)."));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<ContactWrenchConeConstraint>> ContactWrenchConeConstraint::Create(
    const SwitchedModelReferenceManager& referenceManager,
    const ContactRectangle& contactRectangle,
    size_t contactPointIndex,
    const PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    Config config,
    bool scheduleGated) {
  RETURN_IF_ERROR(checkWrenchSpaceInput(mpcRobotModel, contactPointIndex));
  RETURN_IF_ERROR(validateConfig(config));
  return std::make_unique<ContactWrenchConeConstraint>(referenceManager, contactRectangle, contactPointIndex, pinocchioInterface,
                                                       mpcRobotModel, std::move(config), scheduleGated);
}

ContactWrenchConeConstraint::ContactWrenchConeConstraint(const SwitchedModelReferenceManager& referenceManager,
                                                         const ContactRectangle& contactRectangle,
                                                         size_t contactPointIndex,
                                                         const PinocchioInterface& pinocchioInterface,
                                                         const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                                         Config config,
                                                         bool scheduleGated)
    : StateInputConstraint(ConstraintOrder::Linear),
      referenceManagerPtr_(&referenceManager),
      pinocchioInterfacePtr_(&pinocchioInterface),
      mpcRobotModelPtr_(&mpcRobotModel),
      contactRectangle_(contactRectangle),
      contactPointIndex_(contactPointIndex),
      config_(scheduleGated ? std::move(config) : withoutLoadedFootOffsets(std::move(config))),
      scheduleGated_(scheduleGated),
      generalizedCoordinatesStateJacobian_(generalizedCoordinatesStateJacobian(mpcRobotModel)) {
  // The documented guard for the callers that construct directly; Create() returns the same refusal as a Status.
  const absl::Status wrenchSpace = checkWrenchSpaceInput(mpcRobotModel, contactPointIndex);
  CHECK(wrenchSpace.ok()) << wrenchSpace.message();
  const absl::Status validConfig = validateConfig(config_);
  CHECK(validConfig.ok()) << validConfig.message();
  // The state derivative takes the frame Jacobian, a derivative in the generalized VELOCITY, as the derivative in the
  // generalized coordinates; that is exact only when the two have the same dimension and every rate is a coordinate's.
  CHECK_EQ(pinocchioInterface.getModel().nq, pinocchioInterface.getModel().nv)
      << "[ContactWrenchConeConstraint] expects a Pinocchio model whose configuration and velocity spaces coincide";
  CHECK_EQ(generalizedCoordinatesStateJacobian_.rows(), pinocchioInterface.getModel().nq)
      << "[ContactWrenchConeConstraint] the robot model's generalized coordinates do not match the Pinocchio model";
  initializeLocalConstraintMatrix();
}

namespace {

/** InvalidArgument naming `<kConfigField>.<field>` unless `inRange`. */
absl::Status checkConeValue(bool inRange, absl::string_view field, scalar_t value, absl::string_view requirement) {
  if (inRange) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(absl::StrCat("[ContactWrenchConeConstraint] ", ContactWrenchConeConstraint::kConfigField, ".", field,
                                                 " is ", value, " but must be ", requirement, "."));
}

}  // namespace

absl::Status ContactWrenchConeConstraint::validateConfig(const Config& config) {
  RETURN_IF_ERROR(checkConeValue(config.numBasisVectors >= 3, "num_basis_vectors", static_cast<scalar_t>(config.numBasisVectors),
                                 "at least 3 (the facets of the friction pyramid)"));
  RETURN_IF_ERROR(checkConeValue(std::isfinite(config.frictionCoefficient) && config.frictionCoefficient > 0.0, "friction_coefficient",
                                 config.frictionCoefficient, "finite and positive"));
  RETURN_IF_ERROR(checkConeValue(std::isfinite(config.torsionalFrictionCoefficient) && config.torsionalFrictionCoefficient >= 0.0,
                                 "torsional_friction_coefficient", config.torsionalFrictionCoefficient, "finite and non-negative"));
  RETURN_IF_ERROR(checkConeValue(std::isfinite(config.minNormalForce) && config.minNormalForce >= 0.0, "min_normal_force",
                                 config.minNormalForce, "finite and non-negative"));
  RETURN_IF_ERROR(checkConeValue(std::isfinite(config.gripperForce) && config.gripperForce >= 0.0, "gripper_force", config.gripperForce,
                                 "finite and non-negative"));
  if (!config.patchOffset.allFinite()) {
    return absl::InvalidArgumentError(absl::StrCat("[ContactWrenchConeConstraint] the torsional reference point (Config::patchOffset) is [",
                                                   config.patchOffset.x(), ", ", config.patchOffset.y(), ", ", config.patchOffset.z(),
                                                   "] but must be finite."));
  }
  return absl::OkStatus();
}

ContactWrenchConeConstraint::Config ContactWrenchConeConstraint::withoutLoadedFootOffsets(Config config) {
  // `minNormalForce` and `gripperForce` are the only two entries of `b` (the others are homogeneous rows). Both are
  // statements about a foot the mode schedule has declared loaded: the first demands a normal force the foot cannot
  // produce in flight, the second offers an adhesion it does not have. An always-active cone evaluated on a foot at
  // zero wrench would report `-minNormalForce`, i.e. a permanent violation, and the penalty would buy it off by
  // inventing a normal force. Dropping them leaves the homogeneous cone, which the zero wrench satisfies exactly.
  if (config.minNormalForce > 0.0 || config.gripperForce > 0.0) {
    LOG(WARNING) << "[ContactWrenchConeConstraint] the contact-implicit formulation drops " << kConfigField << ".min_normal_force ("
                 << config.minNormalForce << " N) and .gripper_force (" << config.gripperForce
                 << " N): an always-active cone cannot demand either of a foot in flight. "
                 << "The homogeneous friction, center-of-pressure and torsional limits are unchanged.";
  }
  config.minNormalForce = 0.0;
  config.gripperForce = 0.0;
  return config;
}

ContactWrenchConeConstraint::ContactWrenchConeConstraint(const ContactWrenchConeConstraint& other) = default;

bool ContactWrenchConeConstraint::isActive(scalar_t time) const {
  if (!isActive_) {
    return false;
  }
  // Under the contact-implicit formulation the mode schedule no longer decides which foot carries load, so it cannot
  // be allowed to decide which foot's wrench is bounded either; see contactConstraintsAreScheduleGated().
  if (!scheduleGated_) {
    return true;
  }
  return referenceManagerPtr_->getContactFlags(time)[contactPointIndex_];
}

vector3_t contactPatchReferencePoint(const ContactWrenchConeConstraint::Config& config, const ContactRectangle& contactRectangle) {
  if (!config.patchOffset.isZero(kPatchOffsetZeroThreshold)) {
    return config.patchOffset;
  }
  const PolygonBounds& bounds = contactRectangle.getBounds();
  return vector3_t(kHalf * (bounds.x_min + bounds.x_max), kHalf * (bounds.y_min + bounds.y_max), 0.0);
}

ContactWrenchConeRows buildLocalWrenchConeRows(const ContactWrenchConeConstraint::Config& config,
                                               const ContactRectangle& contactRectangle) {
  // Every factory reports this as a Status naming the task-file key; only a caller that bypassed all of them gets here.
  const absl::Status validConfig = ContactWrenchConeConstraint::validateConfig(config);
  CHECK(validConfig.ok()) << validConfig.message();
  const size_t numBasisVectors = config.numBasisVectors;
  const size_t numRows = numBasisVectors + kExtraConstraintsCount;

  ContactWrenchConeRows rows;
  rows.A_f = matrix_t::Zero(numRows, kWrenchForceDim);
  rows.A_tau = matrix_t::Zero(numRows, kWrenchMomentDim);
  rows.b = vector_t::Zero(numRows);

  size_t constraintIdx = 0;
  const scalar_t effectiveGripperNormal = config.frictionCoefficient * config.gripperForce;

  // 1. Friction cone approximation with numBasisVectors basis vectors:
  // mu * (Fz + F_grip) - (cos(theta_k) * Fx + sin(theta_k) * Fy) >= 0
  const scalar_t angleStep = 2.0 * M_PI / static_cast<scalar_t>(numBasisVectors);
  for (size_t k = 0; k < numBasisVectors; ++k) {
    const scalar_t theta_k = k * angleStep;
    rows.A_f(constraintIdx, kForceXIdx) = -std::cos(theta_k);
    rows.A_f(constraintIdx, kForceYIdx) = -std::sin(theta_k);
    rows.A_f(constraintIdx, kForceZIdx) = config.frictionCoefficient;
    rows.b(constraintIdx) = effectiveGripperNormal;
    ++constraintIdx;
  }

  // 2. Normal force limit: Fz - minNormalForce >= 0
  rows.A_f(constraintIdx, kForceZIdx) = 1.0;
  rows.b(constraintIdx) = -config.minNormalForce;
  ++constraintIdx;

  // 3. Center of Pressure (CoP) / Moment constraints (Bounds relative to local foot contact frame)
  const PolygonBounds& bounds = contactRectangle.getBounds();
  // tau_x - y_min * Fz >= 0
  rows.A_f(constraintIdx, kForceZIdx) = -bounds.y_min;
  rows.A_tau(constraintIdx, kMomentXIdx) = 1.0;
  ++constraintIdx;

  // -tau_x + y_max * Fz >= 0
  rows.A_f(constraintIdx, kForceZIdx) = bounds.y_max;
  rows.A_tau(constraintIdx, kMomentXIdx) = -1.0;
  ++constraintIdx;

  // -tau_y - x_min * Fz >= 0
  rows.A_f(constraintIdx, kForceZIdx) = -bounds.x_min;
  rows.A_tau(constraintIdx, kMomentYIdx) = -1.0;
  ++constraintIdx;

  // tau_y + x_max * Fz >= 0
  rows.A_f(constraintIdx, kForceZIdx) = bounds.x_max;
  rows.A_tau(constraintIdx, kMomentYIdx) = 1.0;
  ++constraintIdx;

  // 4. Torsional yaw friction moment about the contact patch center
  const vector3_t offset = contactPatchReferencePoint(config, contactRectangle);
  const scalar_t effectiveTorsionalGripper = config.torsionalFrictionCoefficient * config.gripperForce;

  // mu_torsion * (Fz + F_grip) + tau_patch_z >= 0
  // tau_patch_z = tau_z - (offset.x * Fy - offset.y * Fx) = offset.y * Fx - offset.x * Fy + tau_z
  rows.A_f(constraintIdx, kForceXIdx) = offset.y();
  rows.A_f(constraintIdx, kForceYIdx) = -offset.x();
  rows.A_f(constraintIdx, kForceZIdx) = config.torsionalFrictionCoefficient;
  rows.A_tau(constraintIdx, kMomentZIdx) = 1.0;
  rows.b(constraintIdx) = effectiveTorsionalGripper;
  ++constraintIdx;

  // mu_torsion * (Fz + F_grip) - tau_patch_z >= 0
  // -tau_patch_z = -offset.y * Fx + offset.x * Fy - tau_z
  rows.A_f(constraintIdx, kForceXIdx) = -offset.y();
  rows.A_f(constraintIdx, kForceYIdx) = offset.x();
  rows.A_f(constraintIdx, kForceZIdx) = config.torsionalFrictionCoefficient;
  rows.A_tau(constraintIdx, kMomentZIdx) = -1.0;
  rows.b(constraintIdx) = effectiveTorsionalGripper;
  ++constraintIdx;

  ABSL_CHECK_EQ(constraintIdx, numRows) << "buildLocalWrenchConeRows(): filled another number of rows than it sized";
  return rows;
}

void ContactWrenchConeConstraint::initializeLocalConstraintMatrix() {
  const ContactWrenchConeRows rows = buildLocalWrenchConeRows(config_, contactRectangle_);
  numConstraints_ = rows.numRows();
  A_f_local_ = rows.A_f;
  A_tau_local_ = rows.A_tau;
  b_local_ = rows.b;
}

vector_t ContactWrenchConeConstraint::getValue(scalar_t /*time*/,
                                               const vector_t& state,
                                               const vector_t& input,
                                               const PreComputation& /*preComp*/) const {
  const pinocchio::Model& model = pinocchioInterfacePtr_->getModel();
  pinocchio::Data data = pinocchioInterfacePtr_->getData();
  updateFramePlacements(mpcRobotModelPtr_->getGeneralizedCoordinates(state), model, data);
  const pinocchio::FrameIndex frameID = getContactFrameIndex(*pinocchioInterfacePtr_, *mpcRobotModelPtr_, contactPointIndex_);

  const matrix3_t w_R_l = getRotationMatrixLocalToWorld(data, frameID);
  const matrix3_t l_R_w = w_R_l.transpose();

  const vector3_t forceInWorld = mpcRobotModelPtr_->getContactForce(input, contactPointIndex_);
  const vector3_t momentInWorld = mpcRobotModelPtr_->getContactMoment(input, contactPointIndex_);

  const vector3_t forceInLocal = l_R_w * forceInWorld;
  const vector3_t momentInLocal = l_R_w * momentInWorld;

  return A_f_local_ * forceInLocal + A_tau_local_ * momentInLocal + b_local_;
}

VectorFunctionLinearApproximation ContactWrenchConeConstraint::getLinearApproximation(scalar_t /*time*/,
                                                                                      const vector_t& state,
                                                                                      const vector_t& input,
                                                                                      const PreComputation& /*preComp*/) const {
  const pinocchio::Model& model = pinocchioInterfacePtr_->getModel();
  pinocchio::Data data = pinocchioInterfacePtr_->getData();
  const vector_t q = mpcRobotModelPtr_->getGeneralizedCoordinates(state);
  updateFramePlacements(q, model, data);
  const pinocchio::FrameIndex frameID = getContactFrameIndex(*pinocchioInterfacePtr_, *mpcRobotModelPtr_, contactPointIndex_);

  const matrix3_t w_R_l = getRotationMatrixLocalToWorld(data, frameID);
  const matrix3_t l_R_w = w_R_l.transpose();

  const matrix_t A_f_world = A_f_local_ * l_R_w;
  const matrix_t A_tau_world = A_tau_local_ * l_R_w;

  const vector3_t forceInWorld = mpcRobotModelPtr_->getContactForce(input, contactPointIndex_);
  const vector3_t momentInWorld = mpcRobotModelPtr_->getContactMoment(input, contactPointIndex_);
  const vector3_t forceInLocal = l_R_w * forceInWorld;
  const vector3_t momentInLocal = l_R_w * momentInWorld;

  // THE STATE DERIVATIVE, through the rotation. With w_R_l' = w_R_l [omega_local]x, the local image of a fixed world
  // vector moves as d(l_R_w v)/dt = -[omega_local]x (l_R_w v) = [v_local]x omega_local, and omega_local is the angular
  // half of the contact frame's LOCAL Jacobian times the generalized velocity. Every generalized coordinate of this
  // repository's models is a configuration variable whose rate is its velocity (a translation, ZYX Euler angles and
  // revolute joints), so the same Jacobian is d/dq.
  matrix_t frameJacobian = matrix_t::Zero(6, model.nv);
  pinocchio::computeFrameJacobian(model, data, q, frameID, pinocchio::LOCAL, frameJacobian);
  const matrix_t angularJacobian = frameJacobian.bottomRows<3>();
  const matrix_t dfdq =
      A_f_local_ * skewSymmetric(forceInLocal) * angularJacobian + A_tau_local_ * skewSymmetric(momentInLocal) * angularJacobian;

  VectorFunctionLinearApproximation linearApproximation;
  linearApproximation.f = A_f_local_ * forceInLocal + A_tau_local_ * momentInLocal + b_local_;
  linearApproximation.dfdx = dfdq * generalizedCoordinatesStateJacobian_;
  linearApproximation.dfdu = matrix_t::Zero(numConstraints_, mpcRobotModelPtr_->getInputDim());

  const size_t forceStartIdx = mpcRobotModelPtr_->getContactForceStartIndices(contactPointIndex_);
  const size_t momentStartIdx = mpcRobotModelPtr_->getContactMomentStartIndices(contactPointIndex_);

  linearApproximation.dfdu.block(0, forceStartIdx, numConstraints_, kWrenchForceDim) = A_f_world;
  linearApproximation.dfdu.block(0, momentStartIdx, numConstraints_, kWrenchMomentDim) = A_tau_world;

  return linearApproximation;
}

VectorFunctionQuadraticApproximation ContactWrenchConeConstraint::getQuadraticApproximation(scalar_t time,
                                                                                            const vector_t& state,
                                                                                            const vector_t& input,
                                                                                            const PreComputation& preComp) const {
  const VectorFunctionLinearApproximation linearApprox = getLinearApproximation(time, state, input, preComp);
  VectorFunctionQuadraticApproximation quadraticApproximation;
  quadraticApproximation.f = linearApprox.f;
  quadraticApproximation.dfdx = linearApprox.dfdx;
  quadraticApproximation.dfdu = linearApprox.dfdu;
  quadraticApproximation.dfdxx.resize(numConstraints_, matrix_t::Zero(mpcRobotModelPtr_->getStateDim(), mpcRobotModelPtr_->getStateDim()));
  quadraticApproximation.dfduu.resize(numConstraints_, matrix_t::Zero(mpcRobotModelPtr_->getInputDim(), mpcRobotModelPtr_->getInputDim()));
  quadraticApproximation.dfdux.resize(numConstraints_, matrix_t::Zero(mpcRobotModelPtr_->getInputDim(), mpcRobotModelPtr_->getStateDim()));
  return quadraticApproximation;
}

}  // namespace ocs2::humanoid
