/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

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

#include "humanoid_common_mpc/HumanoidCostConstraintFactory.h"

#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/constraint/StateInputConstraint.h"
#include "ocs2_core/cost/QuadraticStateCost.h"
#include "ocs2_core/penalties/penalties/PieceWisePolynomialBarrierPenalty.h"
#include "ocs2_core/penalties/penalties/RelaxedBarrierPenalty.h"
#include "ocs2_core/penalties/penalties/SquaredHingePenalty.h"
#include "ocs2_core/soft_constraint/StateInputSoftConstraint.h"
#include "ocs2_core/soft_constraint/StateSoftConstraint.h"

#include "humanoid_common_mpc/common/BasisInputsCostTransform.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/costs/CollisionConstraintFromConfig.h"
#include "humanoid_common_mpc/config/costs/ContactsFromConfig.h"
#include "humanoid_common_mpc/config/costs/JointLimitsFromConfig.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/config/weights/StateInputWeightsFromConfig.h"
#include "humanoid_common_mpc/constraint/ContactMomentXYConstraintCppAd.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/constraint/FootCollisionConstraint.h"
#include "humanoid_common_mpc/constraint/FrictionForceConeConstraint.h"
#include "humanoid_common_mpc/constraint/JointLimitsSoftConstraint.h"
#include "humanoid_common_mpc/constraint/ZeroWrenchConstraint.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/cost/BasePoseShapedQuadraticStateCost.h"
#include "humanoid_common_mpc/cost/ComAndAcomTrackingCost.h"
#include "humanoid_common_mpc/cost/EndEffectorKinematicCostHelpers.h"
#include "humanoid_common_mpc/cost/ExternalTorqueQuadraticCostAD.h"
#include "humanoid_common_mpc/cost/InputQuadraticCost.h"
#include "humanoid_common_mpc/cost/StateInputQuadraticCost.h"
#include "humanoid_common_mpc/cost/StateQuadraticCost.h"
#include "humanoid_common_mpc/pinocchio_model/pinocchioUtils.h"
#include "humanoid_mpc_config/task_file.nproto.h"

namespace ocs2::humanoid {

namespace {

/**
 * The error for com_weights or acom_weights, which the cost com_and_acom_tracking_cost reads, when the block does not
 * convert (`conversion`): listing the cost without writing its weights - the state the G1 and SA01 task files are in -
 * is a configuration error fixed in the task file.
 */
absl::Status comAndAcomTrackingWeightsError(const absl::Status& conversion) {
  return absl::InvalidArgumentError(absl::StrCat(
      "[HumanoidCostConstraintFactory] costs lists com_and_acom_tracking_cost, but its weights do not convert (", conversion.message(),
      "). Write com_weights and acom_weights in the task file (terminal_cost_scaling too beside terminal_cost), or remove "
      "com_and_acom_tracking_cost from costs."));
}

/**
 * InvalidArgument naming `field` unless `matrix` is `size` square: the size of the model's state or input, which the
 * typed file's weights are laid out on (StateInputLayout) and the term evaluates them on.
 */
absl::Status checkSquareOfSize(absl::string_view field, const matrix_t& matrix, size_t size) {
  if (static_cast<size_t>(matrix.rows()) == size && static_cast<size_t>(matrix.cols()) == size) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(absl::StrCat("[HumanoidCostConstraintFactory] ", field, " converts to a ", matrix.rows(), "x",
                                                 matrix.cols(), " matrix, but the model's dimension is ", size,
                                                 ": the layout of the factory is not the model's."));
}

}  // namespace

HumanoidCostConstraintFactory::HumanoidCostConstraintFactory(const mpc_config::TaskFile* absl_nonnull taskFile,
                                                             StateInputLayout::Mpc mpc,
                                                             const SwitchedModelReferenceManager& referenceManager,
                                                             const PinocchioInterface& pinocchioInterface,
                                                             const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                                             const MpcRobotModelBase<ad_scalar_t>& mpcRobotModelAD,
                                                             const ModelSettings& modelSettings,
                                                             bool verbose,
                                                             bool scheduleGatedContactConstraints)
    : taskFilePtr_(taskFile),
      layout_(stateInputLayout(modelSettings, mpc)),
      referenceManagerPtr_(&referenceManager),
      pinocchioInterfacePtr_(&pinocchioInterface),
      mpcRobotModelPtr_(&mpcRobotModel),
      mpcRobotModelADPtr_(&mpcRobotModelAD),
      modelSettings_(modelSettings),
      verbose_(verbose),
      scheduleGatedContactConstraints_(scheduleGatedContactConstraints) {
  ABSL_CHECK(taskFilePtr_ != nullptr) << "HumanoidCostConstraintFactory: no task file";
}

/**
 * The penalty a contact cone is wrapped in.
 *
 * A schedule-gated cone is only ever evaluated on a foot the schedule has declared loaded, where the slack is
 * comfortably positive and a relaxed log barrier is the right object. An always-active cone is also evaluated on a
 * foot in flight, which sits exactly ON the boundary of the homogeneous cone. A relaxed log barrier has value
 * -mu*ln(delta) - mu/2 and derivative -2*mu/delta there - with the shipped center-of-pressure settings, mu = 0.6 and
 * delta = 0.03, that is a derivative of -40 per row - so it would pay the solver to leave the origin, i.e. to invent a
 * normal force on a foot in the air. That is precisely the failure that dropping `minNormalForce` and the friction
 * cone's parabolic margin exists to remove, and a log barrier would put it straight back.
 *
 * The squared hinge is zero in value AND zero in derivative at zero slack, and quadratic below it, which is what a
 * unilateral condition whose solution lies on the boundary needs. `delta` is 0 for exactly that reason: a positive
 * delta shifts the hinge's zero into the interior and reintroduces the bias. It is DROPPED, not converted - the
 * barrier's `delta` is a smoothing width in the units of the constraint row, while the hinge's is the offset of its
 * zero, and the two have nothing to do with one another beyond the name.
 *
 * `mu` IS REUSED, AND IT MEANS SOMETHING DIFFERENT IN THE TWO PENALTIES, so the substitution is not scale-preserving
 * and that has to be stated rather than discovered. In the barrier `mu` is a log coefficient whose gradient near the
 * boundary goes as mu/delta; in the hinge it is the quadratic stiffness of `0.5 * mu * h^2`. With the shipped cone
 * settings (mu = 0.2, delta = 5, rows in newtons) the hinge is the STRONGER of the two everywhere a violation is
 * worth caring about - equal at 0.42 N, 2.3x the barrier's restoring gradient at 1 N, 12.5x at 10 N, 22.7x at 100 N,
 * and 25x its curvature throughout. Carrying `mu` across therefore tightens the cone rather than loosening it, which
 * is the safe direction for a bound that is the ONLY thing holding a wrench inside the cone once the schedule gate is
 * gone. Matching the barrier's curvature instead would mean mu/delta^2 = 0.008, an eightyfold softening, and is not
 * what is wanted here.
 */
std::unique_ptr<PenaltyBase> makeContactConePenalty(const RelaxedBarrierPenalty::Config& barrierConfig, bool scheduleGated) {
  const vector_t parameters = contactConePenaltyParameters(barrierConfig, scheduleGated);
  if (scheduleGated) {
    return std::make_unique<RelaxedBarrierPenalty>(RelaxedBarrierPenalty::Config(parameters(0), parameters(1)));
  }
  LOG(INFO) << "[HumanoidCostConstraintFactory] Un-gated cone: squared hinge with stiffness mu = " << parameters(0)
            << " (cost 0.5*mu*h^2 below the cone, zero on and above it). The configured barrier delta of " << barrierConfig.delta
            << " is deliberately not carried over; see makeContactConePenalty().";
  return std::make_unique<SquaredHingePenalty>(SquaredHingePenalty::Config(parameters(0), parameters(1)));
}

vector_t contactConePenaltyParameters(const RelaxedBarrierPenalty::Config& barrierConfig, bool scheduleGated) {
  // An un-gated cone's hinge keeps its zero on the cone, which a foot at zero wrench lies on: its delta is the OFFSET of
  // that zero, not the barrier's smoothing width, and is 0 whatever the file's barrier delta says.
  return (vector_t(2) << barrierConfig.mu, scheduleGated ? barrierConfig.delta : 0.0).finished();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void HumanoidCostConstraintFactory::setBasisInputsCostTransform(BasisInputsCostTransformConfig config) {
  basisCostTransform_ = std::move(config);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::unique_ptr<StateInputConstraint> HumanoidCostConstraintFactory::getZeroWrenchConstraint(size_t contactPointIndex) const {
  return std::make_unique<ZeroWrenchConstraint>(*referenceManagerPtr_, contactPointIndex, *mpcRobotModelPtr_);
}

/******************************************************************************************************/
/***************************** The terms of the typed task file (make*()) *****************************/
/******************************************************************************************************/

absl::StatusOr<matrix_t> HumanoidCostConstraintFactory::stateWeights(const mpc_config::TaskFile& taskFile) const {
  ASSIGN_OR_RETURN(matrix_t Q, stateWeightsFromConfig(taskFile.state_weights, layout_, "state_weights"));
  RETURN_IF_ERROR(checkSquareOfSize("state_weights", Q, mpcRobotModelADPtr_->getStateDim()));
  // With com_and_acom_tracking_cost listed the base pose is regulated in CoM and ACoM coordinates, whichever of the two
  // quadratic state costs carries the rest of Q.
  if (comAndAcomTrackingCostListed_) {
    ComAndAcomTrackingCost::zeroBasePoseWeights(Q);
    if (verbose_) {
      LOG(INFO) << "[HumanoidCostConstraintFactory] com_and_acom_tracking_cost is listed. Zeroing out base pose weights in state_weights.";
    }
  }
  return Q;
}

absl::StatusOr<matrix_t> HumanoidCostConstraintFactory::inputWeights(const mpc_config::TaskFile& taskFile) const {
  ASSIGN_OR_RETURN(matrix_t R, inputWeightsFromConfig(taskFile.input_weights, layout_, "input_weights"));
  if (!basisCostTransform_.has_value()) {
    RETURN_IF_ERROR(checkSquareOfSize("input_weights", R, mpcRobotModelADPtr_->getInputDim()));
    return R;
  }
  // R is written in wrench dimensions; the basis-vector input needs R_basis = M^T R_wrench M + reg * blkdiag(S, 0).
  RETURN_IF_ERROR(checkSquareOfSize("input_weights", R, basisCostTransform_->wrenchInputDim));
  matrix_t R_basis = transformWrenchInputCostToBasisSpace(R, *basisCostTransform_);
  if (verbose_) {
    LOG(INFO) << "\n #### input_weights in wrench space (" << R.rows() << "x" << R.cols() << ") transformed to basis-vector space ("
              << R_basis.rows() << "x" << R_basis.cols() << ", contacts.basis_regularization = " << basisCostTransform_->regularization
              << ", contacts.basis_scaling_regularization = " << basisCostTransform_->lambdaRegularization << ")";
  }
  return R_basis;
}

absl::StatusOr<std::unique_ptr<StateInputCost>> HumanoidCostConstraintFactory::makeStateInputQuadraticCost() const {
  const mpc_config::TaskFile& taskFile = *taskFilePtr_;
  ASSIGN_OR_RETURN(matrix_t Q, stateWeights(taskFile));
  ASSIGN_OR_RETURN(matrix_t R, inputWeights(taskFile));
  if (verbose_) {
    LOG(INFO) << "\n #### Base Tracking Cost Coefficients: \n"
              << " #### =============================================================================\n"
              << "Q:\n"
              << Q << "\n"
              << "R:\n"
              << R << "\n"
              << " #### =============================================================================";
  }
  return std::make_unique<StateInputQuadraticCost>(std::move(Q), std::move(R), *referenceManagerPtr_, *mpcRobotModelPtr_);
}

absl::StatusOr<std::unique_ptr<StateInputCost>> HumanoidCostConstraintFactory::makeStateQuadraticCost() const {
  const mpc_config::TaskFile& taskFile = *taskFilePtr_;
  ASSIGN_OR_RETURN(matrix_t Q, stateWeights(taskFile));
  if (verbose_) {
    LOG(INFO) << "\n #### Base Tracking State Cost Coefficients: \n"
              << " #### =============================================================================\n"
              << "Q:\n"
              << Q << "\n"
              << " #### =============================================================================";
  }
  return std::make_unique<StateQuadraticCost>(std::move(Q), mpcRobotModelADPtr_->getInputDim(), *referenceManagerPtr_);
}

absl::StatusOr<std::unique_ptr<StateInputCost>> HumanoidCostConstraintFactory::makeInputQuadraticCost() const {
  const mpc_config::TaskFile& taskFile = *taskFilePtr_;
  ASSIGN_OR_RETURN(matrix_t R, inputWeights(taskFile));
  if (verbose_) {
    LOG(INFO) << "\n #### Base Tracking Input Cost Coefficients: \n"
              << " #### =============================================================================\n"
              << "R:\n"
              << R << "\n"
              << " #### =============================================================================";
  }
  return std::make_unique<InputQuadraticCost>(std::move(R), mpcRobotModelADPtr_->getStateDim(), *referenceManagerPtr_, *mpcRobotModelPtr_);
}

absl::StatusOr<std::unique_ptr<StateCost>> HumanoidCostConstraintFactory::buildComAndAcomTrackingCost(const mpc_config::TaskFile& taskFile,
                                                                                                      const CentroidalModelInfo& info,
                                                                                                      scalar_t scaling) const {
  absl::StatusOr<matrix_t> comWeights = comWeightsFromConfig(taskFile.com_weights, "com_weights");
  if (!comWeights.ok()) {
    return comAndAcomTrackingWeightsError(comWeights.status());
  }
  absl::StatusOr<matrix_t> acomWeights = acomWeightsFromConfig(taskFile.acom_weights, "acom_weights");
  if (!acomWeights.ok()) {
    return comAndAcomTrackingWeightsError(acomWeights.status());
  }
  matrix_t Q_com = *std::move(comWeights) * scaling;
  matrix_t Q_acom = *std::move(acomWeights) * scaling;
  if (verbose_) {
    LOG(INFO) << "\n #### CoM + ACoM Tracking Cost Coefficients (scaled by " << scaling << "): \n"
              << " #### =============================================================================\n"
              << "com_weights:\n"
              << Q_com << "\n"
              << "acom_weights (rows are ZYX Euler: yaw, pitch, roll):\n"
              << Q_acom << "\n"
              << " #### =============================================================================";
  }
  ASSIGN_OR_RETURN(
      std::unique_ptr<ComAndAcomTrackingCost> cost,
      ComAndAcomTrackingCost::Create(std::move(Q_com), std::move(Q_acom), *pinocchioInterfacePtr_, info, modelSettings_.robotName));
  return std::unique_ptr<StateCost>(std::move(cost));
}

absl::StatusOr<std::unique_ptr<StateCost>> HumanoidCostConstraintFactory::makeComAndAcomTrackingCost(
    const CentroidalModelInfo& info) const {
  const mpc_config::TaskFile& taskFile = *taskFilePtr_;
  return buildComAndAcomTrackingCost(taskFile, info, /*scaling=*/1.0);
}

absl::StatusOr<std::unique_ptr<StateCost>> HumanoidCostConstraintFactory::makeTerminalComAndAcomTrackingCost(
    const CentroidalModelInfo& info) const {
  const mpc_config::TaskFile& taskFile = *taskFilePtr_;
  // The factor makeTerminalCost() applies to final_state_weights, so that the terminal node weighs the CoM and the ACoM
  // against the rest of the terminal state exactly as the running nodes weigh them against the rest of state_weights.
  ASSIGN_OR_RETURN(const scalar_t terminalCostScaling, terminalCostScalingFromConfig(taskFile));
  return buildComAndAcomTrackingCost(taskFile, info, terminalCostScaling);
}

absl::StatusOr<std::unique_ptr<StateCost>> HumanoidCostConstraintFactory::makeTerminalCost() const {
  const mpc_config::TaskFile& taskFile = *taskFilePtr_;
  ASSIGN_OR_RETURN(const scalar_t terminalCostScaling, terminalCostScalingFromConfig(taskFile));
  LOG(INFO) << "terminal_cost_scaling: " << terminalCostScaling;
  ASSIGN_OR_RETURN(matrix_t Qf, stateWeightsFromConfig(taskFile.final_state_weights, layout_, "final_state_weights"));
  RETURN_IF_ERROR(checkSquareOfSize("final_state_weights", Qf, mpcRobotModelPtr_->getStateDim()));
  if (comAndAcomTrackingCostListed_) {
    // Zeroed for the reason the running cost's block is (stateWeights()); makeTerminalComAndAcomTrackingCost()
    // regulates the terminal base pose in its place.
    ComAndAcomTrackingCost::zeroBasePoseWeights(Qf);
    if (verbose_) {
      LOG(INFO) << "[HumanoidCostConstraintFactory] com_and_acom_tracking_cost is listed. Zeroing out base pose weights in "
                   "final_state_weights.";
    }
  }
  Qf *= terminalCostScaling;
  if (verbose_) LOG(INFO) << "final_state_weights:\n" << Qf;
  // Shaped by the locomotion heuristics' base-pose offsets like the running costs, so the horizon's last node does not
  // pull back to the unshaped pose; identical to QuadraticStateCost when none is listed.
  return std::make_unique<BasePoseShapedQuadraticStateCost>(Qf, *referenceManagerPtr_);
}

absl::StatusOr<std::unique_ptr<StateCost>> HumanoidCostConstraintFactory::makeFootCollisionConstraint() const {
  const mpc_config::TaskFile& taskFile = *taskFilePtr_;
  ASSIGN_OR_RETURN(const FootCollisionConstraint::Config collisionConfig,
                   footCollisionConstraintConfigFromConfig(taskFile.collision_constraint));
  ASSIGN_OR_RETURN(const PieceWisePolynomialBarrierPenalty::Config barrierPenaltyConfig,
                   footCollisionBarrierFromConfig(taskFile.collision_constraint));
  std::unique_ptr<PieceWisePolynomialBarrierPenalty> penalty = std::make_unique<PieceWisePolynomialBarrierPenalty>(barrierPenaltyConfig);
  std::unique_ptr<FootCollisionConstraint> footCollisionConstraintPtr = std::make_unique<FootCollisionConstraint>(
      *referenceManagerPtr_, *pinocchioInterfacePtr_, *mpcRobotModelADPtr_, collisionConfig, "FootCollisionConstraint", modelSettings_);
  return std::make_unique<StateSoftConstraint>(std::move(footCollisionConstraintPtr), std::move(penalty));
}

absl::StatusOr<std::unique_ptr<StateCost>> HumanoidCostConstraintFactory::makeJointLimitsConstraint() const {
  const mpc_config::TaskFile& taskFile = *taskFilePtr_;
  ASSIGN_OR_RETURN(const PieceWisePolynomialBarrierPenalty::Config barrierPenaltyConfig,
                   jointLimitsBarrierFromConfig(taskFile.joint_limits));
  if (verbose_) {
    LOG(INFO) << "\n #### Joint Limit Barrier Penalty Config: joint_limits.mu = " << barrierPenaltyConfig.mu
              << ", joint_limits.delta = " << barrierPenaltyConfig.delta;
  }
  LOG(INFO) << "Initialized joint limits constraint with zero crossing cost " << barrierPenaltyConfig.getZeroCrossingValue() << ".";
  std::pair<vector_t, vector_t> jointLimits = readPinocchioJointLimits(*pinocchioInterfacePtr_, mpcRobotModelPtr_->modelSettings);
  return std::make_unique<JointLimitsSoftConstraint>(jointLimits, barrierPenaltyConfig, *mpcRobotModelPtr_);
}

absl::StatusOr<std::unique_ptr<StateInputCost>> HumanoidCostConstraintFactory::makeContactMomentXYConstraint(
    size_t contactPointIndex, const std::string& name) const {
  const mpc_config::TaskFile& taskFile = *taskFilePtr_;
  ASSIGN_OR_RETURN(const RelaxedBarrierPenalty::Config barrierPenaltyConfig, contactMomentXyBarrierFromConfig(taskFile.contacts));
  ASSIGN_OR_RETURN(const ContactRectangle footprint,
                   contactRectangleFromConfig(taskFile.contacts, mpcRobotModelPtr_->modelSettings, static_cast<int>(contactPointIndex)));
  std::unique_ptr<ContactMomentXYConstraintCppAd> contactMomentXYConstraintPtr =
      std::make_unique<ContactMomentXYConstraintCppAd>(*referenceManagerPtr_, footprint, contactPointIndex, *pinocchioInterfacePtr_,
                                                       *mpcRobotModelADPtr_, name, modelSettings_, scheduleGatedContactConstraints_);
  std::unique_ptr<PenaltyBase> penalty = makeContactConePenalty(barrierPenaltyConfig, scheduleGatedContactConstraints_);
  return std::make_unique<StateInputSoftConstraint>(std::move(contactMomentXYConstraintPtr), std::move(penalty));
}

absl::StatusOr<std::unique_ptr<StateInputCost>> HumanoidCostConstraintFactory::makeContactWrenchConeConstraint(
    size_t contactPointIndex) const {
  const mpc_config::TaskFile& taskFile = *taskFilePtr_;
  // The barrier's weight and relaxation, finite and positive (a negative mu would reward leaving the cone), and the
  // ground: friction, torsion, the affine offsets and the pyramid's facets.
  ASSIGN_OR_RETURN(const RelaxedBarrierPenalty::Config barrierPenaltyConfig, contactWrenchConeBarrierFromConfig(taskFile.contacts));
  ASSIGN_OR_RETURN(ContactWrenchConeConstraint::Config config, contactWrenchConeConfigFromConfig(taskFile.contacts));
  ASSIGN_OR_RETURN(const ContactRectangle footprint,
                   contactRectangleFromConfig(taskFile.contacts, mpcRobotModelPtr_->modelSettings, static_cast<int>(contactPointIndex)));
  ASSIGN_OR_RETURN(std::unique_ptr<ContactWrenchConeConstraint> contactWrenchConeConstraintPtr,
                   ContactWrenchConeConstraint::Create(*referenceManagerPtr_, footprint, contactPointIndex, *pinocchioInterfacePtr_,
                                                       *mpcRobotModelPtr_, std::move(config), scheduleGatedContactConstraints_));
  std::unique_ptr<PenaltyBase> penalty = makeContactConePenalty(barrierPenaltyConfig, scheduleGatedContactConstraints_);
  return std::make_unique<StateInputSoftConstraint>(std::move(contactWrenchConeConstraintPtr), std::move(penalty));
}

absl::StatusOr<std::unique_ptr<StateInputCost>> HumanoidCostConstraintFactory::makeFrictionForceConeConstraint(
    size_t contactPointIndex) const {
  const mpc_config::TaskFile& taskFile = *taskFilePtr_;
  ASSIGN_OR_RETURN(const FrictionForceConeConstraint::Config frictionConeConfig, frictionForceConeConfigFromConfig(taskFile.contacts));
  ASSIGN_OR_RETURN(const RelaxedBarrierPenalty::Config barrierPenaltyConfig, frictionForceConeBarrierFromConfig(taskFile.contacts));
  if (verbose_) {
    LOG(INFO) << "\n #### Friction Cone Settings: contacts.friction_force_cone_soft_constraint.friction_coefficient = "
              << frictionConeConfig.frictionCoefficient << ", mu = " << barrierPenaltyConfig.mu
              << ", delta = " << barrierPenaltyConfig.delta;
  }
  std::unique_ptr<FrictionForceConeConstraint> frictionForceConeConstraintPtr = std::make_unique<FrictionForceConeConstraint>(
      *referenceManagerPtr_, frictionConeConfig, contactPointIndex, *mpcRobotModelPtr_, scheduleGatedContactConstraints_);
  std::unique_ptr<PenaltyBase> penalty = makeContactConePenalty(barrierPenaltyConfig, scheduleGatedContactConstraints_);
  return std::make_unique<StateInputSoftConstraint>(std::move(frictionForceConeConstraintPtr), std::move(penalty));
}

absl::StatusOr<std::unique_ptr<StateInputCost>> HumanoidCostConstraintFactory::makeExternalTorqueQuadraticCost(
    size_t contactPointIndex) const {
  const mpc_config::TaskFile& taskFile = *taskFilePtr_;
  if (contactPointIndex > 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("[HumanoidCostConstraintFactory] the external torque cost is a leg's, of the contacts 0 "
                     "(left_leg_torque_cost) and 1 (right_leg_torque_cost), not of contact ",
                     contactPointIndex, "."));
  }
  const bool left = contactPointIndex == 0;
  ASSIGN_OR_RETURN(const ExternalTorqueQuadraticCostAD::Config config,
                   legTorqueCostFromConfig(left ? taskFile.left_leg_torque_cost : taskFile.right_leg_torque_cost, layout_,
                                           left ? "left_leg_torque_cost" : "right_leg_torque_cost"));
  ASSIGN_OR_RETURN(std::unique_ptr<ExternalTorqueQuadraticCostAD> cost,
                   ExternalTorqueQuadraticCostAD::Create(contactPointIndex, config, *referenceManagerPtr_, *pinocchioInterfacePtr_,
                                                         *mpcRobotModelADPtr_, modelSettings_));
  return cost;
}

}  // namespace ocs2::humanoid
