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

#pragma once

#include <memory>
#include <optional>
#include <string>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"
#include "ocs2_core/constraint/StateInputConstraint.h"
#include "ocs2_core/cost/StateCost.h"
#include "ocs2_core/cost/StateInputCost.h"
#include "ocs2_core/penalties/penalties/PenaltyBase.h"
#include "ocs2_core/penalties/penalties/RelaxedBarrierPenalty.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "ocs2_robotic_tools/end_effector/EndEffectorKinematics.h"

#include "humanoid_common_mpc/common/BasisInputsCostTransform.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_mpc_config/task_file.nproto.h"

namespace ocs2::humanoid {

/**
 * The (mu, delta) of the penalty a contact cone - contact_wrench_cone, friction_force_cone, contact_moment_xy - is
 * wrapped in: the task file's barrier parameters for a schedule-gated cone, and (mu, 0) for an un-gated one, whose
 * squared hinge must keep its zero ON the cone (see makeContactConePenalty()). The factory builds the cones' penalties
 * from it and MpcParameterUpdaterModule rewrites them with it on every hot reload, so the two cannot disagree about the
 * hinge's delta: writing the barrier's delta into the hinge would move its zero into the cone and reinstate the force
 * floor that un-gating removes.
 */
vector_t contactConePenaltyParameters(const RelaxedBarrierPenalty::Config& barrierConfig, bool scheduleGated);

/**
 * The penalty a contact cone is wrapped in: a RelaxedBarrierPenalty for a schedule-gated cone, a SquaredHingePenalty
 * with delta 0 for an un-gated one, parameterized by contactConePenaltyParameters(). The definition explains why.
 */
std::unique_ptr<PenaltyBase> makeContactConePenalty(const RelaxedBarrierPenalty::Config& barrierConfig, bool scheduleGated);

/**
 * Builds the costs and constraints of the humanoid MPC formulations from the task file.
 *
 * An MPC interface constructs one with its typed task file and model objects, which must outlive the factory, and calls
 * the make*() functions while it assembles the optimal control problem; each converts the blocks its term reads
 * (humanoid_common_mpc/config/) and returns the conversion's error for one that does not convert. The terms it returns
 * own copies of what they need. Not thread-safe: it is used on the thread that builds the interface.
 */
class HumanoidCostConstraintFactory {
 public:
  /**
   * The factory of the typed task file `taskFile`, for the MPC `mpc`, whose state and input the file's weights are
   * addressed on (StateInputLayout, built from `modelSettings`). The factory keeps `taskFile` and references to the
   * model objects: all of them must outlive it (an interface builds its problem with a factory on its stack).
   */
  HumanoidCostConstraintFactory(const mpc_config::TaskFile* absl_nonnull taskFile,
                                StateInputLayout::Mpc mpc,
                                const SwitchedModelReferenceManager& referenceManager,
                                const PinocchioInterface& pinocchioInterface,
                                const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                const MpcRobotModelBase<ad_scalar_t>& mpcRobotModelAD,
                                const ModelSettings& modelSettings,
                                bool verbose = false,
                                bool scheduleGatedContactConstraints = true);

  ~HumanoidCostConstraintFactory() = default;
  HumanoidCostConstraintFactory(const HumanoidCostConstraintFactory& other) = delete;
  HumanoidCostConstraintFactory& operator=(const HumanoidCostConstraintFactory&) = delete;

  /**
   * Declares that the problem this factory assembles lists `com_and_acom_tracking_cost`, i.e. that ComAndAcomTrackingCost
   * regulates the base pose. Every quadratic state cost built afterwards - makeStateQuadraticCost(),
   * makeStateInputQuadraticCost() and makeTerminalCost() - then has the base-pose block of its state_weights or
   * final_state_weights zeroed (ComAndAcomTrackingCost::zeroBasePoseWeights). Off unless called; the whole-body MPC
   * never calls it.
   */
  void setComAndAcomTrackingCostListed(bool listed) { comAndAcomTrackingCostListed_ = listed; }

  /**
   * Declares the basis-vector contact input parameterization. The input costs built afterwards convert input_weights in
   * wrench dimensions (wrenchInputDim x wrenchInputDim) and transform them into basis space,
   * R_basis = M^T R_wrench M + reg * blkdiag(S, 0), with the regularization S named by `config.regularization`
   * (contacts.basis_regularization) - through transformWrenchInputCostToBasisSpace(R_wrench, config), the same call the
   * online parameter updater makes, so that a hot reload reproduces the start-up R exactly.
   *
   * @param config A config validateBasisInputsCostTransformConfig() accepts; CentroidalMpcInterface builds it.
   */
  void setBasisInputsCostTransform(BasisInputsCostTransformConfig config);

  /** The hard zero-wrench constraint of a contact in swing (zero_wrench). */
  std::unique_ptr<StateInputConstraint> getZeroWrenchConstraint(size_t contactPointIndex) const;

  // The terms of the task file. Each fails with the InvalidArgument of its conversion (which names the field) for a block
  // that does not convert, or whose size is not the model's.

  /**
   * The state-input quadratic cost: state_weights as Q, whose base-pose block is zeroed under
   * setComAndAcomTrackingCostListed, and input_weights as R (inputWeights()).
   */
  absl::StatusOr<std::unique_ptr<StateInputCost>> makeStateInputQuadraticCost() const;

  /** The state quadratic tracking cost: state_weights as Q, zeroed as in makeStateInputQuadraticCost(). */
  absl::StatusOr<std::unique_ptr<StateInputCost>> makeStateQuadraticCost() const;

  /** The input quadratic cost: input_weights as R (inputWeights()). */
  absl::StatusOr<std::unique_ptr<StateInputCost>> makeInputQuadraticCost() const;

  /**
   * The running CoM and ACoM tracking cost on com_weights and acom_weights.
   *
   * @param info Centroidal model info for the same reduced Pinocchio model this factory was constructed with. Passed in
   *     rather than reconstructed here, because CentroidalModelInfo has no default member initializers and a hand-built
   *     one leaves the contact and nominal-inertia fields indeterminate.
   * @return InvalidArgument naming the block when com_and_acom_tracking_cost is listed but the file's weights do not
   *     convert, or ComAndAcomTrackingCost::Create()'s error.
   */
  absl::StatusOr<std::unique_ptr<StateCost>> makeComAndAcomTrackingCost(const CentroidalModelInfo& info) const;

  /** The terminal instance, weighted by terminal_cost_scaling times com_weights and acom_weights. */
  absl::StatusOr<std::unique_ptr<StateCost>> makeTerminalComAndAcomTrackingCost(const CentroidalModelInfo& info) const;

  /**
   * The quadratic terminal cost, terminal_cost_scaling times final_state_weights, whose base-pose block is zeroed under
   * setComAndAcomTrackingCostListed before it is scaled.
   */
  absl::StatusOr<std::unique_ptr<StateCost>> makeTerminalCost() const;

  /** The foot and knee collision constraint of collision_constraint. */
  absl::StatusOr<std::unique_ptr<StateCost>> makeFootCollisionConstraint() const;

  /** The joint limits of the URDF as a soft constraint, with the barrier of joint_limits. */
  absl::StatusOr<std::unique_ptr<StateCost>> makeJointLimitsConstraint() const;

  /** The center-of-pressure constraint of a contact: its sole and contacts.contact_moment_xy_soft_constraint. */
  absl::StatusOr<std::unique_ptr<StateInputCost>> makeContactMomentXYConstraint(size_t contactPointIndex, const std::string& name) const;

  /**
   * The contact wrench cone of one foot as a soft constraint: the ground and the barrier of
   * contacts.contact_wrench_cone_soft_constraint and the foot's sole. Wrench-space models only.
   */
  absl::StatusOr<std::unique_ptr<StateInputCost>> makeContactWrenchConeConstraint(size_t contactPointIndex) const;

  /** The friction cone of one foot, contacts.friction_force_cone_soft_constraint. */
  absl::StatusOr<std::unique_ptr<StateInputCost>> makeFrictionForceConeConstraint(size_t contactPointIndex) const;

  /** The external torque cost of a leg (left_leg_torque_cost, right_leg_torque_cost for the contacts 0 and 1). */
  absl::StatusOr<std::unique_ptr<StateInputCost>> makeExternalTorqueQuadraticCost(size_t contactPointIndex) const;

 private:
  /** state_weights as Q, its base-pose block zeroed under setComAndAcomTrackingCostListed. */
  absl::StatusOr<matrix_t> stateWeights(const mpc_config::TaskFile& taskFile) const;

  /** input_weights as R, transformed into basis space under setBasisInputsCostTransform. */
  absl::StatusOr<matrix_t> inputWeights(const mpc_config::TaskFile& taskFile) const;

  /** com_weights and acom_weights, each multiplied by `scaling`, and a ComAndAcomTrackingCost on them. */
  absl::StatusOr<std::unique_ptr<StateCost>> buildComAndAcomTrackingCost(const mpc_config::TaskFile& taskFile,
                                                                         const CentroidalModelInfo& info,
                                                                         scalar_t scaling) const;

  const mpc_config::TaskFile* absl_nonnull taskFilePtr_;
  // The coordinates the file's weights are addressed on.
  StateInputLayout layout_;
  const SwitchedModelReferenceManager* absl_nonnull referenceManagerPtr_;
  const PinocchioInterface* absl_nonnull pinocchioInterfacePtr_;
  const MpcRobotModelBase<scalar_t>* absl_nonnull mpcRobotModelPtr_;
  const MpcRobotModelBase<ad_scalar_t>* absl_nonnull mpcRobotModelADPtr_;
  const ModelSettings& modelSettings_;
  const bool verbose_;
  /**
   * Whether the contact cones this factory builds may switch themselves off while the mode schedule calls a foot a
   * swing foot. See contactConstraintsAreScheduleGated() in common/MpcFormulationConfig.h: it follows the hard
   * `zero_wrench` constraint, which is what used to make the gate sound, and the contact-implicit formulation removes
   * it. When false the cones are also built with a squared-hinge penalty instead of a relaxed log barrier, because a
   * log barrier has a large negative derivative at zero slack and would pay a foot in flight to leave the origin -
   * i.e. would reinvent the force floor that dropping the affine cone offsets exists to remove.
   */
  const bool scheduleGatedContactConstraints_;
  /// Whether `com_and_acom_tracking_cost` is listed, so that the base-pose blocks of state_weights and final_state_weights
  /// are zeroed.
  bool comAndAcomTrackingCostListed_ = false;

  /// Set under basis-vector contact inputs: input_weights are then converted in wrench dimensions and transformed into
  /// basis space.
  std::optional<BasisInputsCostTransformConfig> basisCostTransform_;
};

}  // namespace ocs2::humanoid
