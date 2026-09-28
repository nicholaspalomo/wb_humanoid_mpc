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

#include "absl/status/statusor.h"

#include <ocs2_centroidal_model/CentroidalModelInfo.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include <ocs2_core/constraint/StateInputConstraint.h>
#include <ocs2_core/cost/StateCost.h>
#include <ocs2_core/cost/StateInputCost.h>
#include <ocs2_core/penalties/penalties/PenaltyBase.h>
#include <ocs2_core/penalties/penalties/RelaxedBarrierPenalty.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>

#include "humanoid_common_mpc/common/BasisInputsCostTransform.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

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
 */

class HumanoidCostConstraintFactory {
 public:
  HumanoidCostConstraintFactory(const std::string& taskFile,
                                const std::string& referenceFile,
                                const SwitchedModelReferenceManager& referenceManager,
                                const PinocchioInterface& pinocchioInterface,
                                const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                const MpcRobotModelBase<ad_scalar_t>& mpcRobotModelAD,
                                const ModelSettings& modelSettings,
                                bool verbose = false,
                                bool scheduleGatedContactConstraints = true);

  ~HumanoidCostConstraintFactory() = default;
  HumanoidCostConstraintFactory(const HumanoidCostConstraintFactory& other) = delete;

  /**
   * Declares that the problem this factory assembles lists `com_and_acom_tracking_cost`, i.e. that ComAndAcomTrackingCost
   * regulates the base pose. Every quadratic state cost built afterwards - getStateQuadraticCost(),
   * getStateInputQuadraticCost() and getTerminalCost() - then has the base-pose block of its Q or Q_final zeroed
   * (ComAndAcomTrackingCost::zeroBasePoseWeights). Off unless called; the whole-body MPC never calls it.
   */
  void setComAndAcomTrackingCostListed(bool listed) { comAndAcomTrackingCostListed_ = listed; }

  /** Creates the state-input quadratic cost (Q, R); Q's base-pose block is zeroed under setComAndAcomTrackingCostListed. */
  std::unique_ptr<StateInputCost> getStateInputQuadraticCost() const;

  /** Creates the state quadratic tracking cost (Q); its base-pose block is zeroed under setComAndAcomTrackingCostListed. */
  std::unique_ptr<StateInputCost> getStateQuadraticCost() const;

  /**
   * Creates the running CoM and ACoM tracking cost, weighted by the task file's Q_com and Q_acom.
   *
   * @param info Centroidal model info for the same reduced Pinocchio model this
   *     factory was constructed with. Passed in rather than reconstructed here,
   *     because CentroidalModelInfo has no default member initializers and a
   *     hand-built one leaves the contact and nominal-inertia fields indeterminate.
   * @return InvalidArgument naming Q_com or Q_acom when the task file does not carry it, or ComAndAcomTrackingCost::Create()'s
   *     error if the weights, the model or the robot's ACoM network do not fit.
   */
  absl::StatusOr<std::unique_ptr<StateCost>> getComAndAcomTrackingCost(const CentroidalModelInfo& info) const;

  /**
   * Creates the terminal CoM and ACoM tracking cost that stands in for Q_final's zeroed base-pose block: the same cost
   * weighted by terminalCostScaling * Q_com and terminalCostScaling * Q_acom, as Q_final is scaled for the rest of the
   * state (see ComAndAcomTrackingCost). Fails like getComAndAcomTrackingCost(), and with an InvalidArgument naming
   * terminalCostScaling when the task file does not carry it.
   */
  absl::StatusOr<std::unique_ptr<StateCost>> getTerminalComAndAcomTrackingCost(const CentroidalModelInfo& info) const;

  std::unique_ptr<StateInputCost> getInputQuadraticCost() const;

  /**
   * Declares the basis-vector contact input parameterization. The input costs built afterwards load R in wrench
   * dimensions (wrenchInputDim x wrenchInputDim) and transform it into basis space,
   * R_basis = M^T R_wrench M + reg * blkdiag(S, 0), with the regularization S named by `config.regularization`
   * (contacts.basisRegularization) - through transformWrenchInputCostToBasisSpace(R_wrench, config), the same call the
   * online parameter updater makes, so that a hot reload reproduces the start-up R exactly.
   *
   * @param config A config validateBasisInputsCostTransformConfig() accepts; CentroidalMpcInterface builds it.
   */
  void setBasisInputsCostTransform(BasisInputsCostTransformConfig config);

  /** Creates the quadratic terminal cost (terminalCostScaling * Q_final); its base-pose block is zeroed under
   * setComAndAcomTrackingCostListed, and getTerminalComAndAcomTrackingCost() then supplies the terminal regulation. */
  std::unique_ptr<StateCost> getTerminalCost() const;

  std::unique_ptr<StateCost> getFootCollisionConstraint() const;

  std::unique_ptr<StateCost> getJointLimitsConstraint() const;

  std::unique_ptr<StateInputCost> getContactMomentXYConstraint(size_t contactPointIndex, const std::string& name) const;

  /**
   * The contact wrench cone of one foot as a soft constraint, from ContactWrenchConeConstraint::loadConfig() and the
   * barrier parameters of the same task-file block. Wrench-space models only.
   *
   * @return InvalidArgument naming contacts.contactWrenchConeSoftConstraint.mu or .delta when the barrier parameter is not
   *         a finite positive number (a negative mu rewards leaving the cone; an absent key keeps the barrier's default),
   *         loadConfig()'s InvalidArgument naming the missing or out-of-range key of the ground, or
   *         ContactWrenchConeConstraint::Create()'s refusal of a model whose input is not a wrench.
   */
  absl::StatusOr<std::unique_ptr<StateInputCost>> getContactWrenchConeConstraint(size_t contactPointIndex) const;

  std::unique_ptr<StateInputConstraint> getZeroWrenchConstraint(size_t contactPointIndex) const;

  std::unique_ptr<StateInputCost> getFrictionForceConeConstraint(size_t contactPointIndex) const;

  std::unique_ptr<StateInputCost> getExternalTorqueQuadraticCost(size_t contactPointIndex) const;

 private:
  /** Loads the R matrix from task file, optionally transforming from wrench to basis-vector space. */
  matrix_t loadAndTransformR() const;

  /** Loads Q_com and Q_acom, each multiplied by `scaling`, and builds a ComAndAcomTrackingCost on them. */
  absl::StatusOr<std::unique_ptr<StateCost>> makeComAndAcomTrackingCost(const CentroidalModelInfo& info, scalar_t scaling) const;

  std::string taskFile_;
  std::string referenceFile_;
  const SwitchedModelReferenceManager* referenceManagerPtr_;
  const PinocchioInterface* pinocchioInterfacePtr_;
  const MpcRobotModelBase<scalar_t>* mpcRobotModelPtr_;
  const MpcRobotModelBase<ad_scalar_t>* mpcRobotModelADPtr_;
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
  /// Whether `com_and_acom_tracking_cost` is listed, so that the base-pose blocks of Q and Q_final are zeroed.
  bool comAndAcomTrackingCostListed_ = false;

  /// Set under basis-vector contact inputs: R is then loaded in wrench dimensions and transformed into basis space.
  std::optional<BasisInputsCostTransformConfig> basisCostTransform_;
};

}  // namespace ocs2::humanoid
