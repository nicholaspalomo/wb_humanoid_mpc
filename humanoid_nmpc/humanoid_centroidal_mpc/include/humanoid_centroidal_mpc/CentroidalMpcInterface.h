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
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/penalties/Penalties.h"
#include "ocs2_mpc/MPC_Settings.h"
#include "ocs2_oc/rollout/TimeTriggeredRollout.h"
#include "ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "ocs2_pinocchio_interface/PinocchioStateInputMapping.h"
#include "ocs2_robotic_tools/common/RobotInterface.h"
#include "ocs2_robotic_tools/end_effector/EndEffectorKinematics.h"
#include "ocs2_sqp/SqpSettings.h"

#include "humanoid_centroidal_mpc/CentroidalMpcConfig.h"
#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_centroidal_mpc/initialization/CentroidalWeightCompInitializer.h"
#include "humanoid_common_mpc/common/BasisInputsCostTransform.h"
#include "humanoid_common_mpc/common/BasisInputsModelDecorator.h"
#include "humanoid_common_mpc/common/ContactInputParameterization.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcFormulationConfig.h"
#include "humanoid_common_mpc/config/solver/SolverSettingsFromConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactPlannerModule.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningModelParameters.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicLayer.h"
#include "humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

/**
 * The centroidal MPC of a humanoid, built from its typed configuration (CentroidalMpcConfig: the task file, the reference
 * file and the contact planner's file) and a URDF: the model and solver settings, the Pinocchio and centroidal robot
 * models (with the basis-vector decorator under contact_input_parameterization: "basis_vectors"), the reference manager
 * the contact_schedule_source names, the locomotion-heuristic layer, and the optimal control problem with its
 * initializer and rollout. Build it with Create(), or with CreateControllerModels() for the models of a robot process
 * whose MPC runs elsewhere; a node then hands getOptimalControlProblem() to its MPC. Not thread-safe: the solver clones
 * the problem for each of its workers, and synchronizes the reference manager and the modules they share.
 */
class CentroidalMpcInterface final : public RobotInterface {
 public:
  /**
   * The MPC of a robot's configuration: by the paths of its task, URDF and reference files (an entry point that takes
   * the files by path, humanoid_nmpc/humanoid_mpc_config/README.md, "Reading a file": loadCentroidalMpcConfig() and the typed form), or of
   * the typed configuration `config` and the URDF at `urdfFile`, which the interface keeps (config()).
   *
   * @return NotFound naming the path when one of the files does not exist, checked before anything reads them;
   *         loadCentroidalMpcConfig()'s errors (a file that does not parse strictly, such as one in YAML); and InvalidArgument for
   *         a configuration the MPC cannot be built from, naming the field (`contacts.basis_generator_set`, ...), or
   *         with what OCS2's model builders threw. Nothing is thrown.
   */
  static absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> Create(const std::string& taskFile,
                                                                        const std::string& urdfFile,
                                                                        const std::string& referenceFile);
  static absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> Create(CentroidalMpcConfig config, const std::string& urdfFile);

  /**
   * The models an MRT joint controller needs and nothing else, for a robot process whose MPC runs elsewhere (the MPC
   * node, over the bus): the model and solver settings, the Pinocchio model, the centroidal model info, the robot models
   * (the effective one included, the basis-vector decorator under contact_input_parameterization: "basis_vectors"), the
   * initial state and the nominal pendulum length. No reference manager and no optimal control problem: nothing is
   * taped, generated or loaded with CppAD. getOptimalControlProblem(), getInitializer(), getRollout() and the reference
   * manager, contact planner and locomotion-heuristic getters must not be used on it (hasOptimalControlProblem() is
   * false); every other getter is the same as on an interface of Create(). The overloads, and their errors, are
   * Create()'s.
   */
  static absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> CreateControllerModels(const std::string& taskFile,
                                                                                        const std::string& urdfFile,
                                                                                        const std::string& referenceFile);
  static absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> CreateControllerModels(CentroidalMpcConfig config,
                                                                                        const std::string& urdfFile);

  ~CentroidalMpcInterface() override = default;

  /** False for an interface of CreateControllerModels(). */
  bool hasOptimalControlProblem() const { return problemPtr_ != nullptr; }

  const OptimalControlProblem& getOptimalControlProblem() const override {
    CHECK(problemPtr_ != nullptr) << "[CentroidalMpcInterface] built by CreateControllerModels(): there is no optimal control problem";
    return *problemPtr_;
  }

  // CAREFUL: This function is not const, so it can easily be abused. It is currently only for gui purposes. Use with care!
  OptimalControlProblem& getOptimalControlProblemRef() const { return *problemPtr_; }

  /** The configuration the interface was built from: what its parameter updater compares a reload with. */
  const CentroidalMpcConfig& config() const { return config_; }

  const ModelSettings& modelSettings() const { return modelSettings_; }
  const mpc::Settings& mpcSettings() const { return solverSettings_.mpcSettings; }
  const rollout::Settings& rolloutSettings() const { return solverSettings_.rolloutSettings; }
  const sqp::Settings& sqpSettings() const { return solverSettings_.sqpSettings; }

  const vector_t& getInitialState() const { return initialState_; }
  /**
   * [m] The model's center of mass above its feet at the task file's initial_state (computeComHeightAboveFeet): the
   * pendulum length a dcm_terminal_cost or a contact_planning.textproto shared block without com_height stands for.
   */
  scalar_t getNominalComHeight() const { return nominalComHeight_; }
  const RolloutBase& getRollout() const {
    CHECK(rolloutPtr_ != nullptr) << "[CentroidalMpcInterface] built by CreateControllerModels(): there is no rollout";
    return *rolloutPtr_;
  }
  PinocchioInterface& getPinocchioInterface() { return *pinocchioInterfacePtr_; }
  const CentroidalModelInfo& getCentroidalModelInfo() const { return centroidalModelInfo_; }
  std::shared_ptr<SwitchedModelReferenceManager> getSwitchedModelReferenceManagerPtr() const { return referenceManagerPtr_; }

  const CentroidalWeightCompInitializer& getInitializer() const override {
    CHECK(initializerPtr_ != nullptr) << "[CentroidalMpcInterface] built by CreateControllerModels(): there is no initializer";
    return *initializerPtr_;
  }
  std::shared_ptr<ReferenceManagerInterface> getReferenceManagerPtr() const override { return referenceManagerPtr_; }

  const CentroidalMpcRobotModel<scalar_t>& getMpcRobotModel() const { return *mpcRobotModelPtr_; }
  const CentroidalMpcRobotModel<ad_scalar_t>& getMpcRobotModelAD() const { return *mpcRobotModelADPtr_; }

  /** Returns the effective model used by the OCP — either the concrete model or the basis-vector decorator. */
  const MpcRobotModelBase<scalar_t>& getEffectiveMpcRobotModel() const { return *effectiveMpcRobotModelPtr_; }
  const MpcRobotModelBase<ad_scalar_t>& getEffectiveMpcRobotModelAD() const { return *effectiveMpcRobotModelADPtr_; }

  /** The contact input parameterization the task file selects (contact_input_parameterization). */
  ContactInputParameterization contactInputParameterization() const { return contactInputParameterization_; }
  bool usesContactBasisVectorInputs() const { return contactInputParameterization_ == ContactInputParameterization::kBasisVectors; }

  /** The locomotion-heuristic layer, for the parameter updater's hot reload. Never null after construction. */
  const std::shared_ptr<LocomotionHeuristicLayer>& getLocomotionHeuristicLayerPtr() const { return locomotionHeuristicLayerPtr_; }

  /** Where the mode schedule and the footholds come from, as the task file's contact_schedule_source names it. */
  ContactScheduleSource contactScheduleSource() const { return contactScheduleSource_; }
  /**
   * True when the mode schedule and footholds come from the online contact planner (contact_schedule_source:
   * "contact_planner"; contact_planning.textproto planner.type chooses which planner).
   */
  bool usesContactPlanning() const { return contactPlannerModulePtr_ != nullptr; }
  /**
   * The contact planner module, or nullptr when contact planning is off. It must be registered with the solver
   * (SolverBase::addSynchronizedModule) by the node that owns the MPC so that it runs before every solve.
   */
  std::shared_ptr<ContactPlannerModule> getContactPlannerModulePtr() const { return contactPlannerModulePtr_; }

  /**
   * Basis-vector formulation parameters (only meaningful when usesContactBasisVectorInputs() is true).
   * The map M = blkdiag(B_0, B_1, I_joints) converts the basis-vector input into the wrench-space input with every
   * contact wrench expressed in its local contact frame (see BasisInputsModelDecorator::getLocalBasisToWrenchMap).
   */
  const std::optional<matrix_t>& getBasisToWrenchMap() const { return basisToWrenchMap_; }
  size_t getWrenchInputDim() const { return centroidalModelInfo_.inputDim; }
  size_t getNumBasisInputs() const { return basisDecoratorPtr_ ? basisDecoratorPtr_->getNumBasisPerFoot() * kNumContacts : 0; }
  scalar_t getBasisScalingRegularization() const { return basisScalingRegularization_; }
  /** The generator set the basis was built from (contacts.basis_generator_set); empty in wrench-space mode. */
  const std::string& getBasisGeneratorSet() const { return basisGeneratorSet_; }
  const BasisInputsModelDecorator<scalar_t>* absl_nullable getBasisDecoratorPtr() const { return basisDecoratorPtr_.get(); }

  /**
   * The transform of a wrench-space input cost R into basis space, with the regularization the task file names
   * (contacts.basis_regularization); std::nullopt in wrench-space mode. The OCP factory and the online parameter updater
   * both transform through it, so a hot reload reproduces the start-up R.
   */
  std::optional<BasisInputsCostTransformConfig> getBasisInputsCostTransformConfig() const {
    if (!usesContactBasisVectorInputs() || !basisToWrenchMap_.has_value()) {
      return std::nullopt;
    }
    return makeBasisInputsCostTransformConfig(*basisToWrenchMap_);
  }

  std::vector<std::string> getCostNames() const;
  std::vector<std::string> getTerminalCostNames() const;
  std::vector<std::string> getStateSoftConstraintNames() const;
  std::vector<std::string> getSoftConstraintNames() const;
  std::vector<std::string> getEqualityConstraintNames() const;

 private:
  /**
   * Private: Create() and CreateControllerModels() convert the solver and model settings first (solverSettingsFromConfig(),
   * ModelSettings::Create()) and then run the Status-returning set-up steps below, in order. Keeps `config`.
   */
  CentroidalMpcInterface(CentroidalMpcConfig config,
                         ModelSettings modelSettings,
                         SolverSettings solverSettings,
                         const std::string& urdfFile);

  /** What a factory builds: the models of CreateControllerModels(), or the whole MPC of Create(). */
  enum class Scope { kControllerModels, kMpc };

  /**
   * The interface of `config` and the URDF at `urdfFile` with its models (setupModels(), setupContactInputParameterization())
   * and, for the kMpc scope, its reference manager and optimal control problem: Create() and CreateControllerModels().
   */
  static absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> build(CentroidalMpcConfig config,
                                                                       const std::string& urdfFile,
                                                                       Scope scope);

  /**
   * Builds the robot models: the Pinocchio model (loadCustomPinocchioInterface, which checks its actuated joints against
   * model_settings in order), the centroidal model info (centroidal_model and the reference file's
   * default_joint_state), the wrench-space MPC robot models, the initial state (initial_state) and the nominal pendulum
   * length. A model that does not match its settings is an InvalidArgument naming the first joint that differs, a field
   * that does not convert one naming the field.
   */
  absl::Status setupModels();

  /**
   * Reads contact_input_parameterization and, under basis vectors, builds the per-foot bases (contacts.basis_generator_set
   * on contacts.contact_wrench_cone_soft_constraint and each foot's contacts.contact_rectangle), the decorated robot
   * models and the regularization of the input cost (contacts.basis_regularization, contacts.basis_scaling_regularization).
   * Keys the CppAD model folder by the parameterization and the basis. Every configuration error is an InvalidArgument
   * naming the field to change.
   */
  absl::Status setupContactInputParameterization();

  /** Appends `libraryKey` to the CppAD model folder, so that no cached library is shared across parameterizations or bases. */
  void keyCppAdModelFolder(absl::string_view libraryKey);

  /** The transform of getBasisInputsCostTransformConfig() around `basisToWrenchMap`, under basis-vector inputs. */
  BasisInputsCostTransformConfig makeBasisInputsCostTransformConfig(const matrix_t& basisToWrenchMap) const {
    BasisInputsCostTransformConfig config;
    config.basisToWrenchMap = basisToWrenchMap;
    config.wrenchInputDim = getWrenchInputDim();
    config.numBasisInputs = getNumBasisInputs();
    config.lambdaRegularization = basisScalingRegularization_;
    config.regularization = basisRegularization_;
    return config;
  }

  /**
   * Reads contact_schedule_source and builds the reference manager it names: the gait schedule's (the reference file's
   * initial_mode_schedule and default_mode_sequence_template), or the online contact planner's under "contact_planner",
   * on the contact planner's file or, for a robot without one, the planner's library defaults.
   */
  absl::Status setupReferenceManager();

  /**
   * Builds the locomotion-heuristic layer from the task file's locomotion_heuristics and installs it on the reference
   * manager.
   *
   * Called first by setupOptimalControlProblem(), because the two input costs it builds afterwards read their
   * contact-force reference through the reference manager and so through this layer.
   */
  absl::Status setupLocomotionHeuristics();

  absl::Status setupOptimalControlProblem();

  std::unique_ptr<StateInputConstraint> getStanceFootConstraint(const EndEffectorKinematics<scalar_t>& eeKinematics,
                                                                size_t contactPointIndex);
  std::unique_ptr<StateInputConstraint> getNormalVelocityConstraint(const EndEffectorKinematics<scalar_t>& eeKinematics,
                                                                    size_t contactPointIndex);

  /** The task_space_costs of the task file, one cost per entry; InvalidArgument naming the entry that does not convert. */
  absl::Status addTaskSpaceKinematicsCosts(
      const PinocchioStateInputMapping<ad_scalar_t>& pinocchioMappingCppAd,
      const PinocchioEndEffectorKinematicsCppAd::update_pinocchio_interface_callback& velocityUpdateCallback);

  // The configuration the interface is built from (config()); the cost and constraint factory reads the task file in it.
  CentroidalMpcConfig config_;
  ModelSettings modelSettings_;
  // The MPC, SQP and rollout settings and interface.verbose of the task file.
  SolverSettings solverSettings_;

  std::unique_ptr<PinocchioInterface> pinocchioInterfacePtr_;
  CentroidalModelInfo centroidalModelInfo_;

  std::unique_ptr<OptimalControlProblem> problemPtr_;
  std::shared_ptr<SwitchedModelReferenceManager> referenceManagerPtr_;
  std::shared_ptr<ContactPlannerModule> contactPlannerModulePtr_;
  std::optional<ContactPlanningModelParameters> contactPlanningModelParameters_;  // derived from the model, applied to every planner config
  // The reference-shaping layer of Bledt's RPC heuristics, also held by the reference manager. Kept here so that the
  // parameter updater can reach it to hot-reload the coefficients; empty, and therefore an exact no-op, on every robot
  // whose task file lists none. See humanoid_nmpc/docs/locomotion_heuristics/README.md.
  std::shared_ptr<LocomotionHeuristicLayer> locomotionHeuristicLayerPtr_;

  std::unique_ptr<CentroidalMpcRobotModel<scalar_t>> mpcRobotModelPtr_;
  std::unique_ptr<CentroidalMpcRobotModel<ad_scalar_t>> mpcRobotModelADPtr_;

  /// Effective model used by the OCP. Points to either the concrete model or the decorator.
  /// Owned by either mpcRobotModelPtr_ (non-decorated) or basisDecoratorPtr_ (decorated).
  MpcRobotModelBase<scalar_t>* absl_nullable effectiveMpcRobotModelPtr_ = nullptr;
  MpcRobotModelBase<ad_scalar_t>* absl_nullable effectiveMpcRobotModelADPtr_ = nullptr;

  /// Basis-vector decorator models (owning pointers, only populated when active).
  std::unique_ptr<BasisInputsModelDecorator<scalar_t>> basisDecoratorPtr_;
  std::unique_ptr<BasisInputsModelDecorator<ad_scalar_t>> basisDecoratorADPtr_;

  ContactInputParameterization contactInputParameterization_ = kDefaultContactInputParameterization;
  ContactScheduleSource contactScheduleSource_ = kDefaultContactScheduleSource;
  /// Local-frame basis-to-wrench map M (wrenchInputDim × basisInputDim); populated only in basis-vector mode.
  std::optional<matrix_t> basisToWrenchMap_;
  /// Name of the generator set of the bases (contacts.basis_generator_set); set only in basis-vector mode.
  std::string basisGeneratorSet_;
  /// Weight of the regularization added to the λ block of the transformed input cost R_basis.
  scalar_t basisScalingRegularization_ = 0.0;
  /// Name of its shape S (contacts.basis_regularization); set only in basis-vector mode.
  std::string basisRegularization_;

  std::unique_ptr<RolloutBase> rolloutPtr_;
  std::unique_ptr<CentroidalWeightCompInitializer> initializerPtr_;

  vector_t initialState_;
  scalar_t nominalComHeight_ = 0.0;  // [m] see getNominalComHeight()

  const std::string urdfFile_;
  // The task file's interface.verbose: whether the set-up logs what it builds.
  bool verbose_;
};

}  // namespace ocs2::humanoid
