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
#include <string>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/penalties/Penalties.h"
#include "ocs2_mpc/MPC_Settings.h"
#include "ocs2_oc/rollout/TimeTriggeredRollout.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "ocs2_robotic_tools/common/RobotInterface.h"
#include "ocs2_sqp/SqpSettings.h"

#include "humanoid_common_mpc/HumanoidCostConstraintFactory.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcFormulationConfig.h"
#include "humanoid_common_mpc/config/solver/SolverSettingsFromConfig.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/initialization/WeightCompInitializer.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_mpc_config/joint_weights.nproto.h"
#include "humanoid_mpc_config/reference_file.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"
#include "humanoid_wb_mpc/end_effector/EndEffectorDynamics.h"

namespace ocs2::humanoid {

/**
 * The whole-body MPC of a robot, built from its task file, URDF and reference file: the model and solver settings, the
 * Pinocchio model, the whole-body robot models, the gait-schedule reference manager and the optimal control problem the
 * task file's formulation lists (costs, soft and hard constraints, the CppAD-taped dynamics), with its rollout and
 * initializer. Made by Create(), or by CreateControllerModels() for the models of a robot process whose MPC runs
 * elsewhere. The accessors are const and safe to call concurrently; the reference manager and the Pinocchio interface it
 * hands out are not thread-safe themselves.
 *
 * Both factories come in two forms with the same meaning: of the typed task and reference files
 * (humanoid_nmpc/humanoid_mpc_config: TaskFile, ReferenceFile), and of the files at a path - the root of the MPC's
 * configuration - which loads them strictly (loadTaskFile(), loadReferenceFile()) and builds the typed form, prefixing its
 * errors with the file they are about.
 */
class WBMpcInterface final : public RobotInterface {
 public:
  /**
   * The whole-body MPC of the task file `taskFile`, the URDF `urdfFile` and the reference file `referenceFile`. Nothing
   * of the two files is retained.
   *
   * @return The interface, or InvalidArgument: naming the field of a block that does not convert
   *         (state_weights.joint_positions, ...); naming a formulation this MPC does not implement - the
   *         contact-implicit terms, com_and_acom_tracking_cost, dcm_terminal_cost, contact_input_parameterization
   *         "basis_vectors" or contact_schedule_source "contact_planner", all of them the centroidal MPC's; naming the
   *         URDF when it does not parse or its joints are not model_settings'; or with what OCS2's CppAD code
   *         generation threw for a library it cannot build or load. Nothing is thrown.
   */
  static absl::StatusOr<std::unique_ptr<WBMpcInterface>> Create(const mpc_config::TaskFile& taskFile,
                                                                const std::string& urdfFile,
                                                                const mpc_config::ReferenceFile& referenceFile);

  /**
   * Create() of the files at the paths `taskFile`, `urdfFile` and `referenceFile`.
   *
   * @return NotFound naming the path when one of the three files does not exist, checked before anything reads them;
   *         InvalidArgument naming the file, line and column of a task or reference file that does not parse; and the
   *         typed form's errors, prefixed with the file they are about.
   */
  static absl::StatusOr<std::unique_ptr<WBMpcInterface>> Create(const std::string& taskFile,
                                                                const std::string& urdfFile,
                                                                const std::string& referenceFile);

  /**
   * The models an MRT joint controller needs and nothing else, for a robot process whose MPC runs elsewhere (the MPC
   * node, over the bus): the model and solver settings, the Pinocchio model, the robot models, the reference manager
   * built on them and the initial state. No optimal control problem: nothing is taped, generated or loaded with CppAD.
   * getOptimalControlProblem(), getInitializer() and getRollout() must not be used on it (hasOptimalControlProblem() is
   * false). Fails like Create(), except that the formulation is not read.
   */
  static absl::StatusOr<std::unique_ptr<WBMpcInterface>> CreateControllerModels(const mpc_config::TaskFile& taskFile,
                                                                                const std::string& urdfFile,
                                                                                const mpc_config::ReferenceFile& referenceFile);

  /** CreateControllerModels() of the files at the paths, failing like the path form of Create(). */
  static absl::StatusOr<std::unique_ptr<WBMpcInterface>> CreateControllerModels(const std::string& taskFile,
                                                                                const std::string& urdfFile,
                                                                                const std::string& referenceFile);

  ~WBMpcInterface() override = default;
  WBMpcInterface(const WBMpcInterface&) = delete;
  WBMpcInterface& operator=(const WBMpcInterface&) = delete;

  /** False for an interface of CreateControllerModels(). */
  bool hasOptimalControlProblem() const { return problemPtr_ != nullptr; }

  const OptimalControlProblem& getOptimalControlProblem() const override {
    CHECK(problemPtr_ != nullptr) << "[WBMpcInterface] built by CreateControllerModels(): there is no optimal control problem";
    return *problemPtr_;
  }

  const ModelSettings& modelSettings() const { return modelSettings_; }
  /**
   * Returns the task file the interface was built from: what the parameter updater compares a reloaded file's
   * RELOAD_START_UP fields with.
   */
  const mpc_config::TaskFile& taskFile() const { return taskFile_; }
  const mpc::Settings& mpcSettings() const { return mpcSettings_; }
  const rollout::Settings& rolloutSettings() const { return rolloutSettings_; }
  const sqp::Settings& sqpSettings() const { return sqpSettings_; }

  const vector_t& getInitialState() const { return initialState_; }
  const RolloutBase& getRollout() const {
    CHECK(rolloutPtr_ != nullptr) << "[WBMpcInterface] built by CreateControllerModels(): there is no rollout";
    return *rolloutPtr_;
  }
  PinocchioInterface& getPinocchioInterface() { return *pinocchioInterfacePtr_; }
  std::shared_ptr<SwitchedModelReferenceManager> getSwitchedModelReferenceManagerPtr() const { return referenceManagerPtr_; }

  const WeightCompInitializer& getInitializer() const override {
    CHECK(initializerPtr_ != nullptr) << "[WBMpcInterface] built by CreateControllerModels(): there is no initializer";
    return *initializerPtr_;
  }
  std::shared_ptr<ReferenceManagerInterface> getReferenceManagerPtr() const override { return referenceManagerPtr_; }

  const WBAccelMpcRobotModel<scalar_t>& getMpcRobotModel() const { return *mpcRobotModelPtr_; }
  const WBAccelMpcRobotModel<ad_scalar_t>& getMpcRobotModelAD() const { return *mpcRobotModelADPtr_; }

 private:
  /** What a factory builds: the models of CreateControllerModels(), or the whole MPC of Create(). */
  enum class Scope {
    kControllerModels,
    kMpc,
  };

  /**
   * The names the errors of a typed file's conversion are prefixed with: the files' paths for the path forms of the
   * factories, empty for the typed forms, whose errors name their fields alone.
   */
  struct ConfigSources {
    std::string taskFile;
    std::string referenceFile;
  };

  /** The factories: the interface of `scope` from the typed files, its errors prefixed with `sources`. */
  static absl::StatusOr<std::unique_ptr<WBMpcInterface>> build(const mpc_config::TaskFile& taskFile,
                                                               const std::string& urdfFile,
                                                               const mpc_config::ReferenceFile& referenceFile,
                                                               const ConfigSources& sources,
                                                               Scope scope);

  /** The path forms of the factories: the files loaded (loadTaskFile(), loadReferenceFile()) and build(). */
  static absl::StatusOr<std::unique_ptr<WBMpcInterface>> buildFromFiles(const std::string& taskFile,
                                                                        const std::string& urdfFile,
                                                                        const std::string& referenceFile,
                                                                        Scope scope);

  /** Takes over the settings; the factories then build the models (setupModels()) and the problem. */
  explicit WBMpcInterface(ModelSettings modelSettings, const SolverSettings& solverSettings);

  /**
   * Builds the Pinocchio model of `urdfFile` (loadCustomPinocchioInterface(), which checks its actuated joints against
   * model_settings in order), the MPC robot models, the reference manager on `gaitSchedule` and the swing trajectory
   * planner of `taskFile`, and the initial state. A model that does not match its settings is an InvalidArgument naming
   * the first joint that differs, returned rather than thrown.
   */
  absl::Status setupModels(const mpc_config::TaskFile& taskFile, const std::string& urdfFile, std::shared_ptr<GaitSchedule> gaitSchedule);

  /**
   * Builds the optimal control problem of the formulation `taskFile` lists, with its rollout and initializer, after
   * setupModels(). The factory reads `taskFile` only while this runs.
   */
  absl::Status setupOptimalControlProblem(const mpc_config::TaskFile& taskFile);

  /** Adds the costs, the terminal cost and the state soft constraints `tasks` lists, in this order, to the problem. */
  absl::Status addCostsAndStateConstraints(const MpcFormulationTasks& tasks,
                                           const mpc_config::TaskFile& taskFile,
                                           const HumanoidCostConstraintFactory& factory);

  /** Adds the terms of each foot that `tasks` lists - its cones, its stance, swing and mimic constraints and its foot cost. */
  absl::Status addContactTerms(const MpcFormulationTasks& tasks,
                               const mpc_config::TaskFile& taskFile,
                               const HumanoidCostConstraintFactory& factory);

  /** The zero-acceleration constraint of the stance foot `contactPointIndex`; `eeDynamics` has its one end effector. */
  absl::StatusOr<std::unique_ptr<StateInputConstraint>> getStanceFootConstraint(const EndEffectorDynamics<scalar_t>& eeDynamics,
                                                                                size_t contactPointIndex);
  /** The normal-velocity constraint of the swing foot `contactPointIndex`; `eeDynamics` has its one end effector. */
  absl::StatusOr<std::unique_ptr<StateInputConstraint>> getNormalVelocityConstraint(const EndEffectorDynamics<scalar_t>& eeDynamics,
                                                                                    size_t contactPointIndex);
  /**
   * The joint torque cost on the joint_torque_weights `weights`, one per joint of the MPC model; the errors of
   * jointTorqueWeightsFromConfig().
   */
  absl::StatusOr<std::unique_ptr<StateInputCost>> makeJointTorqueCost(const mpc_config::JointWeights& weights) const;

  // The task file build() made the interface from.
  mpc_config::TaskFile taskFile_;
  ModelSettings modelSettings_;
  mpc::Settings mpcSettings_;
  sqp::Settings sqpSettings_;
  rollout::Settings rolloutSettings_;
  // The task file's interface.verbose: whether the settings and terms are logged as they are built.
  bool verbose_ = false;

  std::unique_ptr<PinocchioInterface> pinocchioInterfacePtr_;

  std::unique_ptr<OptimalControlProblem> problemPtr_;
  std::shared_ptr<SwitchedModelReferenceManager> referenceManagerPtr_;

  std::unique_ptr<WBAccelMpcRobotModel<scalar_t>> mpcRobotModelPtr_;
  std::unique_ptr<WBAccelMpcRobotModel<ad_scalar_t>> mpcRobotModelADPtr_;

  std::unique_ptr<RolloutBase> rolloutPtr_;
  std::unique_ptr<WeightCompInitializer> initializerPtr_;

  vector_t initialState_;
};

}  // namespace ocs2::humanoid
