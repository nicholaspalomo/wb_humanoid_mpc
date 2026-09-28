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

#pragma once

#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include "absl/status/statusor.h"

#include <ocs2_mpc/MPC_BASE.h>
#include <ocs2_oc/oc_problem/OptimalControlProblem.h>
#include <ocs2_oc/synchronized_module/SolverSynchronizedModule.h>

#include "humanoid_common_mpc/common/BasisInputsCostTransform.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/contact/ContactWrenchGate.h"
#include "humanoid_common_mpc/contact_planning/ContactPlannerModule.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicLayer.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

/**
 * SolverSynchronizedModule that detects changes to task.yaml and updates cost/constraint
 * parameters in-place on the solver's existing OCP objects. This avoids rebuilding the
 * CentroidalMpcInterface (which would cause dangling-reference segfaults) by using the
 * existing setGains/setWeights/setConfig methods on each cost and penalty object.
 *
 * Supports two update pathways:
 *   1. File-watching: detects task.yaml changes on disk (~1 Hz polling).
 *   2. ROS topic: subscribes to /mpc_parameter_updates (std_msgs/String)
 *      for real-time slider-driven updates without touching the YAML file.
 */
class MpcParameterUpdaterModule : public SolverSynchronizedModule {
 public:
  /**
   * Builds the updater, or returns the InvalidArgument that says which argument is inconsistent.
   *
   * @param mpcPtr             MPC whose SqpSolver OCPs are updated in place (may be nullptr; updates are then skipped).
   * @param taskFile           task.yaml watched for changes; also the base name of the temp file used for topic updates.
   * @param stateDim           Dimension of the OCP state.
   * @param inputDim           Dimension of the OCP input, i.e. the input layout the solver actually optimizes over. With
   *                           basis-vector contact inputs this is the basis-space dimension, not the wrench-space one.
   * @param contactNames       Contact names used to look up per-foot costs and constraints.
   * @param referenceManager   Optional reference manager whose swing trajectory planner is re-configured and which owns
   *                           the ground: a reloaded `terrainHeight` is handed to it (setTerrainHeight), and the
   *                           contact-implicit terms follow the ground it applied (getAppliedTerrainHeight) from the
   *                           next solve on, so that they never disagree with the swing trajectories or the landing
   *                           targets about where it is. Without one, the terms take a reloaded height at once.
   * @param basisCostTransform When set, the OCP uses basis-vector contact inputs. The R matrix in task.yaml is indexed in
   *                           wrench space (forces/moments), so it is loaded with wrenchInputDim rows/cols and transformed
   *                           into basis space with exactly the transform the OCP factory used
   *                           (CentroidalMpcInterface::getBasisInputsCostTransformConfig). The regularization weight and
   *                           its shape (contacts.basisScalingRegularization, contacts.basisRegularization) are reloaded
   *                           with R; a reload that would make the λ block indefinite, names an unknown shape or
   *                           carries a weight that is not a number is refused by key and the running R kept.
   * @return InvalidArgument when basisCostTransform is set and inputDim is not its basisInputDim(), its map does not
   *         have wrenchInputDim rows, its numBasisInputs exceeds inputDim, or it fails
   *         validateBasisInputsCostTransformConfig(). The setGains() the updater writes R with performs no size check, so
   *         a wrench-versus-basis dimension mix-up is refused here, at construction, rather than corrupting the input
   *         cost online.
   */
  static absl::StatusOr<std::unique_ptr<MpcParameterUpdaterModule>> Create(
      MPC_BASE* mpcPtr,
      const std::string& taskFile,
      const std::string& urdfFile,
      const std::string& referenceFile,
      size_t stateDim,
      size_t inputDim,
      const std::vector<std::string>& contactNames,
      SwitchedModelReferenceManager* referenceManager = nullptr,
      std::optional<BasisInputsCostTransformConfig> basisCostTransform = std::nullopt);

  ~MpcParameterUpdaterModule() override = default;

  /**
   * Subscribe to the /mpc_parameter_updates ROS topic for real-time
   * parameter updates from the GUI (without writing to task.yaml).
   */
  void subscribe(rclcpp::Node::SharedPtr node);

  /**
   * Registers the contact planner module (may be nullptr) so that its configuration is hot-reloadable as well: the
   * `contact_planning` block of contact_planning.yaml next to the task file (watched like the task file), or of the task
   * file itself when the block still lives there, and of the YAML published on the parameter topic.
   */
  void setContactPlannerModule(std::shared_ptr<ContactPlannerModule> contactPlannerModule) {
    contactPlannerModulePtr_ = std::move(contactPlannerModule);
  }

  /**
   * Registers the locomotion-heuristic layer so that its COEFFICIENTS follow edits to the task file, exactly as the
   * cost weights beside them do - all of them or, if any is rejected, none. Which heuristics are listed is structural
   * and is not reloaded; the layer says so once per distinct edit to a list on disk. Without this registration the
   * block is launch-time only.
   */
  void setLocomotionHeuristicLayer(std::shared_ptr<LocomotionHeuristicLayer> layer) { locomotionHeuristicLayerPtr_ = std::move(layer); }

  /**
   * Registers something whose parameters come from reference.yaml, to be reloaded when that file changes on disk.
   *
   * The command limits and the command ramps (`maxDisplacementVelocityX/Y`, `maxRotationVelocity`,
   * `maxDeltaPelvisHeight`, the acceleration limits, `defaultBaseHeight`) are read from that file at construction by
   * the target trajectories calculator and the procedural motion manager, so a change to it used to need a restart.
   * The reloaders registered here are called with the file's path whenever it changes, on the solver thread; each
   * consumer is responsible for the thread safety of what it writes (both of the above hold their limits in atomics).
   *
   * A generic hook rather than a typed setter per consumer: what the file feeds differs per robot and per node, and
   * this way the updater does not have to know any of them.
   */
  void addReferenceFileReloader(std::function<void(const std::string&)> reloader) {
    referenceFileReloaders_.push_back(std::move(reloader));
  }

  /**
   * The `contactEstimator` name of the last YAML applied since this was last called (task file or parameter topic), or
   * nullopt when none carried the key. The estimator itself belongs to the MRT joint controller and is swapped on its
   * control thread, so the simulator node polls this from its loop and resolves the name through its
   * ContactEstimatorRegistry. Thread-safe.
   */
  std::optional<std::string> takeContactEstimatorUpdate();

  /**
   * The `contact_wrench_gate` block (debounceTime, rampTime) of the last YAML applied since this was last called, or
   * nullopt when none carried the block. Applied by the simulator node to the MRT joint controller like the contact
   * estimator. Thread-safe.
   */
  std::optional<ContactWrenchGate::Config> takeContactWrenchGateUpdate();

  void preSolverRun(scalar_t initTime,
                    scalar_t finalTime,
                    const vector_t& currentState,
                    const ReferenceManagerInterface& referenceManager) override;

  void postSolverRun(const PrimalSolution& primalSolution) override {}

 private:
  /** Use Create(), which checks the arguments this only stores. */
  MpcParameterUpdaterModule(MPC_BASE* mpcPtr,
                            const std::string& taskFile,
                            const std::string& urdfFile,
                            const std::string& referenceFile,
                            size_t stateDim,
                            size_t inputDim,
                            const std::vector<std::string>& contactNames,
                            SwitchedModelReferenceManager* referenceManager,
                            std::optional<BasisInputsCostTransformConfig> basisCostTransform);

  /**
   * Re-parses a YAML file and applies all parameter updates in-place to every
   * thread-local OCP in the SqpSolver.
   * @param yamlFile  Path to the YAML file to parse (task.yaml or a temp file).
   */
  void applyParameterUpdates(const std::string& yamlFile);

  /** Applies the `contact_planning` block of a YAML file to the contact planner, if both exist. */
  void applyContactPlanningUpdates(const std::string& yamlFile);

  /**
   * Moves the contact-implicit complementarity and ground-penetration terms of every worker's problem onto the ground the
   * reference manager built this solve's references on, when it has moved. Called first thing in every preSolverRun(),
   * which runs right after the reference manager's own, so a reloaded `terrainHeight` reaches the terms in the very solve
   * whose swing trajectories and landing targets were built on it.
   */
  void followAppliedTerrainHeight();

  /** Sets the terrain height of the complementarity and ground-penetration terms of every foot in every problem. */
  void setContactImplicitTermsTerrainHeight(std::vector<OptimalControlProblem>& ocpDefinitions, scalar_t terrainHeight);

  /**
   * Records the controller-side settings of a YAML file for the simulator node: the `contactEstimator` key
   * (takeContactEstimatorUpdate) and the `contact_wrench_gate` block (takeContactWrenchGateUpdate), where present.
   */
  void recordControllerSettings(const std::string& yamlFile);

  /** ROS topic callback — stores the incoming YAML string for the solver thread. */
  void topicCallback(const std_msgs::msg::String::SharedPtr msg);

  MPC_BASE* mpcPtr_;
  const std::string taskFile_;
  const std::string urdfFile_;
  const std::string referenceFile_;
  /// The planner's own configuration file (resolveContactPlanningConfigFile); equals taskFile_ when the block is inline.
  const std::string contactPlanningFile_;

  const size_t stateDim_;
  const size_t inputDim_;
  const std::vector<std::string> contactNames_;
  SwitchedModelReferenceManager* referenceManagerPtr_;
  /// [m] The ground the contact-implicit terms were last set to; empty until the first preSolverRun().
  std::optional<scalar_t> contactImplicitTermsTerrainHeight_;
  /// Set only when the OCP uses basis-vector contact inputs; maps the wrench-space R of task.yaml into basis space. Its
  /// regularization follows the last reload that was applied.
  std::optional<BasisInputsCostTransformConfig> basisCostTransform_;
  /// Optional contact planner whose configuration is hot-reloaded from the `contact_planning` section.
  std::shared_ptr<ContactPlannerModule> contactPlannerModulePtr_;
  std::shared_ptr<LocomotionHeuristicLayer> locomotionHeuristicLayerPtr_;

  // File-watching state
  std::filesystem::file_time_type taskFileLastWriteTime_;
  std::filesystem::file_time_type contactPlanningFileLastWriteTime_;
  std::filesystem::file_time_type referenceFileLastWriteTime_;
  std::vector<std::function<void(const std::string&)>> referenceFileReloaders_;
  size_t checkCounter_{0};

  // Controller-side settings (takeContactEstimatorUpdate, takeContactWrenchGateUpdate)
  std::mutex controllerSettingsMutex_;
  std::optional<std::string> pendingContactEstimator_;
  std::optional<ContactWrenchGate::Config> pendingContactWrenchGate_;

  // ROS topic state
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
  std::mutex pendingMutex_;
  std::string pendingYamlContent_;
  std::atomic<bool> hasNewTopicData_{false};
};

}  // namespace ocs2::humanoid
