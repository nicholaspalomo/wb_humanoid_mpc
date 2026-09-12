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
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include <ocs2_mpc/MPC_BASE.h>
#include <ocs2_oc/synchronized_module/SolverSynchronizedModule.h>

#include "humanoid_common_mpc/common/BasisInputsCostTransform.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
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
   * @param mpcPtr             MPC whose SqpSolver OCPs are updated in place (may be nullptr; updates are then skipped).
   * @param taskFile           task.yaml watched for changes; also the base name of the temp file used for topic updates.
   * @param stateDim           Dimension of the OCP state.
   * @param inputDim           Dimension of the OCP input, i.e. the input layout the solver actually optimizes over. With
   *                           basis-vector contact inputs this is the basis-space dimension, not the wrench-space one.
   * @param contactNames       Contact names used to look up per-foot costs and constraints.
   * @param referenceManager   Optional reference manager whose swing trajectory planner is re-configured.
   * @param basisCostTransform When set, the OCP uses basis-vector contact inputs. The R matrix in task.yaml is indexed in
   *                           wrench space (forces/moments), so it is loaded with wrenchInputDim rows/cols and transformed
   *                           into basis space with exactly the same transform the OCP factory used. inputDim must equal
   *                           basisCostTransform->basisInputDim(); otherwise std::invalid_argument is thrown.
   */
  MpcParameterUpdaterModule(MPC_BASE* mpcPtr,
                            const std::string& taskFile,
                            const std::string& urdfFile,
                            const std::string& referenceFile,
                            size_t stateDim,
                            size_t inputDim,
                            const std::vector<std::string>& contactNames,
                            const SwitchedModelReferenceManager* referenceManager = nullptr,
                            std::optional<BasisInputsCostTransformConfig> basisCostTransform = std::nullopt);

  ~MpcParameterUpdaterModule() override = default;

  /**
   * Subscribe to the /mpc_parameter_updates ROS topic for real-time
   * parameter updates from the GUI (without writing to task.yaml).
   */
  void subscribe(rclcpp::Node::SharedPtr node);

  void preSolverRun(scalar_t initTime,
                    scalar_t finalTime,
                    const vector_t& currentState,
                    const ReferenceManagerInterface& referenceManager) override;

  void postSolverRun(const PrimalSolution& primalSolution) override {}

 private:
  /**
   * Re-parses a YAML file and applies all parameter updates in-place to every
   * thread-local OCP in the SqpSolver.
   * @param yamlFile  Path to the YAML file to parse (task.yaml or a temp file).
   */
  void applyParameterUpdates(const std::string& yamlFile);

  /** ROS topic callback — stores the incoming YAML string for the solver thread. */
  void topicCallback(const std_msgs::msg::String::SharedPtr msg);

  MPC_BASE* mpcPtr_;
  const std::string taskFile_;
  const std::string urdfFile_;
  const std::string referenceFile_;

  const size_t stateDim_;
  const size_t inputDim_;
  const std::vector<std::string> contactNames_;
  const SwitchedModelReferenceManager* referenceManagerPtr_;
  /// Set only when the OCP uses basis-vector contact inputs; maps the wrench-space R of task.yaml into basis space.
  const std::optional<BasisInputsCostTransformConfig> basisCostTransform_;

  // File-watching state
  std::filesystem::file_time_type taskFileLastWriteTime_;
  size_t checkCounter_{0};

  // ROS topic state
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
  std::mutex pendingMutex_;
  std::string pendingYamlContent_;
  std::atomic<bool> hasNewTopicData_{false};
};

}  // namespace ocs2::humanoid
