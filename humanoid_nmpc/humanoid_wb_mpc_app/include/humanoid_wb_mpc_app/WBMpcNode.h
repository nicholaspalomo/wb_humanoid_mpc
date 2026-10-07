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

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_mpc/SystemObservation.h"
#include "ocs2_sqp/SqpMpc.h"

#include "humanoid_common_mpc/parameter_update/MpcParameterUpdaterModule.h"
#include "humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h"
#include "humanoid_common_mpc_app/node/MpcFiles.h"
#include "humanoid_common_mpc_app/node/MpcNodeRuntime.h"
#include "humanoid_common_mpc_app/visualization/VisualizationPublisher.h"
#include "humanoid_mpc_ipc/MpcMessageConversions.h"
#include "humanoid_wb_mpc/WBMpcInterface.h"
#include "humanoid_wb_mpc/command/WBMpcTargetTrajectoriesCalculator.h"
#include "robot_ipc/Bus.h"
#include "robot_realtime/RealtimeThread.h"

namespace ocs2::humanoid {

/**
 * The MPC node of the whole-body MPC (humanoid_wb_mpc_node; humanoid_nmpc/docs/distributed_runtime/README.md): the MPC
 * the whole-body SQP node of the ROS era built, served on the bus by an MpcServer (node::MpcNodeRuntime) to the robot
 * process, or to a dummy simulator:
 *
 *   - WBMpcInterface::Create() and an SqpMpc on its optimal control problem;
 *   - the procedural motion manager (gait changes and the velocity targets of WBMpcTargetTrajectoriesCalculator), fed
 *     from operator/walking_velocity_command; a reset of the MPC resets it and, through its reset hook, the target
 *     calculator;
 *   - resets to wbMpcResetTargetTrajectories(), the reset target the MRT joint controller hands its in-process link;
 *   - the MPC parameter updater (makeWholeBodyMpcParameterUpdater()), registered after the motion manager: the RELOAD_HOT
 *     fields of an update on operator/mpc_parameters (checked against the robot and this task file's configuration by
 *     MpcNodeRuntime), and of the task file when it is saved, are written into the running problem before the next
 *     solve, the RELOAD_START_UP ones that differ logged as taking effect at the next start; a saved reference file's
 *     command limits reach the calculator and the motion manager (makeCommandLimitsReloaders());
 *   - every policy carries the scaled velocity command (ViewerAnnotations);
 *   - the visualization publisher (humanoid_common_mpc_app/visualization), as the ROS nodes always ran their
 *     visualizer: viz/scene and viz/telemetry for the Rerun bridge, from every published policy and every robot/state
 *     sample, on a thread of its own.
 *
 * As in the ROS node, the whole-body MPC has no contact planner, and the policies carry no target contact patch.
 */
class WBMpcNode {
 public:
  struct Options {
    /** The solver thread: the MPC cores of ThreadAffinity.h, SCHED_FIFO when its priority is positive. */
    robot::realtime::RealtimeThreadConfig solverThread = node::defaultSolverThreadConfig(/*realtimePriority=*/0);
    /** The visualization publisher's thread and queue (its defaults suit the node). */
    visualization::VisualizationPublisherOptions visualization;
  };

  /**
   * Builds the MPC from `files` and registers it on `bus`, which must not be running yet and publishes as the MPC node
   * ("mpc"). The errors of WBMpcInterface::Create() (NotFound for a missing file), of the visualization publisher (the
   * task file's visualization fields) and of node::MpcNodeRuntime::Create().
   */
  static absl::StatusOr<std::unique_ptr<WBMpcNode>> Create(const node::MpcFiles& files,
                                                           std::unique_ptr<robot::ipc::Bus> bus,
                                                           Options options);

  /** Stops the node. */
  ~WBMpcNode();
  WBMpcNode(const WBMpcNode&) = delete;
  WBMpcNode& operator=(const WBMpcNode&) = delete;

  /** Starts the visualization thread, the bus and the solver thread, which waits for the robot's first observation. */
  absl::Status start();

  /** Stops the visualization thread, the solver thread and the bus. Idempotent. */
  void stop();

  /** The target the node resets the MPC to from `observation` (wbMpcResetTargetTrajectories()). */
  TargetTrajectories resetTargetTrajectories(const SystemObservation& observation) const;

  /** The dimensions every observation is checked against. */
  ipc::ModelDimensions dimensions() const;

  WBMpcInterface& interface() { return *interface_; }
  /** The MPC, whose solver holds the per-worker copies of the problem the parameter updater writes. */
  SqpMpc& mpc() { return *mpc_; }
  ProceduralMpcMotionManager& motionManager() { return *motionManager_; }
  MpcParameterUpdaterModule& parameterUpdater() { return *parameterUpdater_; }
  node::MpcNodeRuntime& runtime() { return *runtime_; }
  visualization::VisualizationPublisher& visualization() { return *visualization_; }

 private:
  WBMpcNode() = default;

  std::unique_ptr<WBMpcInterface> interface_;
  std::unique_ptr<SqpMpc> mpc_;
  std::unique_ptr<WBMpcTargetTrajectoriesCalculator> targetCalculator_;
  std::shared_ptr<ProceduralMpcMotionManager> motionManager_;
  // Registered with the solver after the motion manager; shared because OCS2's addSynchronizedModule() takes a
  // std::shared_ptr. Never null after Create().
  std::shared_ptr<MpcParameterUpdaterModule> parameterUpdater_;
  // Before the runtime, so that it outlives the MpcServer whose post-solve observer feeds it.
  std::unique_ptr<visualization::VisualizationPublisher> visualization_;
  // Last, so that it is destroyed first: its solver thread drives everything above.
  std::unique_ptr<node::MpcNodeRuntime> runtime_;
};

}  // namespace ocs2::humanoid
