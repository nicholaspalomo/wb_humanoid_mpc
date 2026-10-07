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
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_mpc/SystemObservation.h"
#include "ocs2_sqp/SqpMpc.h"

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_centroidal_mpc/command/CentroidalMpcTargetTrajectoriesCalculator.h"
#include "humanoid_common_mpc/parameter_update/MpcParameterUpdaterModule.h"
#include "humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h"
#include "humanoid_common_mpc_app/node/MpcFiles.h"
#include "humanoid_common_mpc_app/node/MpcNodeRuntime.h"
#include "humanoid_common_mpc_app/visualization/VisualizationPublisher.h"
#include "humanoid_mpc_ipc/MpcMessageConversions.h"
#include "robot_ipc/Bus.h"
#include "robot_realtime/RealtimeThread.h"

namespace ocs2::humanoid {

/**
 * The MPC node of the centroidal MPC (humanoid_centroidal_mpc_node; humanoid_nmpc/docs/distributed_runtime/README.md):
 * the MPC the SQP node of the ROS era built, served on the bus by an MpcServer (node::MpcNodeRuntime) to the robot
 * process, or to a dummy simulator:
 *
 *   - CentroidalMpcInterface::Create() and an SqpMpc on its optimal control problem;
 *   - the procedural motion manager (gait changes and the velocity targets of CentroidalMpcTargetTrajectoriesCalculator),
 *     fed from operator/walking_velocity_command; a reset of the MPC resets it and, through its reset hook, the target
 *     calculator, whose filters are state of the same command path;
 *   - under contact_schedule_source: "contact_planner", the contact planner module;
 *   - the MPC parameter updater (makeCentroidalMpcParameterUpdater(), the live-tuning machinery both formulations share
 *     with the centroidal appliers; makeCommandLimitsReloaders() hands the command limits of a reloaded reference file
 *     to the target calculator and the motion manager), fed from operator/mpc_parameters;
 *   - resets to centroidalMpcResetTargetTrajectories(), the reset target the MRT joint controller hands its in-process
 *     link, so that both paths restart the MPC from the same target;
 *   - every policy carries the planner's target contact poses and the scaled velocity command (ViewerAnnotations);
 *   - the visualization publisher (humanoid_common_mpc_app/visualization), as the ROS nodes always ran their
 *     visualizer: viz/scene and viz/telemetry for the Rerun bridge, from every published policy and every robot/state
 *     sample, on a thread of its own.
 *
 * The synchronized modules run in the order the ROS node registered them: motion manager, contact planner, parameter
 * updater.
 */
class CentroidalMpcNode {
 public:
  struct Options {
    /** The solver thread: the MPC cores of ThreadAffinity.h, SCHED_FIFO when its priority is positive. */
    robot::realtime::RealtimeThreadConfig solverThread = node::defaultSolverThreadConfig(/*realtimePriority=*/0);
    /** The visualization publisher's thread and queue (its defaults suit the node). */
    visualization::VisualizationPublisherOptions visualization;
  };

  /**
   * Builds the MPC from `files` and registers it on `bus`, which must not be running yet and publishes as the MPC node
   * ("mpc"). The errors of CentroidalMpcInterface::Create() (NotFound for a missing file), of the parameter updater, of
   * the visualization publisher (the task file's visualization keys) and of node::MpcNodeRuntime::Create().
   */
  static absl::StatusOr<std::unique_ptr<CentroidalMpcNode>> Create(const node::MpcFiles& files,
                                                                   std::unique_ptr<robot::ipc::Bus> bus,
                                                                   Options options);

  /** Stops the node. */
  ~CentroidalMpcNode();
  CentroidalMpcNode(const CentroidalMpcNode&) = delete;
  CentroidalMpcNode& operator=(const CentroidalMpcNode&) = delete;

  /** Starts the visualization thread, the bus and the solver thread, which waits for the robot's first observation. */
  absl::Status start();

  /** Stops the visualization thread, the solver thread and the bus. Idempotent. */
  void stop();

  /** The target the node resets the MPC to from `observation` (centroidalMpcResetTargetTrajectories()). */
  TargetTrajectories resetTargetTrajectories(const SystemObservation& observation) const;

  /** The dimensions every observation is checked against: those of the effective model, the OCP's input layout. */
  ipc::ModelDimensions dimensions() const;

  CentroidalMpcInterface& interface() { return *interface_; }
  ProceduralMpcMotionManager& motionManager() { return *motionManager_; }
  MpcParameterUpdaterModule& parameterUpdater() { return *parameterUpdater_; }
  /** The MPC, whose solver the parameter updater writes; touch it from the solver thread only, or while it is stopped. */
  SqpMpc& mpc() { return *mpc_; }
  node::MpcNodeRuntime& runtime() { return *runtime_; }
  visualization::VisualizationPublisher& visualization() { return *visualization_; }

 private:
  CentroidalMpcNode() = default;

  std::unique_ptr<CentroidalMpcInterface> interface_;
  std::unique_ptr<SqpMpc> mpc_;
  std::unique_ptr<CentroidalMpcTargetTrajectoriesCalculator> targetCalculator_;
  std::shared_ptr<ProceduralMpcMotionManager> motionManager_;
  std::shared_ptr<MpcParameterUpdaterModule> parameterUpdater_;
  // Before the runtime, so that it outlives the MpcServer whose post-solve observer feeds it.
  std::unique_ptr<visualization::VisualizationPublisher> visualization_;
  // Last, so that it is destroyed first: its solver thread drives everything above.
  std::unique_ptr<node::MpcNodeRuntime> runtime_;
};

}  // namespace ocs2::humanoid
