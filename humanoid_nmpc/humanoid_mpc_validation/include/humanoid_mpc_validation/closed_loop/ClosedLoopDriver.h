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

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <mujoco_sim_interface/MujocoSimInterface.h>
#include <ocs2_mpc/SystemObservation.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <ocs2_sqp/SqpMpc.h>
#include <robot_model/ContactEstimatorRegistry.h>
#include <robot_model/RobotDescription.h>
#include <robot_model/RobotJointAction.h>
#include <robot_model/RobotState.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/command/TargetTrajectoriesCalculatorBase.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/mrt/InProcessMpcLink.h"
#include "humanoid_common_mpc/mrt/MpcLink.h"
#include "humanoid_common_mpc/mrt/MpcResetSupervisor.h"
#include "humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h"
#include "humanoid_common_mpc_app/robot/ControllerSideSettings.h"
#include "humanoid_common_mpc_app/robot/RobotController.h"
#include "humanoid_common_mpc_app/robot/RobotProcessSettings.h"
#include "humanoid_common_mpc_app/robot/TaskFileWatcher.h"
#include "humanoid_mpc_validation/closed_loop/RobotConfiguration.h"

namespace ocs2::humanoid::validation {

/** How a driver builds its MPC. */
struct ClosedLoopDriverOptions {
  /// Overrides the task file's sqp.nThreads; the determinism test uses 1, because with more threads the order in which
  /// the per-thread cost sums are added varies and the solves are not reproducible bit for bit.
  std::optional<size_t> solverThreads;
};

/** One solver iteration, timed. */
struct DriverSolveOutcome {
  absl::Status status;             ///< InProcessMpcLink::runSolverIteration(): MPC_MRT_Interface::advanceMpc()
  double retryDelay = 0.0;         ///< [s] MpcResetSupervisor::onSolveResult(): wait this long before the next attempt
  double wallTimeMs = 0.0;         ///< [ms] the whole iteration: a reset served, the synchronized modules and the SQP
  double lqApproximationMs = 0.0;  ///< SqpSolver::getBenchmarks() of the iteration's solve
  double solveQpMs = 0.0;
  double linesearchMs = 0.0;
  double computeControllerMs = 0.0;
};

/**
 * The gap delta_x0 a solve started from: the observation against node 0 of the SQP's warm start (the previous solution
 * at the observation time, or the observation itself after a reset).
 */
struct InitialStateGap {
  double rotation = 0.0;  ///< [rad] of its base orientation part
  double norm = 0.0;      ///< of the whole state difference
};

/**
 * A formulation's two processes in one, for the lockstep closed loop, without the bus and without threads:
 *  - the MPC as the formulation's MPC node builds it (CentroidalMpcNode::Create(), WBMpcNode::Create()): the interface,
 *    the SQP MPC with the reference manager and the synchronized modules, the target trajectories calculator and the
 *    procedural motion manager behind the velocity command;
 *  - the robot process's controller as the formulation's robot binary builds it (CentroidalMpcRobotMain.cpp,
 *    WBMpcRobotMain.cpp): the MRT joint controller configured from the task file's robot-process keys
 *    (loadRobotProcessSettings()), behind the RobotController the robot process drives (MrtRobotController, with the
 *    binary's CycleInputOrder), and the controller-side keys of the task file watched and applied as the robot process
 *    applies them;
 *  - in place of the bus between the two, an InProcessMpcLink that runs no solver thread (Execution::kCaller): the
 *    runner calls its solver iterations (solve()) on the simulation's clock;
 *  - the GUI's command path: a WalkingVelocityCommand message, applied to the motion manager by the MPC node's own
 *    node::applyWalkingVelocityCommand(), which clamps it and scales it (ProceduralMpcMotionManager::setAndScaleVelocityCommand());
 *  - the measurements of the formulation the metrics need (contact positions, the initial-state gap).
 * Left out: the bus, the FSM bridge and the fall recovery (the runner drives the modes and the gantry itself), the
 * visualization, telemetry, the operator's parameter and PD-gain messages.
 *
 * This base class holds what both formulations share; CentroidalClosedLoopDriver and WholeBodyClosedLoopDriver add
 * their controller.
 */
class ClosedLoopDriver {
 public:
  virtual ~ClosedLoopDriver() = default;

  ClosedLoopDriver(const ClosedLoopDriver&) = delete;
  ClosedLoopDriver& operator=(const ClosedLoopDriver&) = delete;

  const RobotConfiguration& configuration() const { return configuration_; }
  /** The robot process's keys of the task file (loadRobotProcessSettings()), as the robot binaries read them. */
  const RobotProcessSettings& settings() const { return settings_; }
  const robot::model::RobotDescription& robotDescription() const { return *robotDescription_; }
  /** The joints of the MPC model, as indices of robotDescription(). */
  const std::vector<robot::joint_index_t>& mpcJointIndices() const { return mpcJointIndices_; }

  /** [Hz] The task file's mpcDesiredFrequency and mrtDesiredFrequency. */
  double mpcFrequency() const { return mpcFrequency_; }
  double mrtFrequency() const { return mrtFrequency_; }
  /** The SQP's threads, after ClosedLoopDriverOptions::solverThreads. */
  size_t solverThreads() const { return solverThreads_; }
  /** The MPC's initial state, from the task file. */
  const vector_t& initialMpcState() const { return initialMpcState_; }
  /** The state the robot is spawned in (createInitialSimState()), which is also the JOINT_PD posture. */
  const robot::model::RobotState& initialRobotState() const { return *initialRobotState_; }

  /**
   * The simulator of the robot binary, headless: the MujocoSimConfig of its mujoco backend (MujocoRobotBackend::makeConfig())
   * for the scene, the initial state, the contact frames and the simulator keys of the task file. The gantry starts
   * locked at the initial height. The viewer's target contact patches are left out: there is no viewer.
   */
  absl::StatusOr<robot::mujoco_sim_interface::MujocoSimConfig> simulatorConfig() const;

  // ---------------------------------------------------------------- the operator's side

  /**
   * The GUI publishes `message` (linear_velocity_x, linear_velocity_y, desired_pelvis_height, angular_velocity_z): the
   * MPC node's node::applyWalkingVelocityCommand() hands it to the motion manager, clamped and scaled, and the next
   * solve reads it. A message the node refuses (a value that is not finite) is logged and changes nothing.
   */
  void setGuiVelocityCommand(const Eigen::Vector4d& message);
  /** maxDisplacementVelocityX, maxDisplacementVelocityY and maxRotationVelocity of reference.yaml. */
  const Eigen::Vector3d& commandLimits() const { return commandLimits_; }
  /** [m] defaultBaseHeight of reference.yaml, the pelvis height the GUI's slider sends. */
  double defaultPelvisHeight() const { return defaultPelvisHeight_; }
  /** The velocity reference of the last solve: (forward, lateral, yaw rate) after the command filter and the ramps. */
  Eigen::Vector3d referenceVelocity() const;
  /** [m] The world height of the base a command with `pelvisHeight` asks for. */
  double referenceBaseHeight(double pelvisHeight) const;

  // ---------------------------------------------------------------- the solver

  /**
   * One solver iteration of the MPC link (InProcessMpcLink::runSolverIteration()), timed, with the SQP's phase times:
   * what the solver thread of the MPC node runs between its waits.
   */
  DriverSolveOutcome solve();

  /**
   * The gap the last solve started from, as the SQP recorded it (SqpSolver::getInitialStateGap()); empty before the
   * first solve. Call it after a successful solve(): it then includes the reset the iteration may have served first and
   * the trajectory spread, which an estimate from the previous solution before the solve would miss.
   */
  std::optional<InitialStateGap> lastSolveInitialStateGap() const;

  /** max | |xi_k| - 1 | over the nodes of the last solution; empty for the Euler formulation, which has no quaternion. */
  virtual std::optional<double> maxQuaternionNormDeviation() const { return std::nullopt; }

  /** The world positions of the MPC's contact frames at the configuration of `observation`. */
  std::array<Eigen::Vector3d, 2> contactPositions(const SystemObservation& observation);

  // ---------------------------------------------------------------- the robot process's side

  /** The controller as the robot process drives it. */
  RobotController& robotController() { return *robotController_; }
  const RobotController& robotController() const { return *robotController_; }

  /**
   * Installs the task file's contact estimator, as RobotProcess::Create() does: the backend's estimators (cheater_sim,
   * reading `sim`'s ground truth) registered next to the built-in ones. InvalidArgument for a name nobody registered.
   */
  absl::Status connectSimulator(const robot::mujoco_sim_interface::MujocoSimInterface& sim);

  /**
   * The robot process's start of the MPC (RobotController::startMpc()): the link serves the start-up reset from the
   * observation of `robotState`, here, on the calling thread. The link starts once: a later call (the solve benchmark's
   * next pass) requests a full reset instead, which the next solve serves from the observation then current.
   */
  void startMpc(const robot::model::RobotState& robotState);

  /**
   * The controller-side keys of the task file (`contactEstimator`, `contact_wrench_gate`), re-read when the file has
   * changed and applied as RobotProcess::applyControllerSettings() applies them. The robot process checks the file once
   * a second of wall time; the runner calls this every control cycle with the simulation time and the file is checked
   * once a second of it.
   */
  void applyControllerSideSettings(double time);

  /** True while the controller holds the robot for a policy solved after the entry into WB_MPC, or ramps into it. */
  virtual bool isEnteringMpc() const = 0;
  /** The supervisor of the MPC link's resets and solves. */
  const MpcResetSupervisor& resetSupervisor() const { return mpcLink_->getResetSupervisor(); }
  virtual const SystemObservation& currentObservation() const = 0;
  virtual const vector_t& latestPolicyInput() const = 0;

 protected:
  ClosedLoopDriver() = default;

  /**
   * Fills the members below that every formulation shares, after the derived driver has built its interface: the MPC
   * on `problem`, the command path and the measurement model. `robotModel` is the model the controller observes with.
   */
  absl::Status initializeShared(const RobotConfiguration& configuration,
                                const ClosedLoopDriverOptions& options,
                                const ModelSettings& modelSettings,
                                const mpc::Settings& mpcSettings,
                                const sqp::Settings& sqpSettings,
                                const OptimalControlProblem& problem,
                                const Initializer& initializer,
                                std::shared_ptr<ReferenceManagerInterface> referenceManager,
                                const MpcRobotModelBase<scalar_t>& robotModel,
                                const PinocchioInterface& pinocchioInterface,
                                const vector_t& initialMpcState);

  /** The procedural motion manager behind the velocity command, with `calculator` building its targets. */
  absl::Status initializeCommandPath(std::unique_ptr<TargetTrajectoriesCalculatorBase> calculator,
                                     std::shared_ptr<SwitchedModelReferenceManager> referenceManager,
                                     const MpcRobotModelBase<scalar_t>& commandModel);

  /**
   * The factory of the controller's MPC link: an InProcessMpcLink over the MPC that runs no solver thread, reported to
   * mpcLink_ when the controller's constructor makes it.
   */
  MpcLinkFactory lockstepMpcLinkFactory(absl::string_view solverName);

  /** Destroys the controller, then the MPC: the derived destructor calls it before its interface goes. */
  void releaseControllerAndMpc();

  RobotConfiguration configuration_;
  RobotProcessSettings settings_;
  std::unique_ptr<robot::model::RobotDescription> robotDescription_;
  std::vector<robot::joint_index_t> mpcJointIndices_;
  const ModelSettings* modelSettings_ = nullptr;  ///< the interface's, which the derived driver owns
  double mpcFrequency_ = 0.0;
  double mrtFrequency_ = 0.0;
  size_t solverThreads_ = 0;
  vector_t initialMpcState_;
  std::optional<robot::model::RobotState> initialRobotState_;

  std::unique_ptr<SqpMpc> mpc_;
  std::unique_ptr<TargetTrajectoriesCalculatorBase> calculator_;
  std::shared_ptr<ProceduralMpcMotionManager> motionManager_;
  Eigen::Vector3d commandLimits_ = Eigen::Vector3d::Ones();
  double defaultPelvisHeight_ = 0.0;

  std::unique_ptr<MpcRobotModelBase<scalar_t>> measurementModel_;
  std::unique_ptr<PinocchioInterface> measurementPinocchio_;
  std::array<size_t, 2> contactFrameIds_{0, 0};

  // The robot process's controller (made by the derived driver) and the link it owns, which this driver solves.
  std::unique_ptr<RobotController> robotController_;
  InProcessMpcLink* mpcLink_ = nullptr;
  bool mpcStarted_ = false;

  // The controller-side keys of the task file (applyControllerSideSettings()).
  robot::model::ContactEstimatorRegistry contactEstimators_;
  std::string contactEstimatorName_;  ///< canonical name of the estimator the controller holds
  std::unique_ptr<TaskFileWatcher> taskFileWatcher_;
  std::optional<ControllerSideSettings> pendingControllerSideSettings_;
  double nextTaskFileCheckTime_ = 0.0;
};

/** The driver of `configuration`'s formulation. */
absl::StatusOr<std::unique_ptr<ClosedLoopDriver>> createClosedLoopDriver(const RobotConfiguration& configuration,
                                                                         const ClosedLoopDriverOptions& options);

}  // namespace ocs2::humanoid::validation
