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

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include <humanoid_common_mpc/common/Types.h>
#include <robot_model/ContactEstimatorRegistry.h>
#include <robot_model/RobotState.h>

#include "humanoid_common_mpc_app/robot/FsmStateMailbox.h"
#include "humanoid_common_mpc_app/robot/MujocoViewerAnnotator.h"
#include "humanoid_common_mpc_app/robot/OperatorCommandMailbox.h"
#include "humanoid_common_mpc_app/robot/RealtimeEventLog.h"
#include "humanoid_common_mpc_app/robot/RealtimeLoopRunner.h"
#include "humanoid_common_mpc_app/robot/RobotBackend.h"
#include "humanoid_common_mpc_app/robot/RobotController.h"
#include "humanoid_common_mpc_app/robot/RobotProcessSettings.h"
#include "humanoid_common_mpc_app/robot/SimFallRecovery.h"
#include "humanoid_common_mpc_app/robot/SimFsmBridge.h"
#include "humanoid_common_mpc_app/robot/TaskFileWatcher.h"
#include "humanoid_common_mpc_app/robot/TelemetrySampler.h"
#include "humanoid_common_mpc_app/robot/TelemetrySink.h"
#include "humanoid_mpc_msgs/fsm_state.nproto.h"
#include "humanoid_mpc_msgs/fsm_state.pb.h"
#include "humanoid_mpc_msgs/loop_timing.pb.h"
#include "humanoid_mpc_msgs/robot_state_sample.pb.h"
#include "humanoid_mpc_msgs/viewer_annotations.nproto.h"
#include "robot_ipc/Bus.h"

namespace ocs2::humanoid {

/**
 * The robot process: the realtime loop over a robot backend and an MRT joint controller, and everything around it on
 * the bus. The same for both formulations and every backend; the binaries (humanoid_centroidal_mpc_robot,
 * humanoid_wb_mpc_robot) build the controller and its MPC link and hand them over. See
 * humanoid_nmpc/humanoid_common_mpc_app/README.md and humanoid_nmpc/docs/distributed_runtime/README.md.
 *
 * THE REALTIME THREAD runs, every control period:
 *   1. read the robot state from the backend;
 *   2. take the operator's joint targets into the JOINT_PD posture (mailbox);
 *   3. hand the controller the mode and the posture, and run computeJointControlAction(), which writes the observation
 *      into the MPC link (for a remote link: a triple buffer that the bus's IO thread publishes);
 *   4. tell the viewer the contact state the policy in use plans for now (the contact timeline);
 *   5. every telemetry decimation, copy the cycle into a slot of the telemetry ring;
 *   6. apply a contact estimator or a contact wrench gate the operator or the task file changed (mailbox);
 *   7. the FSM bridge: a dodgeball throw, then one FSM command (mailbox), or the gantry height slider;
 *   8. the fall recovery, and the MPC resets and FSM states it calls for;
 *   9. send the joint action to the backend, unless the torques are off;
 *  10. the MPC's health into the FSM state.
 * The action goes to the backend after the cycle's FSM command and fall recovery (9), not before them as in the ROS
 * sims: a cycle that switches the torques back on hands the backend the action it computed - in ZERO_TORQUE, no gain
 * and no torque - instead of leaving it the action latched when the torques went off, possibly long before and in
 * another mode, for a control period. The simulator also refuses to execute an action applied before its torques came
 * back on (RobotHWInterfaceBase::discardAppliedJointAction()).
 * It talks to nothing but lock-free mailboxes: no bus, no protobuf, no YAML, no file, no lock and no log line (the
 * process's and the controller's reports go to a RealtimeEventLog); the backend's state and action go through
 * RobotHWInterfaceBase's triple buffers. The robot process adds no allocation; the controllers allocate nothing in the
 * passive modes and the holds, and executing a policy still allocates inside the model's accessors and the inverse
 * dynamics (see the package README).
 *
 * A CYCLE THAT THROWS ends the loop (RealtimeLoopRunner): the backend is put in its safe state (enterSafeState(): zero
 * torque), faulted() turns true, and runUntilShutdown() returns the error, so that the binary exits with a failure and
 * the container's restart policy brings the robot back in ZERO_TORQUE.
 *
 * THE COMMUNICATION THREAD is the bus's IO thread. Its handlers fill the mailboxes (OperatorCommandMailbox), and its
 * periodic callbacks empty the realtime thread's: the telemetry ring to the telemetry sinks (robot/state), the FSM state
 * to robot/fsm_state (on change and at 2 Hz), the loop timing to robot/loop_timing (once per reporting window), the
 * event log to the log, and the MPC's viewer annotations to the MuJoCo viewer. It also polls the PD gains file and the
 * task file's controller-side keys.
 *
 * LIFECYCLE. Create() registers everything on the bus, which must not be running; the caller starts the bus, then
 * start(): the backend comes up, the MPC link starts from the robot's state, the backend's threads start and the
 * realtime loop runs, with the robot in ZERO_TORQUE on the gantry. stop() ends the loop and then stops the bus, so no
 * callback outlives the process. The bus, the backend, the controller and the estimator registry must outlive it.
 */
class RobotProcess {
 public:
  struct Config {
    /** [Hz] The control rate (mpc.mrtDesiredFrequency). */
    scalar_t controlFrequency = 500.0;
    /** SCHED_FIFO priority of the realtime thread (--realtime_priority); 0: not realtime. */
    int realtimePriority = 0;
    /** Cores of the realtime thread and of the backend's threads; empty: not pinned. */
    std::vector<int> realtimeCores;
    std::vector<int> backendCores;
    /** The task file's keys (loadRobotProcessSettings()). */
    RobotProcessSettings settings;
    /** The state the robot starts in (createInitialSimState()): the JOINT_PD posture until the operator moves it. */
    std::optional<robot::model::RobotState> initialState;
    /** The task file, watched for its controller-side keys at about 1 Hz; empty: not watched. */
    std::string taskFile;
    /** The joints the fall recovery judges rest on: those of the MPC model. */
    std::vector<size_t> restJointIndices;
    /** The PD gains file is checked every this many control periods (100 centroidal, 500 whole-body, ~1 Hz). */
    size_t pdGainsFileCheckInterval = 100;
    /** The reporting window of robot/loop_timing. */
    std::chrono::nanoseconds loopTimingWindow{std::chrono::seconds(1)};
  };

  /** What the formulation's binary adds; every function runs on the communication thread and may be empty. */
  struct Hooks {
    /** The viewer annotations of the newest policy, when there are new ones (RemoteMpcLink::takeAnnotations()). */
    std::function<bool(msgs::ViewerAnnotations& annotations)> takeViewerAnnotations;
    /** LoopTiming's fields of the MPC link: stale_policies_dropped and policy_age_s. */
    std::function<void(humanoid_mpc_msgs::LoopTiming& loopTiming)> fillLinkStatistics;
  };

  /**
   * Builds the process around `backend` and `controller` and registers it on `bus` (not running yet). Installs the
   * task file's contact estimator (resolved through `contactEstimators`, which already holds the backend's): an unknown
   * name is InvalidArgument listing the available ones. A backend without a simulator is Unimplemented: the FSM bridge
   * and the fall recovery are the simulator's, and a hardware backend needs an FSM bridge of its own.
   */
  static absl::StatusOr<std::unique_ptr<RobotProcess>> Create(robot::ipc::Bus& bus,
                                                              RobotBackend& backend,
                                                              RobotController& controller,
                                                              const robot::model::ContactEstimatorRegistry& contactEstimators,
                                                              Config config,
                                                              Hooks hooks = Hooks());

  ~RobotProcess();
  RobotProcess(const RobotProcess&) = delete;
  RobotProcess& operator=(const RobotProcess&) = delete;

  /**
   * Brings the robot up and starts the realtime loop; the bus must be running. The robot starts in ZERO_TORQUE and the
   * MPC is not waited for: observations stream until the MPC node answers. See LIFECYCLE.
   */
  absl::Status start();

  /** Ends the realtime loop, puts the backend in its safe state, then stops the bus. Idempotent; called by the destructor. */
  void stop();

  /** True once a cycle has thrown and the loop has stopped (see A CYCLE THAT THROWS). Any thread. */
  bool faulted() const { return loop_.faulted(); }
  /** What the cycle threw, once faulted(). */
  std::string faultMessage() const { return loop_.faultMessage(); }

  /**
   * The main thread's wait: polls `shutdownRequested` (the binaries' SIGINT/SIGTERM flag) and faulted() every 50 ms, then
   * stops the process. OK on a shutdown, Internal with the cycle's exception once the loop has faulted.
   */
  absl::Status runUntilShutdown(const std::function<bool()>& shutdownRequested);

  /** Cycles the realtime loop has run. Any thread. */
  std::uint64_t cycles() const { return loop_.cycles(); }
  const RealtimeLoopRunner& loop() const { return loop_; }
  OperatorCommandMailbox& mailbox() { return *mailbox_; }
  const TelemetrySampler* telemetrySampler() const { return sampler_.get(); }
  /** [s] The control period. */
  scalar_t controlPeriod() const { return 1.0 / config_.controlFrequency; }

 private:
  RobotProcess(robot::ipc::Bus& bus, RobotBackend& backend, RobotController& controller, Config config, Hooks hooks);

  absl::Status registerOnBus();

  // ---- The realtime thread.
  void cycle();
  void applyControllerSettings();
  /** The loop's fault handler: the backend's safe state. */
  void onCycleFault();

  // ---- The communication thread.
  void pumpRealtimeMailboxes();
  void publishFsmState(bool republish);
  void publishLoopTiming(const robot::realtime::LoopTimingSnapshot& snapshot);
  void onTaskFileChanged(const std::string& file);

  robot::ipc::Bus& bus_;
  RobotBackend& backend_;
  robot::mujoco_sim_interface::MujocoSimInterface* const simulator_;
  RobotController& controller_;
  const Config config_;
  const Hooks hooks_;

  RealtimeEventLog eventLog_;
  FsmStateMailbox fsmStates_;
  std::unique_ptr<OperatorCommandMailbox> mailbox_;
  std::unique_ptr<SimFsmBridge> fsmBridge_;
  std::unique_ptr<SimFallRecovery> fallRecovery_;
  std::unique_ptr<TelemetrySampler> sampler_;
  std::vector<std::unique_ptr<TelemetrySink>> sinks_;
  std::unique_ptr<MujocoViewerAnnotator> annotator_;
  std::unique_ptr<TaskFileWatcher> taskFileWatcher_;
  RealtimeLoopRunner loop_;

  // ---- Realtime thread state, sized before the loop starts.
  std::string currentMode_;
  std::string contactEstimatorName_;
  std::vector<bool> plannedContactFlags_;
  const std::vector<bool> noContactFlags_;
  std::array<vector3_t, N_CONTACTS> measuredContactForces_;

  // ---- Communication thread state.
  msgs::FsmState fsmState_;
  humanoid_mpc_msgs::FsmState fsmStateMessage_;
  bool haveFsmState_ = false;
  std::chrono::steady_clock::time_point lastFsmStatePublish_;
  humanoid_mpc_msgs::RobotStateSample telemetryMessage_;
  humanoid_mpc_msgs::LoopTiming loopTimingMessage_;
  msgs::ViewerAnnotations annotations_;
  robot::realtime::LoopTimingSnapshot timingSnapshot_;

  bool started_ = false;
  bool stopped_ = false;
};

}  // namespace ocs2::humanoid
