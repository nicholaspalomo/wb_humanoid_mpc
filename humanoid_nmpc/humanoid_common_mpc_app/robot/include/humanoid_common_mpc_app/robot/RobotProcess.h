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
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc_app/robot/ConfigFileStore.h"
#include "humanoid_common_mpc_app/robot/FallRecovery.h"
#include "humanoid_common_mpc_app/robot/FsmBridge.h"
#include "humanoid_common_mpc_app/robot/FsmStateMailbox.h"
#include "humanoid_common_mpc_app/robot/MujocoViewerAnnotator.h"
#include "humanoid_common_mpc_app/robot/OperatorCommandMailbox.h"
#include "humanoid_common_mpc_app/robot/RealtimeEventLog.h"
#include "humanoid_common_mpc_app/robot/RealtimeLoopRunner.h"
#include "humanoid_common_mpc_app/robot/RobotBackend.h"
#include "humanoid_common_mpc_app/robot/RobotController.h"
#include "humanoid_common_mpc_app/robot/RobotProcessSettings.h"
#include "humanoid_common_mpc_app/robot/TaskFileWatcher.h"
#include "humanoid_common_mpc_app/robot/TelemetrySampler.h"
#include "humanoid_common_mpc_app/robot/TelemetrySink.h"
#include "humanoid_mpc_msgs/fsm_state.nproto.h"
#include "humanoid_mpc_msgs/fsm_state.pb.h"
#include "humanoid_mpc_msgs/loop_timing.pb.h"
#include "humanoid_mpc_msgs/robot_state_sample.pb.h"
#include "humanoid_mpc_msgs/viewer_annotations.nproto.h"
#include "humanoid_state_estimator/StateEstimator.h"
#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "robot_ipc/Bus.h"
#include "robot_model/ContactEstimatorRegistry.h"
#include "robot_model/RobotState.h"

namespace ocs2::humanoid {

/**
 * The robot process: the realtime loop over a robot backend and an MRT joint
 * controller, and everything around it on the bus. The same for both
 * formulations and every backend; the binaries (humanoid_centroidal_mpc_robot,
 * humanoid_wb_mpc_robot) build the controller and its MPC link and hand them
 * over. See humanoid_nmpc/humanoid_common_mpc_app/robot/README.md and
 * humanoid_nmpc/docs/distributed_runtime/README.md.
 *
 * THE BACKEND'S SUPERVISOR. The FSM bridge and the fall recovery are the ones
 * the backend names (RobotBackend::supervisorName(), RobotSupervisorRegistry):
 * `sim` drives the simulator's torque switch, gantry, dodgeballs and resets
 * (SimFsmBridge, SimFallRecovery); `hardware` the backend's torque switch only,
 * with no gantry and a tilt cut-off (HardwareFsmBridge, HardwareFallRecovery).
 * The viewer's contact timeline and annotations exist only on a simulator
 * (RobotBackend::simulator()).
 *
 * THE REALTIME THREAD runs, every control period:
 *   1. read the robot state from the backend, and run the task file's state
 * estimator over it (state_estimator: `ground_truth` hands the backend's state
 * on as it is, `kalman` replaces the base's position and linear velocity with
 * its estimate from the IMU and the leg kinematics): what every later step
 * reads;
 *   2. take the operator's joint targets into the JOINT_PD posture (mailbox);
 *   3. hand the controller the mode and the posture, and run
 * computeJointControlAction(), which writes the observation into the MPC link
 * (for a remote link: a triple buffer that the bus's IO thread publishes);
 *   4. tell the viewer the contact state the policy in use plans for now (the
 * contact timeline);
 *   5. every telemetry decimation, copy the cycle into a slot of the telemetry
 * ring;
 *   6. apply a contact estimator or a contact wrench gate the operator or the
 * task file changed (mailbox);
 *   7. the FSM bridge: a dodgeball throw, then one FSM command (mailbox), or
 * the gantry height slider;
 *   8. the fall recovery, and the MPC resets and FSM states it calls for; a
 * base the backend relocated (the simulator's reset) restarts the state
 * estimator from the backend's state;
 *   9. send the joint action to the backend, unless the torques are off;
 *  10. the MPC's health into the FSM state.
 * The action goes to the backend after the cycle's FSM command and fall
 * recovery (9), not before them: a cycle that switches the torques back on
 * hands the backend the action it computed - in ZERO_TORQUE, no gain and no
 * torque - instead of leaving it the action latched when the torques went off,
 * possibly long before and in another mode, for a control period. The simulator
 * also refuses to execute an action applied before its torques came back on
 * (RobotHWInterfaceBase::discardAppliedJointAction()). It talks to nothing but
 * lock-free mailboxes: no bus, no protobuf, no configuration parsing, no file,
 * no lock and no log line (the process's and the controller's reports go to a
 * RealtimeEventLog); the backend's state and action go through
 * RobotHWInterfaceBase's triple buffers. The robot process adds no allocation;
 * the controllers allocate nothing in the passive modes and the holds, and
 * executing a policy still allocates inside the model's accessors and the
 * inverse dynamics (see the package README). The state estimators allocate
 * nothing either.
 *
 * A CYCLE THAT THROWS ends the loop (RealtimeLoopRunner): the backend is put in
 * its safe state (enterSafeState(): zero torque), faulted() turns true, and
 * runUntilShutdown() returns the error, so that the binary exits with a failure
 * and the container's restart policy brings the robot back in ZERO_TORQUE.
 *
 * THE COMMUNICATION THREAD is the bus's IO thread. Its handlers fill the
 * mailboxes (OperatorCommandMailbox), and its periodic callbacks empty the
 * realtime thread's: the telemetry ring to the telemetry sinks (robot/state),
 * the FSM state to robot/fsm_state (on change and at 2 Hz), the loop timing to
 * robot/loop_timing (once per reporting window), the event log to the log, and
 * the MPC's viewer annotations to the MuJoCo viewer. It also polls the PD gains
 * file and the task file's controller-side keys, and takes the GUI's saves of
 * operator/config_save, which the configuration store's writer thread checks
 * and writes into the robot's persistent copies (ConfigFileStore,
 * Config::configStore); the watchers then apply a saved task or PD gains file
 * as they apply an edit.
 *
 * LIFECYCLE. Create() registers everything on the bus, which must not be
 * running; the caller starts the bus, then start(): the backend comes up, the
 * MPC link starts from the robot's state, the backend's threads start and the
 * realtime loop runs, with the robot in ZERO_TORQUE on the gantry, and the
 * configuration store's writer thread. stop() ends the loop, then the writer,
 * then the bus, so no callback outlives the process. The bus, the backend, the
 * controller and the estimator registry must outlive it.
 */
class RobotProcess {
 public:
  struct Config {
    /** The defaults of controlFrequency [Hz] and of pdGainsFileCheckInterval.
     */
    static constexpr scalar_t kDefaultControlFrequency = 500.0;
    static constexpr size_t kDefaultPdGainsFileCheckInterval = 100;

    /** [Hz] The control rate (mpc.mrt_desired_frequency). */
    scalar_t controlFrequency = kDefaultControlFrequency;
    /** SCHED_FIFO priority of the realtime thread (--realtime_priority); 0: not
     * realtime. */
    int realtimePriority = 0;
    /** Cores of the realtime thread and of the backend's threads; empty: not
     * pinned. */
    std::vector<int> realtimeCores;
    std::vector<int> backendCores;
    /** The task file's keys (loadRobotProcessSettings()). */
    RobotProcessSettings settings;
    /** The state the robot starts in (createInitialSimState()): the JOINT_PD
     * posture until the operator moves it. */
    std::optional<robot::model::RobotState> initialState;
    /** The task file, watched for its controller-side keys at about 1 Hz;
     * empty: not watched. */
    std::string taskFile;
    /**
     * The write time `taskFile` had when `settings` were read from it
     * (TaskFileWatcher::writeTimeOf(), taken before the read): the watcher
     * starts from it, so that a save while the process starts is applied by its
     * first poll. nullopt: the file's time at Create().
     */
    std::optional<std::filesystem::file_time_type> taskFileReadAt;
    /**
     * The running robot (ModelSettings::robotName): the operator mailbox
     * refuses an MPC parameter update of another robot
     * (OperatorCommandMailbox::Config::robotName). Empty: not checked.
     */
    std::string robotName;
    /**
     * The running configuration: configFileIdentity() of the bundled task file
     * (RobotConfigDirectory::identity()). The operator mailbox refuses an MPC
     * parameter update of another configuration
     * (OperatorCommandMailbox::Config::taskFileIdentity). Empty: not checked.
     */
    std::string taskFileIdentity;
    /**
     * The running PD gains configuration: configFileIdentity() of the bundled
     * joint PD gains file (RobotConfigDirectory::identity(kJointPdGains)). The
     * operator mailbox refuses a joint PD gains update of another configuration
     * (OperatorCommandMailbox::Config::pdGainsFileIdentity). Empty: not
     * checked.
     */
    std::string pdGainsFileIdentity;
    /**
     * The persistent copies a save of operator/config_save is written to
     * (configFileStoreConfig() of the robot's RobotConfigDirectory). The
     * default has none: every save is answered NOT_STORED.
     */
    ConfigFileStore::Config configStore;
    /** The joints the fall recovery judges rest on: those of the MPC model. */
    std::vector<size_t> restJointIndices;
    /**
     * The robot's URDF and the MPC's contact frames, in contact order
     * (model_settings): the leg kinematics of the `kalman` state estimator,
     * with the contact frames' parent joints, whose frames stand in for a
     * contact frame the URDF does not have (the MPC builds those from the task
     * file's contact rectangles). Unused by `ground_truth`.
     */
    std::string urdfFile;
    std::vector<std::string> contactFrameNames;
    std::vector<std::string> contactParentJointNames;
    /** The PD gains file is checked every this many control periods (100
     * centroidal, 500 whole-body, ~1 Hz). */
    size_t pdGainsFileCheckInterval = kDefaultPdGainsFileCheckInterval;
    /** The reporting window of robot/loop_timing. */
    std::chrono::nanoseconds loopTimingWindow{std::chrono::seconds(1)};
  };

  /** What the formulation's binary adds; every function runs on the
   * communication thread and may be empty. */
  struct Hooks {
    /** The viewer annotations of the newest policy, when there are new ones
     * (RemoteMpcLink::takeAnnotations()). */
    std::function<bool(msgs::ViewerAnnotations& annotations)>
        takeViewerAnnotations;
    /** LoopTiming's fields of the MPC link: stale_policies_dropped and
     * policy_age_s. */
    std::function<void(humanoid_mpc_msgs::LoopTiming& loopTiming)>
        fillLinkStatistics;
    /**
     * The configuration store's check of a save
     * (ConfigFileStore::Hooks::validateConfigFile): the binaries bind it to
     * checkConfigFileCandidate() on the stored files with the running process's
     * estimator registry. Runs on the store's writer thread, not the
     * communication thread. Empty: a text that parses is stored.
     */
    std::function<absl::Status(const ConfigFileCandidate& candidate)>
        validateConfigFile;
  };

  /**
   * Builds the process around `backend` and `controller` and registers it on
   * `bus` (not running yet). Installs the task file's contact estimator
   * (resolved through `contactEstimators`, which already holds the backend's)
   * and its state estimator (StateEstimatorRegistry), and the supervisor the
   * backend names (RobotSupervisorRegistry): an unknown name is InvalidArgument
   * listing the available ones, and so is the `sim` supervisor of a backend
   * that is no simulator.
   */
  static absl::StatusOr<std::unique_ptr<RobotProcess>> Create(
      robot::ipc::Bus* absl_nonnull bus,
      RobotBackend* absl_nonnull backend,
      RobotController* absl_nonnull controller,
      const robot::model::ContactEstimatorRegistry* absl_nonnull
      contactEstimators,
      Config config,
      Hooks hooks = Hooks());

  ~RobotProcess();
  RobotProcess(const RobotProcess&) = delete;
  RobotProcess& operator=(const RobotProcess&) = delete;

  /**
   * Brings the robot up and starts the realtime loop; the bus must be running.
   * The robot starts in ZERO_TORQUE and the MPC is not waited for: observations
   * stream until the MPC node answers. See LIFECYCLE.
   */
  absl::Status start();

  /** Ends the realtime loop, puts the backend in its safe state, then stops the
   * bus. Idempotent; called by the destructor. */
  void stop();

  /** True once a cycle has thrown and the loop has stopped (see A CYCLE THAT
   * THROWS). Any thread. */
  bool faulted() const { return loop_.faulted(); }
  /** What the cycle threw, once faulted(). */
  std::string faultMessage() const { return loop_.faultMessage(); }

  /**
   * The main thread's wait: polls `shutdownRequested` (the binaries'
   * SIGINT/SIGTERM flag) and faulted() every 50 ms, then stops the process. OK
   * on a shutdown, Internal with the cycle's exception once the loop has
   * faulted.
   */
  absl::Status runUntilShutdown(absl::FunctionRef<bool()> shutdownRequested);

  /**
   * runUntilShutdown(), which also calls `onRunning` once, on this thread, when
   * the realtime loop has run for `runningFor` without a fault: the binaries
   * confirm their boot with it (RobotConfigDirectory::confirmBoot()).
   */
  absl::Status runUntilShutdown(absl::FunctionRef<bool()> shutdownRequested,
                                absl::Duration runningFor,
                                absl::FunctionRef<void()> onRunning);

  /** Cycles the realtime loop has run. Any thread. */
  uint64_t cycles() const { return loop_.cycles(); }
  const RealtimeLoopRunner& loop() const { return loop_; }
  OperatorCommandMailbox& mailbox() { return *mailbox_; }
  ConfigFileStore& configStore() { return *configStore_; }
  /** [s] The control period. */
  scalar_t controlPeriod() const { return 1.0 / config_.controlFrequency; }

 private:
  RobotProcess(robot::ipc::Bus* absl_nonnull bus,
               RobotBackend* absl_nonnull backend,
               RobotController* absl_nonnull controller,
               Config config,
               Hooks hooks);

  absl::Status registerOnBus();

  // ---- The realtime thread.
  void cycle();
  void applyControllerSettings();
  /** The loop's fault handler: the backend's safe state. */
  void onCycleFault();

  // ---- The communication thread.
  void pumpRealtimeMailboxes();
  /** Publishes the FSM state the realtime thread wrote, when it is new or
   * kFsmStateRepublishPeriod has passed. */
  void publishFsmState();
  void publishLoopTiming(const robot::realtime::LoopTimingSnapshot& snapshot);
  // Applies the controller-side settings of the changed task `file`, but for a
  // file written before the newest operator/mpc_parameters arrived
  // (OperatorCommandMailbox::lastMpcParametersArrival()), an older edit.
  void onTaskFileChanged(const std::string& file);

  robot::ipc::Bus* absl_nonnull bus_;
  RobotBackend* absl_nonnull backend_;
  /** backend_->simulator(): the viewer's contact timeline and annotations;
   * nullptr on a backend without one. */
  robot::mujoco_sim_interface::MujocoSimInterface* absl_nullable const
      simulator_;
  RobotController* absl_nonnull controller_;
  const Config config_;
  const Hooks hooks_;

  RealtimeEventLog eventLog_;
  FsmStateMailbox fsmStates_;
  std::unique_ptr<OperatorCommandMailbox> mailbox_;
  std::unique_ptr<ConfigFileStore> configStore_;
  std::unique_ptr<FsmBridge> fsmBridge_;
  std::unique_ptr<FallRecovery> fallRecovery_;
  std::unique_ptr<estimation::StateEstimator> stateEstimator_;
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
  std::array<vector3_t, kNumContacts> measuredContactForces_;

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
