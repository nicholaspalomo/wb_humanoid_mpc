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
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

#include "humanoid_common_mpc/config/robot/ControllerSideSettingsFromConfig.h"
#include "humanoid_common_mpc/contact/ContactWrenchGate.h"
#include "humanoid_common_mpc_app/robot/FsmCommand.h"
#include "humanoid_mpc_config/joint_pd_gains_file.nproto.h"
#include "humanoid_mpc_config/joint_pd_gains_file.pb.h"
#include "humanoid_mpc_config/mpc_parameter_update.pb.h"
#include "humanoid_mpc_msgs/dodgeball_throw.pb.h"
#include "humanoid_mpc_msgs/fsm_command.pb.h"
#include "humanoid_mpc_msgs/joint_targets.pb.h"
#include "humanoid_mpc_msgs/walking_velocity_command.pb.h"
#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "robot_core/TripleBuffer.h"
#include "robot_ipc/Bus.h"
#include "robot_model/ContactEstimator.h"
#include "robot_model/ContactEstimatorRegistry.h"
#include "robot_realtime/SpscQueue.h"

namespace ocs2::humanoid {

/**
 * A contact estimator and a contact wrench gate for the MRT joint controller, as the realtime thread takes them from
 * the mailbox: the estimator already built (and kept alive by the mailbox, so that the one it replaces is never freed
 * on the realtime thread), the name already canonical.
 */
struct ControllerSettingsUpdate {
  bool hasContactEstimator = false;
  std::string contactEstimatorName;
  std::shared_ptr<robot::model::ContactEstimator> contactEstimator;
  bool hasContactWrenchGate = false;
  ContactWrenchGate::Config contactWrenchGate;
};

/**
 * The operator's commands on their way to the realtime thread of the robot process. The handlers run on the bus's IO
 * thread (registerOnBus()); every conversion, every name lookup and every log line happens there. The realtime thread
 * only takes what they leave, without a lock and without allocating:
 *
 *   operator/fsm_command             FsmCommand      -> parseFsmCommand()         -> SpscQueue<FsmCommandEvent>, in order
 *   operator/joint_targets           JointTargets    -> merged into the posture   -> TripleBuffer of the whole posture
 *   operator/dodgeball_throw         DodgeballThrow  -> dodgeballThrowFromMessage() -> SpscQueue<DodgeballThrow>
 *   operator/walking_velocity_command  desired_pelvis_height, clamped              -> atomic gantry height
 *   operator/mpc_parameters          MpcParameterUpdate (humanoid_mpc_config) -> controllerSideSettingsFromConfig() of
 *                                                       its task file, estimator built -> SpscQueue<ControllerSettingsUpdate>
 *   operator/pd_gains                JointPdGainsFile (humanoid_mpc_config) -> the controller's setPdGains() (it
 *                                                       resolves the gains on this thread and hands them to the control
 *                                                       thread in its own mailbox)
 *
 * The same handlers are public methods, for the tests and for the task-file watcher (postControllerSettings()).
 *
 * FSM commands carry a sequence number (FsmCommand.sequence); a message with the sequence of the command accepted
 * last is a repeat and is dropped. A command name parseFsmCommand() does not know is logged and dropped, as are joint
 * targets that are not finite, a throw dodgeballThrowFromMessage() refuses and an unknown contact estimator.
 *
 * The three payloads of the tuning GUI are checked before anything of them is applied (OperatorPayloadChecks.h): a
 * DodgeballThrow, a JointPdGainsFile or an MpcParameterUpdate with a field this build's schema does not know, an
 * MpcParameterUpdate of another schema fingerprint, one whose task file is another robot's (Config::robotName) and one of
 * another configuration (Config::taskFileIdentity) are refused whole, logged and counted: a GUI built from another
 * version, or pointed at another robot or configuration, applies nothing.
 */
class OperatorCommandMailbox {
 public:
  struct Config {
    /** The robot's joints, by joint index (jointNamesByIndex()). */
    std::vector<std::string> jointNames;
    /** The JOINT_PD posture at start-up, by joint index: the operator's joint targets are merged into it. */
    std::vector<double> initialNominalPositions;
    /**
     * The running robot (ModelSettings::robotName, the task file's model_settings.robot_name): an operator/mpc_parameters
     * update whose task file names another robot is refused. Empty: not checked.
     */
    std::string robotName;
    /**
     * The running configuration (configFileIdentity() of the bundled task file, RobotConfigDirectory::identity()): an
     * operator/mpc_parameters update whose config_path is another is refused before its controller-side settings are
     * taken, so that a GUI opened on another configuration of the robot (the centroidal G1's against the whole-body G1,
     * whose robot_name is the same) changes nothing here. Empty: not checked.
     */
    std::string taskFileIdentity;
    /** Capacity of the FSM command and the controller settings queues; a burst beyond it is dropped and counted. */
    size_t commandQueueCapacity = 16;
    size_t dodgeballQueueCapacity = 4;
  };

  /**
   * What the robot process plugs into the mailbox: the consumer of operator/pd_gains, which the mailbox subscribes
   * because the bus takes one subscription per topic, and the check of every contact estimator the mailbox builds.
   */
  struct Hooks {
    /** operator/pd_gains, the controller's setPdGains(). IO thread. */
    std::function<absl::Status(const mpc_config::JointPdGainsFile& gains)> pdGains;
    /**
     * Whether a contact estimator the mailbox has just built suits the controller (the robot process's:
     * checkContactEstimator() on its initial state). Called once per estimator, on the thread that resolves its name,
     * before the realtime thread can use it; an error refuses the estimator. Unset: every estimator is accepted.
     */
    std::function<absl::Status(robot::model::ContactEstimator& estimator, absl::string_view name)> checkContactEstimator;
  };

  /** What the handlers did; every counter only grows. Any thread. */
  struct Statistics {
    uint64_t fsmCommandsQueued = 0;
    uint64_t fsmCommandsRepeated = 0;
    uint64_t fsmCommandsUnknown = 0;
    uint64_t jointTargetsApplied = 0;
    uint64_t jointTargetsRejected = 0;
    uint64_t dodgeballsQueued = 0;
    /** Refused by dodgeballThrowFromMessage(), or of another schema version. */
    uint64_t dodgeballsRejected = 0;
    uint64_t controllerSettingsQueued = 0;
    /**
     * An unknown or refused contact estimator, and an operator/mpc_parameters message refused whole: one that does not
     * convert, is of another schema version, of another robot or of another configuration.
     */
    uint64_t controllerSettingsRejected = 0;
    /** operator/pd_gains messages received. */
    uint64_t pdGainsDocuments = 0;
    /** operator/pd_gains messages refused: of another schema version, not converting, or refused by the controller. */
    uint64_t pdGainsRejected = 0;
    /** Dropped because the realtime thread had not taken the previous ones yet. */
    uint64_t queueDrops = 0;
  };

  /**
   * `contactEstimators` resolves the names of the controller-side settings; it must outlive the mailbox and already
   * hold every estimator (the backend's included). InvalidArgument when the initial posture does not match the joints.
   */
  static absl::StatusOr<std::unique_ptr<OperatorCommandMailbox>> Create(Config config,
                                                                        const robot::model::ContactEstimatorRegistry& contactEstimators,
                                                                        Hooks hooks = Hooks());

  OperatorCommandMailbox(const OperatorCommandMailbox&) = delete;
  OperatorCommandMailbox& operator=(const OperatorCommandMailbox&) = delete;
  ~OperatorCommandMailbox() = default;

  /** Subscribes the six topics on `bus`, which must not be running yet. */
  absl::Status registerOnBus(robot::ipc::Bus& bus);

  // ------------------------------------------------------------------ the realtime thread

  /** The oldest FSM command not taken yet. */
  bool takeFsmCommand(FsmCommandEvent& command);
  /** The newest dodgeball throw; older ones still queued are dropped with it (the button is a one-shot). */
  bool takeDodgeballThrow(robot::mujoco_sim_interface::MujocoSimInterface::DodgeballThrow& throwCommand);
  /** The JOINT_PD posture with every joint target received so far, when it changed since the last call. */
  bool takeNominalPosture(std::vector<double>& nominalPositions);
  /** The gantry height the operator's slider asks for [m], once one walking command has been received. */
  std::optional<double> desiredGantryHeight() const;
  /** Calls `apply` with the oldest controller settings update not taken yet; false when there is none. */
  template <typename Apply>
  bool takeControllerSettings(Apply&& apply) {
    return controllerSettings_.tryPopInPlace(std::forward<Apply>(apply));
  }

  // ------------------------------------------------------------------ the IO thread (and tests)

  void onFsmCommand(const humanoid_mpc_msgs::FsmCommand& message);
  void onJointTargets(const humanoid_mpc_msgs::JointTargets& message);
  void onDodgeballThrow(const humanoid_mpc_msgs::DodgeballThrow& message);
  void onWalkingVelocityCommand(const humanoid_mpc_msgs::WalkingVelocityCommand& message);
  void onMpcParameters(const humanoid_mpc_config::MpcParameterUpdate& message);
  void onPdGains(const humanoid_mpc_config::JointPdGainsFile& message);

  /**
   * Resolves the estimator `settings` names and queues the update for the realtime thread (the task-file watcher and
   * onMpcParameters()), treating the settings as those of a whole task file (humanoid_nmpc/humanoid_mpc_config/README.md, "Live updates"):
   * the file's contact estimator - the schema's default when the file names none - and its contact wrench gate, the instantaneous gate
   * (ContactWrenchGate::Config{}) when the file has no contact_wrench_gate block. An unknown estimator name is logged with the available
   * ones and leaves the estimator as it is; the gate still goes through. A gate controllerSideSettingsFromConfig() refused (one of
   * `settings.problems`) is logged and leaves the gate as it is. `source` names the file or the topic for the log.
   */
  void postControllerSettings(const ControllerSideConfig& settings, absl::string_view source);

  /**
   * The estimator `name` names, built once and kept for the life of the mailbox, so that an estimator the realtime
   * thread replaces is never freed there. NotFound listing the available names for an unknown one; the error of
   * Hooks::checkContactEstimator for one that does not suit the controller, which is then not kept. Thread-safe; call it
   * for the controller's initial estimator too.
   */
  absl::StatusOr<std::shared_ptr<robot::model::ContactEstimator>> contactEstimator(const std::string& name);

  Statistics statistics() const;

 private:
  OperatorCommandMailbox(Config config, const robot::model::ContactEstimatorRegistry& contactEstimators, Hooks hooks);

  const Config config_;
  const robot::model::ContactEstimatorRegistry& contactEstimatorRegistry_;
  const Hooks hooks_;

  robot::realtime::SpscQueue<FsmCommandEvent> fsmCommands_;
  robot::realtime::SpscQueue<robot::mujoco_sim_interface::MujocoSimInterface::DodgeballThrow> dodgeballs_;
  robot::realtime::SpscQueue<ControllerSettingsUpdate> controllerSettings_;
  robot::TripleBuffer<std::vector<double>> nominalPosture_;
  /** [m] NaN until a walking command has arrived. */
  std::atomic<double> desiredGantryHeight_;

  // ---- IO thread.
  std::vector<double> ioNominalPositions_;
  absl::flat_hash_map<std::string, size_t> jointIndexByName_;
  uint64_t lastFsmSequence_ = 0;
  bool haveFsmSequence_ = false;

  absl::Mutex estimatorsMutex_;
  absl::flat_hash_map<std::string, std::shared_ptr<robot::model::ContactEstimator>> estimators_ ABSL_GUARDED_BY(estimatorsMutex_);

  std::atomic<uint64_t> fsmCommandsQueued_{0};
  std::atomic<uint64_t> fsmCommandsRepeated_{0};
  std::atomic<uint64_t> fsmCommandsUnknown_{0};
  std::atomic<uint64_t> jointTargetsApplied_{0};
  std::atomic<uint64_t> jointTargetsRejected_{0};
  std::atomic<uint64_t> dodgeballsQueued_{0};
  std::atomic<uint64_t> dodgeballsRejected_{0};
  std::atomic<uint64_t> controllerSettingsQueued_{0};
  std::atomic<uint64_t> controllerSettingsRejected_{0};
  std::atomic<uint64_t> pdGainsDocuments_{0};
  std::atomic<uint64_t> pdGainsRejected_{0};
};

}  // namespace ocs2::humanoid
