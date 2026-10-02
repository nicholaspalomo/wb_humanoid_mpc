/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.

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

#include <ocs2_mpc/MPC_BASE.h>

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"

#include <robot_model/ContactEstimator.h>
#include <robot_model/ControllerBase.h>
#include "humanoid_common_mpc/contact/ContactWrenchGate.h"
#include "humanoid_common_mpc/mrt/ControllerEventSink.h"
#include "humanoid_common_mpc/mrt/JointPdGains.h"
#include "humanoid_common_mpc/mrt/JointPdGainsMailbox.h"
#include "humanoid_common_mpc/mrt/MpcLink.h"
#include "humanoid_common_mpc/mrt/MpcResetSupervisor.h"
#include "humanoid_common_mpc/mrt/SafetyDecay.h"
#include "humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h"
#include "humanoid_mpc_ipc/RealtimePolicyEvaluator.h"
#include "robot_model/RobotDescription.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

namespace ocs2::humanoid {

class CentroidalMpcMrtJointController final : public ::robot::model::ControlBase {
 public:
  /**
   * Constructor of a controller whose MPC runs in this process (InProcessMpcLink over `mpc`).
   *
   * @param [in] mpc: The underlying MPC class to be used; it must outlive the controller.
   * @param [in] mpcDesiredFrequency: The max frequency to run the mpc at; <= 0 runs the solves back to back.
   * @param [in] pdGainsFile: joint_pd_gains.yaml; missing or empty: the default gains. Reloaded by pollPdGainsFile().
   *        A file that exists but cannot be read, or that parseJointPdGainsYaml() refuses, throws std::invalid_argument
   *        naming the file and what is wrong with it: the controller does not start on gains nobody wrote.
   */
  CentroidalMpcMrtJointController(const ::robot::model::RobotDescription& robotDescription,
                                  const ModelSettings& modelSettings,
                                  const CentroidalMpcRobotModel<scalar_t>& mpcRobotModel,
                                  MPC_BASE& mpc,
                                  PinocchioInterface pinocchioInterface,
                                  scalar_t mpcDesiredFrequency = -1,
                                  const std::string& pdGainsFile = "",
                                  const MpcRobotModelBase<scalar_t>* effectiveMpcRobotModel = nullptr);

  /**
   * Constructor of a controller that reaches its MPC through the link `mpcLinkFactory` makes (MpcLink.h), for instance a
   * remote one. The factory is called once, here, with this controller's reset target.
   */
  CentroidalMpcMrtJointController(const ::robot::model::RobotDescription& robotDescription,
                                  const ModelSettings& modelSettings,
                                  const CentroidalMpcRobotModel<scalar_t>& mpcRobotModel,
                                  const MpcLinkFactory& mpcLinkFactory,
                                  PinocchioInterface pinocchioInterface,
                                  const std::string& pdGainsFile = "",
                                  const MpcRobotModelBase<scalar_t>* effectiveMpcRobotModel = nullptr);

  /**
   * Destructor. Stops the link (for InProcessMpcLink: the solver thread) and waits for it.
   */
  ~CentroidalMpcMrtJointController();

  bool ready() {
    mpcLink_->updatePolicy();
    return mpcLink_->initialPolicyReceived();
  }
  bool ready() const { return mpcLink_->initialPolicyReceived(); }

  /**
   * Handles the low level controller loop that updates the mpc observation, reads out the latest policy and sets the joint control action.
   *
   * In the passive modes (ZERO_TORQUE, JOINT_PD, GRAVITY_COMP, SAFETY) and the hold of WB_MPC its own code allocates
   * nothing once the controller is built, takes no lock and writes no log line (reports go to the event sink); Pinocchio's
   * passes over the composite base joint make one temporary each. Executing a policy in WB_MPC still allocates inside
   * the model's accessors and the inverse dynamics (humanoid_common_mpc_app/robot/README.md).
   */

  void computeJointControlAction(scalar_t time,
                                 const ::robot::model::RobotState& robotState,
                                 ::robot::model::RobotJointAction& robotJointAction) override;

  /** Starts the MPC link (MpcLink::start()) from the observation of `initRobotState`. Call it once, before the control loop. */
  void startMpcThread(const ::robot::model::RobotState& initRobotState);

  /** The link the controller reaches its MPC through. */
  MpcLink& getMpcLink() { return *mpcLink_; }
  const MpcLink& getMpcLink() const { return *mpcLink_; }

  /**
   * Hands the controller the joint PD gains of a joint_pd_gains.yaml document (as the tuning GUI publishes it), without
   * writing the file. Thread-safe, for any non-realtime thread: the text is parsed on the caller's thread into a
   * preallocated set of gains, which the control thread takes into use at its next cycle without locking or
   * allocating (JointPdGainsMailbox). A document parseJointPdGainsYaml() refuses is returned as its InvalidArgument and
   * changes nothing. Of documents handed in concurrently (here, or by pollPdGainsFile()), the one handed in last wins,
   * whichever is parsed first.
   */
  absl::Status setPdGainsYaml(absl::string_view yamlText);

  /**
   * Reloads the PD gains file of the constructor when it has been written since it was last read; the file watcher of
   * the gains, for a non-realtime thread to call at the rate it should be checked. Like setPdGainsYaml(): a file that
   * does not parse is reported and leaves the gains as they are. Thread-safe.
   */
  void pollPdGainsFile();

  /** The gains the control thread commands. Control thread. */
  const JointPdGains& getPdGains() const { return pdGains_; }

  /**
   * @brief Set the active control mode. The passive modes (ZERO_TORQUE, JOINT_PD, GRAVITY_COMP, SAFETY) compute their
   *        action here, independent of the MPC; WB_MPC (or MPC_ACTIVE) executes the MPC policy.
   *
   * Entering WB_MPC from a passive mode resets the MPC from the observation at that moment (requestMpcResetAndHold()):
   * the controller keeps the action of the mode it came from (JOINT_PD for ZERO_TORQUE, which holds nothing) until a
   * policy solved after that reset is active, then ramps into it over mpcEntryBlendTime (0: at once). No policy solved
   * before the entry reaches the robot. Call it from the thread that runs computeJointControlAction().
   */
  void setControlMode(absl::string_view mode);
  const std::string& getControlMode() const { return controlMode_; }

  /**
   * @brief Requests a reset of the MPC, served by the solver thread before its next solve: the solver, the reference
   *        manager and every synchronized module go back to their state after construction (MPC_BASE::reset()), the
   *        policy waiting in the MRT buffer is dropped, and the MPC restarts from the observation current when the
   *        reset is served. Callable from any thread.
   *
   * The policy in use is executed until the first policy solved after the reset replaces it, as when the gantry is
   * unlocked under a robot standing in WB_MPC. Where nothing solved before may reach the robot, use
   * requestMpcResetAndHold().
   */
  void requestMpcReset() {
    // The policy in use no longer counts as solved after every reset requested (getPlannedContactFlags()).
    policyActivated_.store(false);
    mpcLink_->requestReset();
  }

  /**
   * @brief requestMpcReset(), and WB_MPC holds the robot until a policy solved after the reset is active: with the
   *        GRAVITY_COMP action when `holdGravityComp`, with the JOINT_PD action otherwise, then it ramps into the MPC
   *        action over mpcEntryBlendTime. For discontinuities of the plant - a fall caught on the gantry, a reset of
   *        the simulator, the clock going backwards - and for entering WB_MPC. Call it from the thread that runs
   *        computeJointControlAction().
   */
  void requestMpcResetAndHold(bool holdGravityComp = false);

  /**
   * False while the MPC solver keeps failing (MpcResetSupervisor): WB_MPC then holds the robot with the JOINT_PD
   * action, and the solver retries from a full reset with an exponential back-off until a solve succeeds.
   */
  bool isMpcHealthy() const { return mpcLink_->isHealthy(); }
  const MpcResetSupervisor& getResetSupervisor() const { return mpcLink_->getResetSupervisor(); }

  /**
   * @brief Set nominal joint positions for JOINT_PD mode.
   */
  void setNominalJointPositions(const std::vector<scalar_t>& positions) { nominalJointPositions_ = positions; }

  const ocs2::SystemObservation& getCurrentObservation() const { return currentMpcObservation_; }

  /**
   * Contact flags the MPC policy being executed plans for `time`. Empty until a policy solved after every reset
   * requested so far is in use (in WB_MPC; the passive modes leave the last answer). Call from the thread that runs
   * computeJointControlAction(): the policy is not thread-safe.
   */
  std::optional<contact_flag_t> getPlannedContactFlags(scalar_t time) const;

  /**
   * Source of the measured contact state (robot_model/ContactEstimator.h). It is asked once per control cycle; its
   * answer is the observation mode handed to the MPC and decides which planned contact wrenches the inverse dynamics
   * projects into feedforward torques (gateContactWrenchesByMeasuredContacts). Defaults to the flags of the RobotState
   * (RobotStateContactEstimator); in simulation the nodes install a CheaterSimContactEstimator. Set it before the
   * control loop runs; a null pointer restores the default.
   */
  void setContactEstimator(std::shared_ptr<::robot::model::ContactEstimator> contactEstimator);
  const ::robot::model::ContactEstimator& getContactEstimator() const { return *contactEstimator_; }
  /** Measured contact flags of the last control cycle (updateMpcObservation), as reported by the contact estimator. */
  const contact_flag_t& getMeasuredContactFlags() const { return measuredContactFlags_; }

  /**
   * Shaping of the planned contact wrenches at touch-down in the inverse dynamics (ContactWrenchGate: debounce and ramp
   * after measured contact; task file `contact_wrench_gate`, hot-reloadable). Defaults to the instantaneous gate.
   */
  void setContactWrenchGateConfig(const ContactWrenchGate::Config& config);
  const ContactWrenchGate& getContactWrenchGate() const { return contactWrenchGate_; }

  /**
   * Where the control thread's reports go (ControllerEventSink.h), so that computeJointControlAction() and the setters
   * the control loop calls write no log line: the robot process hands its RealtimeEventLog. nullptr (the default):
   * LoggingControllerEventSink, which logs at once, for tests and tools. Set it before the control loop runs; the sink
   * must outlive the controller.
   */
  void setEventSink(ControllerEventSink* eventSink);

  const vector_t& getLatestPolicyInput() const { return latestPolicyInput_; }
  const CommandData& getCommandData() const { return mpcLink_->getCommand(); }

  /**
   * @brief Enable/disable using gravity compensation instead of full inverse dynamics feedforward torques in WB_MPC mode.
   */
  void setUseGravityCompFeedforward(bool enable) { useGravityCompFeedforward_ = enable; }
  bool getUseGravityCompFeedforward() const { return useGravityCompFeedforward_; }

  /**
   * Duration [s] of the ramp into WB_MPC (task.yaml `mpcEntryBlendTime`, 0: switch at once). Entering WB_MPC from a
   * passive mode always keeps sending that mode's action until the first policy solved after the entry reset is active
   * (see setControlMode()); this duration then blends targets, gains and feedforward torques linearly from the held
   * action to the MPC action. Without it the feedforward jumps in one cycle from the gravity term of the passive mode
   * to the inverse dynamics of the MPC. The same ramp follows every hold of requestMpcResetAndHold() and the recovery
   * of an unhealthy solver.
   */
  void setMpcEntryBlendTime(scalar_t seconds) { mpcEntryBlendTime_ = std::max(0.0, seconds); }
  scalar_t getMpcEntryBlendTime() const { return mpcEntryBlendTime_; }
  /** True while the entry into WB_MPC is being held or blended (for tests and diagnostics). */
  bool isEnteringMpc() const { return awaitingPostResetPolicy_.load() || entryBlendStartTime_.load() >= 0.0; }

  /**
   * Time constant [s] of the SAFETY torque decay (task.yaml `safetyDecayTimeConstant`).
   *
   * In SAFETY the robot holds the posture it had at mode entry with a joint PD whose gains are both multiplied by
   * alpha(t) = exp(-t / tau), so the commanded torque is alpha * (kp * (q_hold - q) - kd * qd) and decays smoothly to
   * nothing instead of being cut in one cycle. Once alpha falls below safety_decay::kCutoff the joints are commanded zero
   * gain and zero feedforward, i.e. true zero torque; at the default tau that takes about 4 * tau seconds.
   */
  void setSafetyDecayTimeConstant(scalar_t seconds) { safetyDecayTimeConstant_ = std::max(safety_decay::kMinTimeConstant, seconds); }
  scalar_t getSafetyDecayTimeConstant() const { return safetyDecayTimeConstant_; }
  /**
   * The SAFETY decay factor alpha = exp(-elapsed / tau), snapped to 0 once it falls below safety_decay::kCutoff so the
   * mode reaches true zero torque in finite time rather than only approaching it (humanoid_common_mpc/mrt/SafetyDecay.h,
   * shared with the whole-body controller). Pure, and static so that the decay law can be tested without standing up a
   * controller and its solver thread.
   */
  static scalar_t safetyDecayFactor(scalar_t elapsedSinceEntry, scalar_t timeConstant) {
    return safety_decay::factor(elapsedSinceEntry, timeConstant);
  }
  /** alpha for the armed decay at the given observation time; 1 when SAFETY has not been entered. */
  scalar_t currentSafetyDecayFactor(scalar_t time) const;
  /** True once the SAFETY decay has reached the cutoff and the joints are commanded zero torque. */
  bool isSafetyDecayComplete() const { return safetyDecayComplete_.load(); }

 private:
  /**
   * Method to convert the latest observation msg to a stable desired trajectory (current position, zero velocity and
   * acceleration). The reset target of the MPC link, called on the thread that serves resets.
   *
   * @param [in] msg: The observation message.
   */
  TargetTrajectories currentObservationToResetTrajectory(const SystemObservation& currentMpcObservation);

  /**
   * The gains of the constructor: those of `pdGainsFile`, or the defaults when it is empty or missing. Throws
   * std::invalid_argument when the file exists but cannot be read or is refused.
   */
  JointPdGains loadInitialPdGains(const std::string& pdGainsFile) const;

  /**
   * Parses `yamlText` (from `source`, for the log) and posts the gains to the control thread with `ticket`, taken from
   * the mailbox when the document was handed in. Caller's thread.
   */
  absl::Status postPdGainsDocument(absl::string_view yamlText, absl::string_view source, uint64_t ticket);

  /** The default gains of the MPC joints when the document does not set them (the others get a fraction of them). */
  static JointPdGainsDefaults pdGainsDefaults();

  /** Control thread: keeps the elapsed times of the time-keyed actions across a rewind of `rewind` seconds, and holds. */
  void handleClockRewind(scalar_t rewind);

  /** Arms the hold of requestMpcResetAndHold() without requesting a reset. */
  void armHold(bool holdGravityComp);

  void updateMpcState(vector_t& mpcState, const ::robot::model::RobotState& robotState);
  void updateMpcObservation(ocs2::SystemObservation& mpcObservation, const ::robot::model::RobotState& robotState);

  /** JOINT_PD action: PD to the nominal posture with gravity compensation feedforward on the MPC joints. */
  void fillJointPdAction(const ::robot::model::RobotState& robotState, ::robot::model::RobotJointAction& robotJointAction);
  /** GRAVITY_COMP action: gravity compensation with light damping, joints move compliantly. */
  void fillGravityCompAction(const ::robot::model::RobotState& robotState, ::robot::model::RobotJointAction& robotJointAction);
  /** The action held while entering WB_MPC: that of the mode the controller came from. */
  void fillEntryHoldAction(const ::robot::model::RobotState& robotState, ::robot::model::RobotJointAction& robotJointAction);
  /** ZERO_TORQUE action: no gain and no feedforward on any joint. The simulator does not apply it; hardware would get nothing. */
  void fillZeroTorqueAction(const ::robot::model::RobotState& robotState, ::robot::model::RobotJointAction& robotJointAction);

  void fillSafetyAction(const ::robot::model::RobotState& robotState, ::robot::model::RobotJointAction& robotJointAction);
  /** Blends the MPC action in `robotJointAction` with the held action according to the entry ramp, if one is running. */
  void applyEntryBlend(const ::robot::model::RobotState& robotState, ::robot::model::RobotJointAction& robotJointAction);
  /** One joint of applyEntryBlend(): the action at `index`, a of the way from holdAction_'s to its own. */
  void blendJointAction(size_t index, scalar_t a, ::robot::model::RobotJointAction& robotJointAction) const;

  /** Hands an event to the event sink. Control thread. */
  void postEvent(ControllerEventCode code, scalar_t value0 = 0.0, scalar_t value1 = 0.0, absl::string_view text = {});

  // The MPC: observations out, policies in, resets and health (MpcLink). Made by the constructor's factory, first, so
  // that it is destroyed last; the destructor stops it before anything its reset target reads goes away.
  std::unique_ptr<MpcLink> mpcLink_;
  std::shared_ptr<::robot::model::ContactEstimator> contactEstimator_;
  contact_flag_t measuredContactFlags_{};  // of the last control cycle, from contactEstimator_
  ContactWrenchGate contactWrenchGate_;    // advanced with measuredContactFlags_ every cycle
  // The policy in use was solved after every reset requested so far (MRT_BASE::isActivePolicyCurrent() and no reset
  // outstanding). Written by the control thread, read by getPlannedContactFlags().
  std::atomic<bool> policyActivated_{false};
  // The reset hand-over with the solver, its failure back-off and the clock check are the link's (MpcResetSupervisor).
  // [s] Observation time of the last reset the divergence check requested; < 0: none. Control thread.
  scalar_t lastDivergenceResetTime_{-1.0};
  // LINT.IfChange(divergence_reset_interval)
  /// [s] The divergence check requests at most one reset per post-reset policy, and at most one per this interval.
  static constexpr scalar_t kDivergenceResetInterval = 0.5;
  // LINT.ThenChange(//humanoid_nmpc/docs/mpc_reset/README.md:controller_reset_events)

  PinocchioInterface pinocchioInterface_;
  ocs2::SystemObservation currentMpcObservation_;
  std::unique_ptr<CentroidalMpcRobotModel<scalar_t>> mpcRobotModelPtr_;
  /// Owned clone of the model matching the OCP input layout (basis-vector decorator or wrench model). Every input read or
  /// write goes through it; contact wrenches must be read with the state-aware ...InWorldFrame accessors because the
  /// input-only accessors return the LOCAL contact-frame wrench in basis-vector mode.
  std::unique_ptr<MpcRobotModelBase<scalar_t>> effectiveModelPtr_;
  std::vector<size_t> mpcJointIndices_;
  std::vector<size_t> otherJointIndices_;

  // Where the control thread reports (setEventSink()); never null.
  ControllerEventSink* eventSink_;
  /// The weight-compensating action of WB_MPC without a policy was reported since the last mode change.
  bool noPolicyReported_{false};

  // Workspaces of the control thread, sized by the constructor, so that a cycle allocates nothing in them.
  vector_t qPinocchio_;                          ///< generalized coordinates of the measured state (updateMpcState())
  vector_t vPinocchio_;                          ///< generalized velocities of the measured state
  vector_t mpcJointPositions_;                   ///< the MPC joints' measured positions
  vector_t mpcJointVelocities_;                  ///< the MPC joints' measured velocities
  std::vector<bool> estimatedContactFlags_;      ///< the contact estimator's answer
  vector_t gravityCoordinates_;                  ///< computeGravityCompensation()'s generalized coordinates
  vector_t gravityJointPositions_;               ///< and joint positions
  vector_t zeroGeneralizedVelocity_;             ///< and zero velocity
  vector_t gravityTorques_;                      ///< and result
  ::robot::model::RobotJointAction holdAction_;  ///< the held action the entry ramp blends from (applyEntryBlend())

  // The policy in use is evaluated as MRT_BASE::evaluatePolicy() would, bit for bit, without its heap allocations and
  // its logging (humanoid_mpc_ipc/RealtimePolicyEvaluator.h), into outputs sized once for the effective model.
  ipc::RealtimePolicyEvaluator policyEvaluator_;
  vector_t mpcPolicyState_;
  vector_t mpcPolicyInput_;

  std::string controlMode_{"WB_MPC"};            ///< Active control mode (JOINT_PD, WB_MPC, etc.)
  std::vector<scalar_t> nominalJointPositions_;  ///< Nominal positions for JOINT_PD mode
  scalar_t previousObservationTime_{0.0};        ///< Previous sim time for computing actual dt
  vector_t latestPolicyInput_;                   ///< Latest MPC policy input (e.g. contact forces, joint accelerations)

  const std::string pdGainsFile_;
  const std::vector<std::string> mpcModelJointNames_;
  const std::vector<std::string> fixedJointNames_;
  // The file watcher of the gains (pollPdGainsFile()). Declared ahead of pdGains_: the constructor records the file's
  // write time BEFORE it reads the initial gains, so that a save landing in between is reloaded by the next poll.
  absl::Mutex pdGainsFileMutex_;
  std::filesystem::file_time_type pdGainsLastWriteTime_ ABSL_GUARDED_BY(pdGainsFileMutex_);
  // The joint PD gains: those the control thread commands, and the mailbox the other threads post new ones to.
  JointPdGains pdGains_;
  JointPdGainsMailbox pdGainsMailbox_;

  bool useGravityCompFeedforward_{false};  ///< When true, use gravity comp instead of full ID torques in WB_MPC mode

  // Hand-over into WB_MPC (setMpcEntryBlendTime). Written from the mode-switch caller, read in the control loop.
  scalar_t mpcEntryBlendTime_{0.0};                   ///< [s] 0: immediate switch (default)
  std::atomic<bool> awaitingPostResetPolicy_{false};  ///< holding the previous mode's action until a post-reset policy is active
  std::atomic<bool> entryHoldGravityComp_{false};     ///< the held action is GRAVITY_COMP (else JOINT_PD)
  std::atomic<scalar_t> entryBlendStartTime_{-1.0};   ///< observation time the ramp started at, < 0: no ramp running

  // SAFETY damped decay (setSafetyDecayTimeConstant, law in safety_decay::factor). Armed by the mode switch, captured
  // and read in the control loop.
  scalar_t safetyDecayTimeConstant_{0.5};             ///< [s] time constant of alpha(t) = exp(-t / tau)
  std::atomic<scalar_t> safetyDecayStartTime_{-1.0};  ///< observation time at entry, < 0: not yet captured
  std::atomic<bool> safetyDecayComplete_{false};      ///< alpha has reached the cutoff, commanding zero torque
  vector_t safetyHoldMpcJointPositions_;              ///< posture held by the MPC joints, captured at entry
  vector_t safetyHoldOtherJointPositions_;            ///< posture held by the non-MPC joints, captured at entry

  /**
   * The base-held gravity torques g_j(q), via Pinocchio's nonLinearEffects at zero velocity: what each joint must apply
   * to hold the chain distal to it, with the base externally supported and the feet carrying nothing. Correct for a robot
   * hanging on the gantry, which is how GRAVITY_COMP is operated.
   *
   * NOT what a robot standing on its own feet needs: those joint rows also carry -J_{c,j}^T f, see fillGravityCompAction.
   * The result is a workspace of this controller, valid until the next call.
   */
  const vector_t& computeGravityCompensation(const ::robot::model::RobotState& robotState);
};

}  // namespace ocs2::humanoid
