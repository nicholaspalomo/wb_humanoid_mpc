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

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "ocs2_core/manifold/StateManifold.h"
#include "ocs2_mpc/MPC_BASE.h"
#include "ocs2_mpc/MPC_MRT_Interface.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "humanoid_common_mpc/contact/ContactWrenchGate.h"
#include "humanoid_common_mpc/mrt/ContactEstimateIntake.h"
#include "humanoid_common_mpc/mrt/ControllerEventSink.h"
#include "humanoid_common_mpc/mrt/JointPdGains.h"
#include "humanoid_common_mpc/mrt/JointPdGainsMailbox.h"
#include "humanoid_common_mpc/mrt/LookaheadPolicyEvaluator.h"
#include "humanoid_common_mpc/mrt/MpcLink.h"
#include "humanoid_common_mpc/mrt/MpcResetSupervisor.h"
#include "humanoid_common_mpc/mrt/SafetyDecay.h"
#include "humanoid_common_mpc/pinocchio_model/RealtimeRigidBodyDynamics.h"
#include "humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h"
#include "humanoid_mpc_config/joint_pd_gains_file.nproto.h"
#include "humanoid_mpc_ipc/RealtimePolicyEvaluator.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"
#include "robot_model/ContactEstimator.h"
#include "robot_model/ControllerBase.h"
#include "robot_model/RobotDescription.h"

namespace ocs2::humanoid {

/**
 * The MRT joint controller of the whole-body MPC: every control cycle it hands
 * the MPC the measured state, executes the policy in use as joint position,
 * velocity and feedforward-torque targets (the inverse dynamics of the planned
 * accelerations and contact wrenches), and runs the passive modes (ZERO_TORQUE,
 * JOINT_PD, GRAVITY_COMP, SAFETY) without it. Made by Create().
 * computeJointControlAction() and the mode setters run on the control thread;
 * the PD-gain setters and the reset requests are thread-safe, as documented on
 * each.
 */
class WBMpcMrtJointController final : public ::robot::model::ControlBase {
 public:
  /**
   * Makes a controller whose MPC runs in this process (InProcessMpcLink over
   * `mpc`).
   *
   * @param [in] mpc: The underlying MPC class to be used; it must outlive the
   * controller.
   * @param [in] solverObservationObserver: Called with the observation of every
   * solve, on the solver thread right before it
   * (InProcessMpcLink::Config::solverObservationObserver), and kept by the
   * link: the observe() of the MPC's ProceduralMpcMotionManager, as the MPC
   * node wires it; null for an MPC without one.
   * @param [in] mpcDesiredFrequency: The max frequency to run the mpc at; <= 0
   * runs the solves back to back.
   * @param [in] pdGainsFile: the robot's joint_pd_gains.textproto
   * (loadJointPdGains()), required: the controller has no gains of its own.
   * Reloaded by pollPdGainsFile(). Its max_joint_torque bounds every
   * feedforward torque the controller commands, as in the centroidal
   * controller.
   * @return The errors of loadRequiredJointPdGains(), naming the file:
   * InvalidArgument when `pdGainsFile` is empty, cannot be read or is refused
   * (a default_gains without kp, kd and max_joint_torque included), NotFound
   * when it does not exist - at start-up there are no gains in use to keep.
   * NotFound naming the joint when `modelSettings` names a joint (an MPC joint
   * or a fixed one) that `robotDescription` does not have, and the error of
   *         checkContactFramesExist() for contact frames the Pinocchio model
   * does not have.
   */
  static absl::StatusOr<std::unique_ptr<WBMpcMrtJointController>> Create(
      const ::robot::model::RobotDescription& robotDescription,
      const ModelSettings& modelSettings,
      MPC_BASE& mpc,
      MPC_MRT_Interface::SolverObservationObserver solverObservationObserver,
      PinocchioInterface pinocchioInterface,
      scalar_t mpcDesiredFrequency,
      const std::string& pdGainsFile);

  /**
   * Makes a controller that reaches its MPC through the link `mpcLinkFactory`
   * makes (MpcLink.h), for instance a remote one. The factory is called once,
   * here, with this controller's reset target.
   *
   * @return InvalidArgument when the factory makes no link, and the errors of
   * the overload above.
   */
  static absl::StatusOr<std::unique_ptr<WBMpcMrtJointController>> Create(
      const ::robot::model::RobotDescription& robotDescription,
      const ModelSettings& modelSettings,
      const MpcLinkFactory& mpcLinkFactory,
      PinocchioInterface pinocchioInterface,
      const std::string& pdGainsFile);

  WBMpcMrtJointController(const WBMpcMrtJointController&) = delete;
  WBMpcMrtJointController& operator=(const WBMpcMrtJointController&) = delete;

  /**
   * Destructor. Stops the link (for InProcessMpcLink: the solver thread) and
   * waits for it.
   */
  ~WBMpcMrtJointController() override;

  bool ready() const override { return mpcLink_->initialPolicyReceived(); }

  /**
   * Handles the low level controller loop that updates the mpc observation,
   * reads out the latest policy and sets the joint control action.
   *
   * In WB_MPC the policy is read one lookahead (the time since the previous
   * cycle, clamped to [kMinPolicyLookahead, kMaxPolicyLookahead]) after the
   * measurement - the joint targets and the feedforward - while its feedback
   * acts on the error at the measurement time (LookaheadPolicyEvaluator): a
   * robot exactly on a moving plan gets exactly the feedforward. Every
   * feedforward torque is clamped to its joint's max_joint_torque, in every
   * mode, and the joints outside the MPC hold the nominal posture in every mode
   * but SAFETY and ZERO_TORQUE.
   *
   * In the passive modes, the hold of WB_MPC, its weight compensation before a
   * policy and the execution of a policy it allocates nothing once the
   * controller is built - Pinocchio runs on the FreeFlyer twin of the model
   * (RealtimeRigidBodyDynamics, humanoid_common_mpc_app/robot/README.md) -,
   * takes no lock and writes no log line (reports go to the event sink). A
   * policy the realtime evaluator refuses is never evaluated by MRT_BASE (which
   * allocates, and aborts on a malformed policy): the robot is held with the
   * JOINT_PD action, and kPolicyRefused posted, until a policy it evaluates is
   * in use. `robotJointAction` is a RobotJointAction of the constructor's robot
   * description: its joints are reached without a check.
   */
  void computeJointControlAction(
      scalar_t time,
      const ::robot::model::RobotState& robotState,
      ::robot::model::RobotJointAction& robotJointAction) override;

  /** Starts the MPC link (MpcLink::start()) from the observation of
   * `initRobotState`. Call it once, before the control loop. */
  void startMpcThread(const ::robot::model::RobotState& initRobotState);

  /** The link the controller reaches its MPC through. */
  MpcLink& getMpcLink() { return *mpcLink_; }
  const MpcLink& getMpcLink() const { return *mpcLink_; }

  /**
   * Hands the controller the joint PD gains of a whole gains file (as the
   * tuning GUI publishes it on operator/pd_gains), without writing the file.
   * Thread-safe, for any non-realtime thread; resolved on the caller's thread
   * (jointPdGainsFromConfig()) and taken into use by the control thread at its
   * next cycle without locking or allocating (see
   * CentroidalMpcMrtJointController::setPdGains). Gains
   * jointPdGainsFromConfig() refuses are returned as its InvalidArgument and
   * change nothing. Of files handed in concurrently (here, or by
   * pollPdGainsFile()), the one handed in last wins, whichever is resolved
   * first.
   */
  absl::Status setPdGains(const mpc_config::JointPdGainsFile& gains);

  /**
   * Reloads the PD gains file of the constructor when it has been written since
   * it was last read; the file watcher of the gains, for a non-realtime thread
   * to call at the rate it should be checked. A file that does not parse is
   * reported and leaves the gains as they are. A file written before the gains
   * setPdGains() was last handed (a Save followed by more tuning within one
   * poll period) is an older edit and is not applied over them. Thread-safe.
   */
  void pollPdGainsFile();

  /** The gains the control thread commands. Control thread. */
  const JointPdGains& getPdGains() const { return pdGains_; }

  const ocs2::SystemObservation& getCurrentObservation() const {
    return currentMpcObservation_;
  }

  /**
   * @brief Set the active control mode, as the centroidal controller does
   * (CentroidalMpcMrtJointController::setControlMode).
   *
   * The passive modes compute their action here, independent of the MPC:
   *  - ZERO_TORQUE: no gain and no feedforward;
   *  - JOINT_PD: PD to the nominal posture (setNominalJointPositions) with
   * gravity compensation on the MPC joints (the joints outside the MPC hold the
   * nominal posture in WB_MPC as well);
   *  - GRAVITY_COMP: gravity compensation with light damping, the joints move
   * compliantly;
   *  - SAFETY: damped PD about the posture at entry, both gains decaying to
   * zero (humanoid_common_mpc/mrt/SafetyDecay.h). WB_MPC (or MPC_ACTIVE)
   * executes the MPC policy. Entering it from a passive mode resets the MPC
   * from the observation at that moment and keeps the action of the mode it
   * came from (JOINT_PD for ZERO_TORQUE) until a policy solved after that reset
   * is in use. The default mode is WB_MPC, which is what the controller did
   * before it had modes. Call it from the thread that runs
   * computeJointControlAction().
   */
  void setControlMode(absl::string_view mode);
  const std::string& getControlMode() const { return controlMode_; }

  /** Nominal joint positions of JOINT_PD, indexed like the RobotDescription's
   * joints. Empty: hold the current posture. */
  void setNominalJointPositions(const std::vector<scalar_t>& positions) {
    nominalJointPositions_ = positions;
  }

  /** Time constant [s] of the SAFETY decay (see
   * CentroidalMpcMrtJointController::setSafetyDecayTimeConstant). */
  void setSafetyDecayTimeConstant(scalar_t seconds);
  scalar_t getSafetyDecayTimeConstant() const {
    return safetyDecayTimeConstant_;
  }

  /**
   * Contact flags the MPC policy being executed plans for `time`. Empty until a
   * policy solved after every reset requested so far is in use. Call from the
   * thread that runs computeJointControlAction(): the policy is not
   * thread-safe.
   */
  std::optional<contact_flag_t> getPlannedContactFlags(scalar_t time) const;

  /**
   * Source of the measured contact state (robot_model/ContactEstimator.h). It
   * is asked once per control cycle; its answer is the observation mode handed
   * to the MPC and decides which planned contact wrenches the inverse dynamics
   * projects into feedforward torques (ContactWrenchGate). Defaults to the
   * flags of the RobotState (RobotStateContactEstimator); in simulation the
   * nodes install a CheaterSimContactEstimator. Control thread, between cycles
   * (the robot process swaps it live, RobotProcess::applyControllerSettings()):
   * a non-null estimator is taken without allocating, as its getName() is a
   * view of a static literal; a null pointer restores the default, which
   * allocates it and is for start-up only.
   */
  void setContactEstimator(
      std::shared_ptr<::robot::model::ContactEstimator> contactEstimator);
  const ::robot::model::ContactEstimator& getContactEstimator() const {
    return *contactEstimator_;
  }
  /** Measured contact flags of the last control cycle (updateMpcObservation),
   * as reported by the contact estimator. */
  const contact_flag_t& getMeasuredContactFlags() const {
    return measuredContactFlags_;
  }
  /**
   * The control cycles whose contact estimate was refused because it did not
   * carry one flag per contact point: a configuration error of the estimator,
   * after which the measured contact flags stay those of the cycle before. The
   * first refusal of each run posts
   * ControllerEventCode::kContactEstimateRefused to the event sink, which the
   * robot process logs as a warning (ContactEstimateIntake). Thread-safe.
   */
  uint64_t getNumRefusedContactEstimates() const {
    return contactEstimateIntake_.numRefused();
  }

  /**
   * Shaping of the planned contact wrenches at touch-down in the inverse
   * dynamics (ContactWrenchGate: debounce and ramp after measured contact; task
   * file `contact_wrench_gate`). Defaults to the instantaneous gate. A
   * configuration ContactWrenchGate::validateConfig() refuses keeps the gate in
   * use and posts kContactWrenchGateRefused.
   */
  void setContactWrenchGateConfig(const ContactWrenchGate::Config& config);
  const ContactWrenchGate& getContactWrenchGate() const {
    return contactWrenchGate_;
  }

  /**
   * Where the control thread's reports go (ControllerEventSink.h); see
   * CentroidalMpcMrtJointController::setEventSink(). nullptr (the default):
   * LoggingControllerEventSink. Set it before the control loop runs; the sink
   * must outlive the controller.
   */
  void setEventSink(ControllerEventSink* absl_nullable eventSink);

  /** The joint names of the robot description the controller was built for, by
   * joint index. */
  const std::vector<std::string>& getRobotJointNames() const {
    return robotJointNames_;
  }

  const vector_t& getLatestPolicyInput() const { return latestPolicyInput_; }
  const CommandData& getCommandData() const { return mpcLink_->getCommand(); }

  /**
   * @brief Requests a reset of the MPC, served by the solver thread before its
   * next solve: the solver, the reference manager and every synchronized module
   * go back to their state after construction (MPC_BASE::reset()), the policy
   * waiting in the MRT buffer is dropped, and the MPC restarts from the
   * observation current when the reset is served. Callable from any thread. The
   * policy in use is executed until the first policy solved after the reset
   * replaces it; where nothing solved before may reach the robot, use
   * requestMpcResetAndHold().
   */
  void requestMpcReset() {
    // The policy in use no longer counts as solved after every reset requested
    // (getPlannedContactFlags()).
    policyActivated_.store(false);
    mpcLink_->requestReset();
  }

  /**
   * @brief requestMpcReset(), and WB_MPC holds the robot until a policy solved
   * after the reset is in use: with the GRAVITY_COMP action when
   * `holdGravityComp`, with the JOINT_PD action otherwise. For discontinuities
   * of the plant and for entering WB_MPC. Call it from the thread that runs
   * computeJointControlAction().
   */
  void requestMpcResetAndHold(bool holdGravityComp = false);

  /** True while WB_MPC is holding the robot for a policy solved after the last
   * reset (or for a healthy solver). */
  bool isHolding() const { return awaitingPostResetPolicy_.load(); }

  /** False while the MPC solver keeps failing; WB_MPC then holds the robot with
   * the JOINT_PD action (MpcResetSupervisor). */
  bool isMpcHealthy() const { return mpcLink_->isHealthy(); }
  const MpcResetSupervisor& getResetSupervisor() const {
    return mpcLink_->getResetSupervisor();
  }

 private:
  /** The gains the controller starts with, and the write time of their file
   * when they were read. */
  struct InitialPdGains {
    std::filesystem::file_time_type fileWriteTime;
    JointPdGains gains;
  };

  /**
   * The controller of Create() once its gains are loaded; `mpcLinkFactory` may
   * still have made no link, which Create() checks.
   */
  WBMpcMrtJointController(
      const ::robot::model::RobotDescription& robotDescription,
      const ModelSettings& modelSettings,
      const MpcLinkFactory& mpcLinkFactory,
      PinocchioInterface pinocchioInterface,
      const std::string& pdGainsFile,
      InitialPdGains initialPdGains,
      std::shared_ptr<const StateManifold> stateManifold,
      std::unique_ptr<RealtimeRigidBodyDynamics> rigidBodyDynamics);

  /**
   * The target the MPC restarts from after a reset: the observed pose held
   * still and upright, over two nodes two seconds apart, with the input that
   * carries the weight on both feet. The reset target of the MPC link, called
   * on the thread that serves resets.
   */
  TargetTrajectories currentObservationToResetTrajectory(
      const SystemObservation& currentObservation);

  /**
   * The gains to start with: those of `pdGainsFile`
   * (loadRequiredJointPdGains()), with the file's write time, recorded BEFORE
   * the gains are read so that a save landing in between is reloaded by the
   * next poll. Its errors when the file is empty, missing or refused.
   */
  static absl::StatusOr<InitialPdGains> loadInitialPdGains(
      const ModelSettings& modelSettings, const std::string& pdGainsFile);

  /**
   * Posts `gains` (resolved from `source`, for the log) to the control thread
   * with `ticket`, taken from the mailbox when they were handed in; an error is
   * logged and returned, and posts nothing. Caller's thread.
   */
  absl::Status postPdGains(const absl::StatusOr<JointPdGains>& gains,
                           absl::string_view source,
                           uint64_t ticket);

  /** Control thread: keeps the elapsed SAFETY time across a rewind of `rewind`
   * seconds, and holds. */
  void handleClockRewind(scalar_t rewind);

  /** Arms the hold of requestMpcResetAndHold() without requesting a reset. */
  void armHold(bool holdGravityComp);

  void updateMpcState(vector_t& mpcState,
                      const ::robot::model::RobotState& robotState);
  void updateMpcObservation(ocs2::SystemObservation& mpcObservation,
                            const ::robot::model::RobotState& robotState);

  /**
   * The base-held gravity torques of the MPC joints at the measured
   * configuration (see the centroidal controller). The result is a workspace of
   * this controller, valid until the next call.
   */
  const vector_t& computeGravityCompensation(
      const ::robot::model::RobotState& robotState);

  /** Hands an event to the event sink. Control thread. */
  void postEvent(ControllerEventCode code,
                 scalar_t value0 = 0.0,
                 scalar_t value1 = 0.0,
                 absl::string_view text = {});
  /** q_des of JOINT_PD for a joint: its nominal position, or its current one
   * while no nominal posture is set. */
  scalar_t nominalJointPosition(const ::robot::model::RobotState& robotState,
                                size_t index) const;
  /** `torque` clamped to the torque limit of MPC joint `mpcJoint` (|ff| <=
   * max_joint_torque). Control thread. */
  scalar_t clampedMpcJointFeedforward(size_t mpcJoint, scalar_t torque) const;
  /**
   * The action of the joints outside the MPC in JOINT_PD, GRAVITY_COMP (kpScale
   * 0.5), the hold and the execution of WB_MPC: their nominal posture
   * (nominalJointPosition()), so that no hand-over between those modes moves
   * them.
   */
  void fillOtherJointsHold(const ::robot::model::RobotState& robotState,
                           scalar_t kpScale,
                           ::robot::model::RobotJointAction& robotJointAction);
  /**
   * The MPC joints' action of the evaluated policy `mpcPolicyState`,
   * `mpcPolicyInput`: the planned joint angles and velocities, and the
   * feedforward of the free-base inverse dynamics
   * (RealtimeRigidBodyDynamics::freeBaseJointTorques()) of the planned state,
   * joint accelerations and gated contact wrenches. Allocation-free. Control
   * thread.
   */
  void executePolicy(const vector_t& mpcPolicyState,
                     const vector_t& mpcPolicyInput,
                     ::robot::model::RobotJointAction& robotJointAction);
  /**
   * WB_MPC before its first policy: the observed state held with the input that
   * carries the robot's weight on the feet measured in contact (none: the base
   * is held by the gantry), written into `mpcPolicyState` and `mpcPolicyInput`
   * (sized for the model), and the MPC joints' action with the base-held
   * inverse dynamics of that input
   * (RealtimeRigidBodyDynamics::heldBaseJointTorques()). Allocation-free.
   * Control thread.
   */
  void fillWeightCompensationAction(
      const ::robot::model::RobotState& robotState,
      vector_t& mpcPolicyState,
      vector_t& mpcPolicyInput,
      ::robot::model::RobotJointAction& robotJointAction);
  /** Writes the Pinocchio configuration of the whole-body `state`, its
   * quaternion safely normalized, into policyCoordinates_. */
  void fillPolicyCoordinates(const vector_t& state);
  /**
   * Posts kPolicyUpdateLockContended once per run of failed try-locks of the
   * MRT's policy buffer longer than
   * MRT_BASE::kPolicyUpdateLockContentionThreshold, which MRT_BASE counts
   * instead of logging. Control thread.
   */
  void reportPolicyUpdateLockContention();

  void fillZeroTorqueAction(const ::robot::model::RobotState& robotState,
                            ::robot::model::RobotJointAction& robotJointAction);
  void fillJointPdAction(const ::robot::model::RobotState& robotState,
                         ::robot::model::RobotJointAction& robotJointAction);
  void fillGravityCompAction(
      const ::robot::model::RobotState& robotState,
      ::robot::model::RobotJointAction& robotJointAction);
  void fillSafetyAction(const ::robot::model::RobotState& robotState,
                        ::robot::model::RobotJointAction& robotJointAction);
  /** The action held while WB_MPC waits for a post-reset policy: that of the
   * mode the controller came from. */
  void fillHoldAction(const ::robot::model::RobotState& robotState,
                      ::robot::model::RobotJointAction& robotJointAction);

  // The MPC: observations out, policies in, resets and health (MpcLink). Made
  // by the constructor's factory, first, so that it is destroyed last; the
  // destructor stops it before anything its reset target reads goes away.
  std::unique_ptr<MpcLink> mpcLink_;
  std::shared_ptr<::robot::model::ContactEstimator> contactEstimator_;
  contact_flag_t measuredContactFlags_{};  // of the last control cycle, from
                                           // contactEstimator_
  ContactEstimateIntake
      contactEstimateIntake_;  // takes or refuses each estimate into
                               // measuredContactFlags_
  ContactWrenchGate
      contactWrenchGate_;  // advanced with measuredContactFlags_ every cycle
  // The policy in use was solved after every reset requested so far
  // (MRT_BASE::isActivePolicyCurrent() and no reset outstanding). Written by
  // the control thread, read by getPlannedContactFlags().
  std::atomic<bool> policyActivated_{false};
  // The reset hand-over with the solver, its failure back-off and the clock
  // check are the link's (MpcResetSupervisor).

  PinocchioInterface pinocchioInterface_;
  ocs2::SystemObservation currentMpcObservation_;
  WBAccelMpcRobotModel<scalar_t> mpcRobotModel_;
  std::vector<size_t> mpcJointIndices_;
  std::vector<size_t> otherJointIndices_;

  // Where the control thread reports (setEventSink()); never null.
  ControllerEventSink* absl_nonnull eventSink_;
  /// The weight-compensating action of WB_MPC without a policy was reported
  /// since the last mode change.
  bool noPolicyReported_ = false;
  /// A refused policy (kPolicyRefused) was reported since the last cycle that
  /// evaluated one.
  bool refusedPolicyReported_ = false;

  // Workspaces of the control thread, sized by the constructor, so that a cycle
  // allocates nothing in them.
  vector_t mpcJointPositions_;   ///< the MPC joints' measured positions
                                 ///< (updateMpcState())
  vector_t mpcJointVelocities_;  ///< and velocities
  std::vector<bool> estimatedContactFlags_;  ///< the contact estimator's answer
  vector_t gravityCoordinates_;  ///< computeGravityCompensation()'s Pinocchio
                                 ///< configuration q (nq)
  vector_t policyCoordinates_;   ///< the Pinocchio configuration q of a planned
                                 ///< or observed state (nq)

  // The state manifold of mpcRobotModel_'s layout (stateManifoldOf()), never
  // null; declared before policyEvaluator_, which retains a pointer to it. A
  // std::shared_ptr because the MRT and the rollout of OCS2 take the manifold
  // as one.
  const std::shared_ptr<const StateManifold> stateManifold_;
  // The policy in use, evaluated without MRT_BASE::evaluatePolicy()'s heap
  // allocations and logging (humanoid_mpc_ipc/RealtimePolicyEvaluator.h) for a
  // command one lookahead after the measurement: the plan and the feedforward
  // at the lookahead, the feedback at the measurement
  // (LookaheadPolicyEvaluator.h), into outputs sized once for the model.
  LookaheadPolicyEvaluator policyEvaluator_;
  // The gravity torques, and the inverse dynamics of an evaluated policy and of
  // the weight compensation, at the sole frames of the model settings, without
  // a heap allocation (on the FreeFlyer twin of pinocchioInterface_'s model).
  // Never null.
  std::unique_ptr<RealtimeRigidBodyDynamics> rigidBodyDynamics_;
  // [N] The weight the feet measured in contact share in WB_MPC before a policy
  // (computeRobotWeight()).
  scalar_t robotWeight_ = 0.0;
  vector_t mpcPolicyState_;
  vector_t mpcPolicyInput_;

  scalar_t previousObservationTime_ =
      0.0;                      ///< Previous sim time for computing actual dt
  vector_t latestPolicyInput_;  ///< Latest MPC policy input

  std::string controlMode_ =
      "WB_MPC";  ///< Active control mode (see setControlMode)
  std::vector<scalar_t>
      nominalJointPositions_;  ///< Nominal positions for JOINT_PD mode
  // Hold of WB_MPC for a post-reset policy (requestMpcResetAndHold). Armed by
  // the mode switch, read in the control loop.
  std::atomic<bool> awaitingPostResetPolicy_{false};
  std::atomic<bool> holdGravityComp_{
      false};  ///< the held action is GRAVITY_COMP (else JOINT_PD)

  // SAFETY damped decay, captured on the first cycle in the mode.
  scalar_t safetyDecayTimeConstant_ =
      safety_decay::kDefaultTimeConstant;  ///< [s]
  /// Observation time at entry; nullopt until the first cycle in the mode
  /// captures it. Negative once a clock rewind past the entry has shifted it
  /// (handleClockRewind()).
  std::optional<scalar_t> safetyDecayStartTime_;
  /// The alpha of the last SAFETY cycle, which the next may not exceed
  /// (safety_decay::latchedFactor()); 1 at entry.
  scalar_t safetyDecayLatchedFactor_ = 1.0;
  vector_t safetyHoldMpcJointPositions_;
  vector_t safetyHoldOtherJointPositions_;

  /** The gains file the controller reads and watches; empty: none. */
  const std::string pdGainsFile_;
  const std::vector<std::string>
      robotJointNames_;  ///< the joints of the robot description, by joint
                         ///< index
  // The joints of the model settings the gains are resolved for: copies, read
  // by the threads that load gains.
  const std::vector<std::string> mpcModelJointNames_;
  const std::vector<std::string> fixedJointNames_;
  // The file watcher of the gains (pollPdGainsFile()), from the write time
  // loadInitialPdGains() recorded before it read the initial gains, so that a
  // save landing in between is reloaded by the next poll.
  absl::Mutex pdGainsFileMutex_;
  std::filesystem::file_time_type pdGainsLastWriteTime_
      ABSL_GUARDED_BY(pdGainsFileMutex_);
  // When setPdGains() was last handed gains, on the clock of the file's
  // modification times: a gains file written before it is an older edit than
  // those gains, and pollPdGainsFile() does not apply it. nullopt: never.
  std::optional<std::filesystem::file_time_type> liveGainsArrival_
      ABSL_GUARDED_BY(pdGainsFileMutex_);
  // The joint PD gains: those the control thread commands, and the mailbox the
  // other threads post new ones to.
  JointPdGains pdGains_;
  JointPdGainsMailbox pdGainsMailbox_;
};

}  // namespace ocs2::humanoid
