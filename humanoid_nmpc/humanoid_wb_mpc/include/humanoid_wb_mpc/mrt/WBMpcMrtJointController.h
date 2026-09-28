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

#include <ocs2_mpc/MPC_MRT_Interface.h>

#include <humanoid_wb_mpc/common/WBAccelMpcRobotModel.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <ocs2_ros2_interfaces/mrt/DummyObserver.h>
#include <robot_model/ContactEstimator.h>
#include <robot_model/ControllerBase.h>
#include "humanoid_common_mpc/contact/ContactWrenchGate.h"
#include "humanoid_common_mpc/mrt/MpcResetSupervisor.h"
#include "humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h"
#include "robot_model/RobotDescription.h"

#include <atomic>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include "absl/strings/string_view.h"

namespace ocs2::humanoid {

class WBMpcMrtJointController final : public ::robot::model::ControlBase {
 public:
  /**
   * Constructor.
   *
   * @param [in] mpc: The underlying MPC class to be used.
   * @param [in] topicPrefix: The robot's name.
   * @param [in] mpcDesiredFrequency: The max frequency to run the mpc at.
   */
  WBMpcMrtJointController(const ::robot::model::RobotDescription& robotDescription,
                          const ModelSettings& modelSettings,
                          MPC_BASE& mpc,
                          PinocchioInterface pinocchioInterface,
                          scalar_t mpcDesiredFrequency = -1,
                          std::shared_ptr<DummyObserver> rVizVisualizerPtr = nullptr,
                          const std::string& pdGainsFile = "");

  /**
   * Destructor. Stops the solver thread and waits for it.
   */
  ~WBMpcMrtJointController();

  bool ready() const { return mcpMrtInterface_.initialPolicyReceived(); }

  /**
   * Handles the low level controller loop that updates the mpc observation, reads out the latest policy and sets the joint control action.
   */

  void computeJointControlAction(scalar_t time,
                                 const ::robot::model::RobotState& robotState,
                                 ::robot::model::RobotJointAction& robotJointAction) override;

  void startMpcThread(const ::robot::model::RobotState& initRobotState);

  void loadPdGains(const std::string& pdGainsFile, const ModelSettings& modelSettings);

  /**
   * Subscribe to the /pd_gains_updates ROS topic for real-time
   * PD gain updates from the GUI (without writing to joint_pd_gains.yaml).
   */
  void subscribePdGains(rclcpp::Node::SharedPtr node);

  const ocs2::SystemObservation& getCurrentObservation() const { return currentMpcObservation_; }

  /**
   * @brief Set the active control mode, as the centroidal controller does (CentroidalMpcMrtJointController::setControlMode).
   *
   * The passive modes compute their action here, independent of the MPC:
   *  - ZERO_TORQUE: no gain and no feedforward;
   *  - JOINT_PD: PD to the nominal posture (setNominalJointPositions) with gravity compensation on the MPC joints;
   *  - GRAVITY_COMP: gravity compensation with light damping, the joints move compliantly;
   *  - SAFETY: damped PD about the posture at entry, both gains decaying to zero (humanoid_common_mpc/mrt/SafetyDecay.h).
   * WB_MPC (or MPC_ACTIVE) executes the MPC policy. Entering it from a passive mode resets the MPC from the observation
   * at that moment and keeps the action of the mode it came from (JOINT_PD for ZERO_TORQUE) until a policy solved after
   * that reset is in use. The default mode is WB_MPC, which is what the controller did before it had modes. Call it
   * from the thread that runs computeJointControlAction().
   */
  void setControlMode(absl::string_view mode);
  const std::string& getControlMode() const { return controlMode_; }

  /** Nominal joint positions of JOINT_PD, indexed like the RobotDescription's joints. Empty: hold the current posture. */
  void setNominalJointPositions(const std::vector<scalar_t>& positions) { nominalJointPositions_ = positions; }

  /** Time constant [s] of the SAFETY decay (see CentroidalMpcMrtJointController::setSafetyDecayTimeConstant). */
  void setSafetyDecayTimeConstant(scalar_t seconds);
  scalar_t getSafetyDecayTimeConstant() const { return safetyDecayTimeConstant_; }

  /**
   * Contact flags the MPC policy being executed plans for `time`. Empty until a policy solved after every reset
   * requested so far is in use. Call from the thread that runs computeJointControlAction(): the policy is not
   * thread-safe.
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
   * after measured contact; task file `contact_wrench_gate`). Defaults to the instantaneous gate.
   */
  void setContactWrenchGateConfig(const ContactWrenchGate::Config& config);
  const ContactWrenchGate& getContactWrenchGate() const { return contactWrenchGate_; }

  const vector_t& getLatestPolicyInput() const { return latestPolicyInput_; }
  const CommandData& getCommandData() const { return mcpMrtInterface_.getCommand(); }

  /**
   * @brief Requests a reset of the MPC, served by the solver thread before its next solve: the solver, the reference
   *        manager and every synchronized module go back to their state after construction (MPC_BASE::reset()), the
   *        policy waiting in the MRT buffer is dropped, and the MPC restarts from the observation current when the
   *        reset is served. Callable from any thread. The policy in use is executed until the first policy solved
   *        after the reset replaces it; where nothing solved before may reach the robot, use requestMpcResetAndHold().
   */
  void requestMpcReset() {
    // The policy in use no longer counts as solved after every reset requested (getPlannedContactFlags()).
    policyActivated_.store(false);
    resetSupervisor_.requestReset();
  }

  /**
   * @brief requestMpcReset(), and WB_MPC holds the robot until a policy solved after the reset is in use: with the
   *        GRAVITY_COMP action when `holdGravityComp`, with the JOINT_PD action otherwise. For discontinuities of the
   *        plant and for entering WB_MPC. Call it from the thread that runs computeJointControlAction().
   */
  void requestMpcResetAndHold(bool holdGravityComp = false);

  /** True while WB_MPC is holding the robot for a policy solved after the last reset (or for a healthy solver). */
  bool isHolding() const { return awaitingPostResetPolicy_.load(); }

  /** False while the MPC solver keeps failing; WB_MPC then holds the robot with the JOINT_PD action (MpcResetSupervisor). */
  bool isMpcHealthy() const { return resetSupervisor_.isHealthy(); }
  const MpcResetSupervisor& getResetSupervisor() const { return resetSupervisor_; }

 private:
  /**
   * Handles the MPC solver thread.
   */
  void solverWorker();

  /**
   * The target the MPC restarts from after a reset: the observed pose held still and upright, over two nodes two
   * seconds apart, with the input that carries the weight on both feet.
   */
  TargetTrajectories currentObservationToResetTrajectory(const SystemObservation& currentMpcObservation);

  /**
   * Solver thread: MPC_MRT_Interface::resetMpcNode() (`full`) or resetMpcSolver() from the observation current now.
   * Logged while the MPC is healthy.
   */
  void resetMpcToCurrentObservation(bool full, absl::string_view reason);

  /** Control thread: keeps the elapsed SAFETY time across a rewind of `rewind` seconds, and holds. */
  void handleClockRewind(scalar_t rewind);

  /** Arms the hold of requestMpcResetAndHold() without requesting a reset. */
  void armHold(bool holdGravityComp);

  void updateMpcState(vector_t& mpcState, const ::robot::model::RobotState& robotState);
  void updateMpcObservation(ocs2::SystemObservation& mpcObservation, const ::robot::model::RobotState& robotState);

  /** The base-held gravity torques of the MPC joints at the measured configuration (see the centroidal controller). */
  vector_t computeGravityCompensation(const ::robot::model::RobotState& robotState);
  /** q_des of JOINT_PD for a joint: its nominal position, or its current one while no nominal posture is set. */
  scalar_t nominalJointPosition(const ::robot::model::RobotState& robotState, size_t index) const;

  void fillZeroTorqueAction(const ::robot::model::RobotState& robotState, ::robot::model::RobotJointAction& robotJointAction);
  void fillJointPdAction(const ::robot::model::RobotState& robotState, ::robot::model::RobotJointAction& robotJointAction);
  void fillGravityCompAction(const ::robot::model::RobotState& robotState, ::robot::model::RobotJointAction& robotJointAction);
  void fillSafetyAction(const ::robot::model::RobotState& robotState, ::robot::model::RobotJointAction& robotJointAction);
  /** The action held while WB_MPC waits for a post-reset policy: that of the mode the controller came from. */
  void fillHoldAction(const ::robot::model::RobotState& robotState, ::robot::model::RobotJointAction& robotJointAction);

  MPC_MRT_Interface mcpMrtInterface_;
  std::shared_ptr<::robot::model::ContactEstimator> contactEstimator_;
  contact_flag_t measuredContactFlags_{};  // of the last control cycle, from contactEstimator_
  ContactWrenchGate contactWrenchGate_;    // advanced with measuredContactFlags_ every cycle
  // The policy in use was solved after every reset requested so far (MRT_BASE::isActivePolicyCurrent() and no reset
  // outstanding). Written by the control thread, read by getPlannedContactFlags().
  std::atomic<bool> policyActivated_{false};
  // Reset hand-over with the solver thread, failure back-off and the clock check (see MpcResetSupervisor).
  MpcResetSupervisor resetSupervisor_;

  PinocchioInterface pinocchioInterface_;
  ocs2::SystemObservation currentMpcObservation_;
  WBAccelMpcRobotModel<scalar_t> mpcRobotModel_;
  std::vector<size_t> mpcJointIndices_;
  std::vector<size_t> otherJointIndices_;

  size_t mpcDeltaTMicroSeconds_;
  bool realtime_;  // True if MPC is to be run as fast as possible

  std::atomic_bool terminateThread_{false};
  std::jthread solver_worker_;

  std::shared_ptr<DummyObserver> visualizerPtr_;

  vector_t mpcJointKp_;
  vector_t mpcJointKd_;
  vector_t otherJointKp_;
  vector_t otherJointKd_;

  scalar_t previousObservationTime_{0.0};  ///< Previous sim time for computing actual dt
  vector_t latestPolicyInput_;             ///< Latest MPC policy input

  std::string controlMode_{"WB_MPC"};            ///< Active control mode (see setControlMode)
  std::vector<scalar_t> nominalJointPositions_;  ///< Nominal positions for JOINT_PD mode
  // Hold of WB_MPC for a post-reset policy (requestMpcResetAndHold). Armed by the mode switch, read in the control loop.
  std::atomic<bool> awaitingPostResetPolicy_{false};
  std::atomic<bool> holdGravityComp_{false};  ///< the held action is GRAVITY_COMP (else JOINT_PD)

  // SAFETY damped decay, captured on the first cycle in the mode.
  scalar_t safetyDecayTimeConstant_{0.5};  ///< [s]
  scalar_t safetyDecayStartTime_{-1.0};    ///< observation time at entry, < 0: not yet captured
  vector_t safetyHoldMpcJointPositions_;
  vector_t safetyHoldOtherJointPositions_;

  std::string pdGainsFile_;
  const ModelSettings& modelSettings_;
  std::filesystem::file_time_type pdGainsLastWriteTime_;
  size_t fileCheckCounter_{0};

  // ROS topic state for real-time PD gains updates
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr pdGainsSubscription_;
  std::mutex pdGainsPendingMutex_;
  std::string pdGainsPendingYamlContent_;
  std::atomic<bool> hasNewPdGainsTopicData_{false};
};

}  // namespace ocs2::humanoid
