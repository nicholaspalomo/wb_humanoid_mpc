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

#include <ocs2_mpc/MPC_MRT_Interface.h>

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"

#include <ocs2_ros2_interfaces/mrt/DummyObserver.h>
#include <robot_model/ControllerBase.h>
#include "humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h"
#include "robot_model/RobotDescription.h"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

namespace ocs2::humanoid {

class CentroidalMpcMrtJointController final : public ::robot::model::ControlBase {
 public:
  /**
   * Constructor.
   *
   * @param [in] mpc: The underlying MPC class to be used.
   * @param [in] topicPrefix: The robot's name.
   * @param [in] mpcDesiredFrequency: The max frequency to run the mpc at.
   */
  CentroidalMpcMrtJointController(const ::robot::model::RobotDescription& robotDescription,
                                  const ModelSettings& modelSettings,
                                  const CentroidalMpcRobotModel<scalar_t>& mpcRobotModel,
                                  MPC_BASE& mpc,
                                  PinocchioInterface pinocchioInterface,
                                  scalar_t mpcDesiredFrequency = -1,
                                  std::shared_ptr<DummyObserver> rVizVisualizerPtr = nullptr,
                                  const std::string& pdGainsFile = "",
                                  const MpcRobotModelBase<scalar_t>* effectiveMpcRobotModel = nullptr);

  /**
   * Destructor.
   */
  ~CentroidalMpcMrtJointController();

  bool ready() {
    mcpMrtInterface_.updatePolicy();
    return mcpMrtInterface_.initialPolicyReceived();
  }
  bool ready() const { return mcpMrtInterface_.initialPolicyReceived(); }

  /**
   * Handles the low level controller loop that updates the mpc observation, reads out the latest policy and sets the joint control action.
   */

  void computeJointControlAction(scalar_t time,
                                 const ::robot::model::RobotState& robotState,
                                 ::robot::model::RobotJointAction& robotJointAction) override;

  void startMpcThread(const ::robot::model::RobotState& initRobotState);

  void loadPdGains(const std::string& pdGainsFile);

  /**
   * Subscribe to the /pd_gains_updates ROS topic for real-time
   * PD gain updates from the GUI (without writing to joint_pd_gains.yaml).
   */
  void subscribePdGains(rclcpp::Node::SharedPtr node);

  /**
   * @brief Set the active control mode. When set to "JOINT_PD", the controller
   *        computes Pinocchio-based gravity compensation + PD tracking to nominal positions.
   */
  void setControlMode(std::string_view mode) {
    std::string newMode(mode);
    if (newMode != controlMode_) {
      if ((controlMode_ == "ZERO_TORQUE" || controlMode_ == "JOINT_PD" || controlMode_ == "GRAVITY_COMP") &&
          (newMode == "WB_MPC" || newMode == "MPC_ACTIVE")) {
        requestMpcReset();
        transitionCounter_ = 0;  // Reset for transition diagnostics
        if (mpcEntryBlendTime_ > 0.0) {
          // Hold the action of the mode we come from until a policy solved after the reset is active, then ramp into
          // the MPC action. A passive mode without a posture hold (ZERO_TORQUE) is held like JOINT_PD.
          entryHoldGravityComp_.store(controlMode_ == "GRAVITY_COMP");
          entryBlendStartTime_.store(-1.0);
          awaitingPostResetPolicy_.store(true);
        }
      }
      controlMode_ = newMode;
    }
  }
  const std::string& getControlMode() const { return controlMode_; }

  /**
   * @brief Request an asynchronous MPC reset. The solver thread will reset the MPC
   *        to a stable trajectory from the current observation on its next iteration.
   *        Use this when external conditions change (e.g. gantry lock/unlock) to
   *        prevent the solver from using a stale warm-start.
   */
  void requestMpcReset() { resetMpcRequested_.store(true); }

  /**
   * @brief Set nominal joint positions for JOINT_PD mode.
   */
  void setNominalJointPositions(const std::vector<scalar_t>& positions) { nominalJointPositions_ = positions; }

  const ocs2::SystemObservation& getCurrentObservation() const { return currentMpcObservation_; }

  /**
   * Contact flags the MPC policy being executed plans for `time`. Empty until a policy has been activated, and again
   * after an MPC reset until the next policy arrives. Call from the thread that runs computeJointControlAction(): the
   * policy is not thread-safe.
   */
  std::optional<contact_flag_t> getPlannedContactFlags(scalar_t time) const;
  const vector_t& getLatestPolicyInput() const { return latestPolicyInput_; }
  const CommandData& getCommandData() const { return mcpMrtInterface_.getCommand(); }

  /**
   * @brief Enable/disable using gravity compensation instead of full inverse dynamics feedforward torques in WB_MPC mode.
   */
  void setUseGravityCompFeedforward(bool enable) { useGravityCompFeedforward_ = enable; }
  bool getUseGravityCompFeedforward() const { return useGravityCompFeedforward_; }

  /**
   * Duration [s] of the hand-over into WB_MPC (task.yaml `mpcEntryBlendTime`, 0 disables and keeps the immediate switch).
   * When enabled, entering WB_MPC from a passive mode keeps sending that mode's action until the first policy solved after
   * the entry reset has been activated (the MRT keeps the pre-reset policy until then), and then blends targets, gains
   * and feedforward torques linearly from the held action to the MPC action over this duration. Without it the
   * feedforward jumps in one cycle from the gravity term of the passive mode to the inverse dynamics of a policy that
   * was solved against the pre-reset reference.
   */
  void setMpcEntryBlendTime(scalar_t seconds) { mpcEntryBlendTime_ = std::max(0.0, seconds); }
  scalar_t getMpcEntryBlendTime() const { return mpcEntryBlendTime_; }
  /** True while the entry into WB_MPC is being held or blended (for tests and diagnostics). */
  bool isEnteringMpc() const { return awaitingPostResetPolicy_.load() || entryBlendStartTime_.load() >= 0.0; }

 private:
  /**
   * Handles the MPC solver thread.
   */
  void solverWorker();

  /**
   * Method to convert the latest observation msg to a stable desired trajectory (current position, zero velocity and
   * acceleration)
   *
   * @param [in] msg: The observation message.
   */
  TargetTrajectories currentObservationToResetTrajectory(const SystemObservation& currentMpcObservation);

  void updateMpcState(vector_t& mpcState, const ::robot::model::RobotState& robotState);
  void updateMpcObservation(ocs2::SystemObservation& mpcObservation, const ::robot::model::RobotState& robotState);

  /** JOINT_PD action: PD to the nominal posture with gravity compensation feedforward on the MPC joints. */
  void fillJointPdAction(const ::robot::model::RobotState& robotState, ::robot::model::RobotJointAction& robotJointAction);
  /** GRAVITY_COMP action: gravity compensation with light damping, joints move compliantly. */
  void fillGravityCompAction(const ::robot::model::RobotState& robotState, ::robot::model::RobotJointAction& robotJointAction);
  /** The action held while entering WB_MPC: that of the mode the controller came from. */
  void fillEntryHoldAction(const ::robot::model::RobotState& robotState, ::robot::model::RobotJointAction& robotJointAction);
  /** Blends the MPC action in `robotJointAction` with the held action according to the entry ramp, if one is running. */
  void applyEntryBlend(const ::robot::model::RobotState& robotState, ::robot::model::RobotJointAction& robotJointAction);

  MPC_MRT_Interface mcpMrtInterface_;
  std::atomic<bool> policyActivated_{false};  // a policy solved after the last reset has been swapped in
  // Solves completed by the solver thread since its last reset. resetMpcNode() does not clear the MRT policy buffers, so a
  // policy swapped in right after a reset may still be the pre-reset one; only a swap after a post-reset solve activates.
  std::atomic<int> solvesSinceReset_{0};

  PinocchioInterface pinocchioInterface_;
  ocs2::SystemObservation currentMpcObservation_;
  std::unique_ptr<CentroidalMpcRobotModel<scalar_t>> mpcRobotModelPtr_;
  /// Owned clone of the model matching the OCP input layout (basis-vector decorator or wrench model). Every input read or
  /// write goes through it; contact wrenches must be read with the state-aware ...InWorldFrame accessors because the
  /// input-only accessors return the LOCAL contact-frame wrench in basis-vector mode.
  std::unique_ptr<MpcRobotModelBase<scalar_t>> effectiveModelPtr_;
  std::vector<size_t> mpcJointIndices_;
  std::vector<size_t> otherJointIndices_;

  size_t mpcDeltaTMicroSeconds_;
  bool realtime_;  // True if MPC is to be run as fast as possible

  std::atomic_bool terminateThread_{false};
  std::atomic_bool resetMpcRequested_{false};
  std::jthread solver_worker_;

  std::shared_ptr<DummyObserver> visualizerPtr_;

  vector_t mpcJointKp_;
  vector_t mpcJointKd_;
  vector_t mpcJointTorqueLimit_;
  vector_t otherJointKp_;
  vector_t otherJointKd_;
  vector_t otherJointTorqueLimit_;

  std::string controlMode_{"WB_MPC"};            ///< Active control mode (JOINT_PD, WB_MPC, etc.)
  std::vector<scalar_t> nominalJointPositions_;  ///< Nominal positions for JOINT_PD mode
  scalar_t previousObservationTime_{0.0};        ///< Previous sim time for computing actual dt
  vector_t latestPolicyInput_;                   ///< Latest MPC policy input (e.g. contact forces, joint accelerations)
  size_t transitionCounter_{100};                ///< Counts cycles since last WB_MPC mode entry (starts past threshold)

  std::string pdGainsFile_;
  std::vector<std::string> mpcModelJointNames_;
  std::vector<std::string> fixedJointNames_;
  std::filesystem::file_time_type pdGainsLastWriteTime_;
  size_t fileCheckCounter_{0};

  bool useGravityCompFeedforward_{false};  ///< When true, use gravity comp instead of full ID torques in WB_MPC mode

  // Hand-over into WB_MPC (setMpcEntryBlendTime). Written from the mode-switch caller, read in the control loop.
  scalar_t mpcEntryBlendTime_{0.0};                   ///< [s] 0: immediate switch (default)
  std::atomic<bool> awaitingPostResetPolicy_{false};  ///< holding the previous mode's action until a post-reset policy is active
  std::atomic<bool> entryHoldGravityComp_{false};     ///< the held action is GRAVITY_COMP (else JOINT_PD)
  std::atomic<scalar_t> entryBlendStartTime_{-1.0};   ///< observation time the ramp started at, < 0: no ramp running

  // ROS topic state for real-time PD gains updates
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr pdGainsSubscription_;
  std::mutex pdGainsPendingMutex_;
  std::string pdGainsPendingYamlContent_;
  std::atomic<bool> hasNewPdGainsTopicData_{false};

  /**
   * @brief Compute per-joint gravity compensation torques via Pinocchio.
   * Uses nonLinearEffects with zero velocity for pure gravity torques.
   */
  vector_t computeGravityCompensation(const ::robot::model::RobotState& robotState);
};

}  // namespace ocs2::humanoid
