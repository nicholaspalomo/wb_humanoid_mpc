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

#include <humanoid_wb_mpc/common/WBAccelMpcRobotModel.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <ocs2_ros2_interfaces/mrt/DummyObserver.h>
#include <robot_model/ContactEstimator.h>
#include <robot_model/ControllerBase.h>
#include "humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h"
#include "robot_model/RobotDescription.h"

#include <atomic>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

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
   * Destructor.
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
   * Contact flags the MPC policy being executed plans for `time`. Empty until a policy has been activated, and again
   * after an MPC reset until the next policy arrives. Call from the thread that runs computeJointControlAction(): the
   * policy is not thread-safe.
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

  const vector_t& getLatestPolicyInput() const { return latestPolicyInput_; }
  const CommandData& getCommandData() const { return mcpMrtInterface_.getCommand(); }

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

  MPC_MRT_Interface mcpMrtInterface_;
  std::shared_ptr<::robot::model::ContactEstimator> contactEstimator_;
  contact_flag_t measuredContactFlags_{};     // of the last control cycle, from contactEstimator_
  std::atomic<bool> policyActivated_{false};  // a policy has been swapped in since the last reset

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
