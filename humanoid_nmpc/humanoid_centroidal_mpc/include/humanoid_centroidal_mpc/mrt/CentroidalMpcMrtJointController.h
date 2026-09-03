/******************************************************************************
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
                                  const std::string& pdGainsFile = "");

  /**
   * Destructor.
   */
  ~CentroidalMpcMrtJointController();

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
   * @brief Set the active control mode. When set to "JOINT_PD", the controller
   *        computes Pinocchio-based gravity compensation + PD tracking to nominal positions.
   */
  void setControlMode(std::string_view mode) {
    std::string newMode(mode);
    if (newMode != controlMode_) {
      if ((newMode == "WB_MPC" || newMode == "MPC_ACTIVE" || newMode == "CENTROIDAL_MPC") &&
          (controlMode_ == "JOINT_PD" || controlMode_ == "ZERO_TORQUE" || controlMode_ == "GRAVITY_COMP")) {
        resetMpcRequested_.store(true);
      }
      controlMode_ = newMode;
    }
  }
  const std::string& getControlMode() const { return controlMode_; }

  /**
   * @brief Set nominal joint positions for JOINT_PD mode.
   */
  void setNominalJointPositions(const std::vector<scalar_t>& positions) { nominalJointPositions_ = positions; }

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

  PinocchioInterface pinocchioInterface_;
  ocs2::SystemObservation currentMpcObservation_;
  std::unique_ptr<CentroidalMpcRobotModel<scalar_t>> mpcRobotModelPtr_;
  std::vector<size_t> mpcJointIndices_;
  std::vector<size_t> otherJointIndices_;

  size_t mpcDeltaTMicroSeconds_;
  bool realtime_;  // True if MPC is to be run as fast as possible

  std::atomic_bool terminateThread_{false};
  std::atomic_bool resetMpcRequested_{false};
  std::jthread solver_worker_;

  std::shared_ptr<DummyObserver> visualizerPtr_;

  vector_t inverse_dynamics_kp_;
  vector_t inverse_dynamics_kd_;

  vector_t mpcJointKp_;
  vector_t mpcJointKd_;
  vector_t mpcJointTorqueLimit_;
  vector_t otherJointKp_;
  vector_t otherJointKd_;
  vector_t otherJointTorqueLimit_;

  std::string controlMode_{"WB_MPC"};  ///< Active control mode (JOINT_PD, WB_MPC, etc.)
  std::vector<scalar_t> nominalJointPositions_;  ///< Nominal positions for JOINT_PD mode

  /**
   * @brief Compute per-joint gravity compensation torques via Pinocchio.
   * Uses nonLinearEffects with zero velocity for pure gravity torques.
   */
  vector_t computeGravityCompensation(const ::robot::model::RobotState& robotState);
};

}  // namespace ocs2::humanoid
