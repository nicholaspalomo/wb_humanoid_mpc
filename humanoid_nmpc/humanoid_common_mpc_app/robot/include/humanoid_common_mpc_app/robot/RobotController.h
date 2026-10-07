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

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact/ContactWrenchGate.h"
#include "humanoid_common_mpc/mrt/ControllerEventSink.h"
#include "humanoid_mpc_config/joint_pd_gains_file.nproto.h"
#include "robot_model/ContactEstimator.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotJointAction.h"
#include "robot_model/RobotState.h"

namespace ocs2::humanoid {

/**
 * The MRT joint controller of a formulation (CentroidalMpcMrtJointController, WBMpcMrtJointController) as the robot
 * process drives it: MrtRobotController adapts either. The realtime loop (RobotProcess) is the same for both.
 *
 * THREADS. The realtime-thread methods are called by the realtime loop only. setPdGains() and pollPdGainsFile() are
 * for the communication thread. startMpc() and policyReady() are for the main thread before the loop runs.
 */
class RobotController {
 public:
  virtual ~RobotController() = default;

  // ------------------------------------------------------------------ the realtime thread

  /** Hands the controller the mode and the JOINT_PD posture of this cycle, in the order the formulation's sim did. */
  virtual void prepareCycle(absl::string_view controlMode, const std::vector<scalar_t>& nominalJointPositions) = 0;
  /** computeJointControlAction() of the controller, at the robot's own clock. */
  virtual void computeJointControlAction(const robot::model::RobotState& robotState, robot::model::RobotJointAction& jointAction) = 0;
  /** The contact flags the policy in use plans for the time of the last observation; nullopt until one is in use. */
  virtual std::optional<contact_flag_t> plannedContactFlags() const = 0;
  /** The contact flags the contact estimator measured in the last cycle. */
  virtual const contact_flag_t& measuredContactFlags() const = 0;
  /** requestMpcReset(): the policy in use carries the robot until a new one is in use. */
  virtual void requestMpcReset() = 0;
  /** requestMpcResetAndHold(): WB_MPC holds the robot in JOINT_PD until a post-reset policy is in use. */
  virtual void requestMpcResetAndHold() = 0;
  /** False while the MPC solver keeps failing or the MPC link is lost (the controller then holds JOINT_PD). */
  virtual bool isMpcHealthy() const = 0;
  virtual void setContactEstimator(std::shared_ptr<robot::model::ContactEstimator> contactEstimator) = 0;
  virtual const ContactWrenchGate::Config& contactWrenchGateConfig() const = 0;
  virtual void setContactWrenchGateConfig(const ContactWrenchGate::Config& config) = 0;

  // ------------------------------------------------------------------ the communication thread

  /** The gains of a whole gains file, as the GUI publishes it on operator/pd_gains; refused gains change nothing. */
  virtual absl::Status setPdGains(const mpc_config::JointPdGainsFile& gains) = 0;
  /** Reloads the controller's gains file when it was written since the last poll. */
  virtual void pollPdGainsFile() = 0;

  // ------------------------------------------------------------------ before the loop

  /**
   * Where the controller's realtime-thread reports go (ControllerEventSink.h): the robot process's RealtimeEventLog,
   * which the communication thread logs, so that the realtime thread writes no log line. The sink outlives the loop;
   * nullptr restores the controller's default sink.
   */
  virtual void setEventSink(ControllerEventSink* absl_nullable eventSink) = 0;

  /** Starts the MPC link from the observation of `initialState` (MpcLink::start()). */
  virtual void startMpc(const robot::model::RobotState& initialState) = 0;
  /** True once a first policy has arrived. */
  virtual bool policyReady() = 0;
  /**
   * The joint names of the robot description the controller was built for, by joint index. Its control cycle writes
   * the joint action it is handed at these indices without a check, so the robot process refuses a controller of another
   * robot (checkControllerRobot()).
   */
  virtual const std::vector<std::string>& robotJointNames() const = 0;
};

/**
 * OK when `controller` was built for the robot `robotDescription` describes: the same joints, by name, at the same
 * joint indices. InvalidArgument naming the first difference otherwise: the controller would write the joint action of
 * that robot at indices that are not its joints. For the thread that sets up the loop, before it runs.
 */
absl::Status checkControllerRobot(const RobotController& controller, const robot::model::RobotDescription& robotDescription);

}  // namespace ocs2::humanoid
