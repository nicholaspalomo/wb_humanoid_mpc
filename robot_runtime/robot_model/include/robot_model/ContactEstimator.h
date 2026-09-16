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

#include <robot_model/RobotState.h>

#include <string>
#include <vector>

namespace robot::model {

/**
 * Measured contact state of the contact points of a controller.
 *
 * A controller asks its estimator once per control cycle, from the control thread and after the robot state has been
 * updated, which of its contact points are touching the environment. The answer is the controller's measured contact
 * state: the observation mode handed to the MPC and the criterion for which planned contact wrenches the inverse
 * dynamics may project into joint torques (a foot in the air cannot transmit a wrench, whatever the plan says).
 *
 * Implementations are free to use the robot state, the interface they were built with (a simulator, sensors), or any
 * history they keep. In simulation the CheaterSimContactEstimator reads the physics engine's ground truth; the
 * RobotStateContactEstimator hands back the flags the hardware interface wrote into the RobotState.
 */
class ContactEstimator {
 public:
  virtual ~ContactEstimator() = default;

  /**
   * One flag per contact point of the controller, in the controller's order: true where the point is touching.
   * @param robotState The robot state of the current control cycle.
   */
  virtual std::vector<bool> estimateContactFlags(const RobotState& robotState) = 0;

  /** Name for logs and diagnostics. */
  virtual std::string getName() const = 0;
};

}  // namespace robot::model
