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

#include <cstddef>

#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/types/span.h"

#include "robot_model/RobotDescription.h"
#include "robot_model/RobotJointAction.h"

namespace ocs2::humanoid {

/**
 * The joint actions an MRT joint controller writes on the control thread, reached without a check, and the check the
 * controller's constructor makes once instead. Shared by CentroidalMpcMrtJointController and WBMpcMrtJointController.
 */

/**
 * Ends the process unless every index of `indices` is a joint of `robotDescription`: a RobotJointAction of that
 * description carries an action at each, so jointActionUnchecked() may reach them. For a constructor, once: it builds a
 * RobotJointAction, which allocates. `controller` names the caller in the message.
 */
inline void checkJointIndices(const ::robot::model::RobotDescription& robotDescription,
                              absl::Span<const size_t> indices,
                              const char* absl_nonnull controller) {
  const ::robot::model::RobotJointAction descriptionAction(robotDescription);
  for (const size_t index : indices) {
    ABSL_CHECK(descriptionAction.inRange(index) && descriptionAction[index].has_value())
        << "[" << controller << "] joint " << index << " is not a joint of the robot description";
  }
}

/**
 * The action of joint `index` of `action`, without a check: `index` passed checkJointIndices() for the description
 * `action` was built from, which the robot process makes sure is the controller's (checkControllerRobot()).
 */
inline ::robot::model::JointAction& jointActionUnchecked(::robot::model::RobotJointAction& action, size_t index) {
  return *action[index];  // NOLINT(bugprone-unchecked-optional-access): checked once, by checkJointIndices().
}
inline const ::robot::model::JointAction& jointActionUnchecked(const ::robot::model::RobotJointAction& action, size_t index) {
  return *action[index];  // NOLINT(bugprone-unchecked-optional-access): checked once, by checkJointIndices().
}

}  // namespace ocs2::humanoid
