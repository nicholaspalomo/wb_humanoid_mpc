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

#include "humanoid_common_mpc_app/robot/RobotController.h"

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace ocs2::humanoid {

absl::Status checkControllerRobot(const RobotController& controller, const robot::model::RobotDescription& robotDescription) {
  const std::vector<std::string>& controllerJoints = controller.robotJointNames();
  const std::vector<std::string>& robotJoints = robotDescription.getJointNames();
  if (controllerJoints.size() != robotJoints.size()) {
    return absl::InvalidArgumentError(absl::StrCat("the controller was built for a robot of ", controllerJoints.size(),
                                                   " joints, but the robot backend's (", robotDescription.getURDFName(), ") has ",
                                                   robotJoints.size(), ": build both from the same URDF"));
  }
  for (size_t joint = 0; joint < robotJoints.size(); ++joint) {
    if (controllerJoints[joint] != robotJoints[joint]) {
      return absl::InvalidArgumentError(absl::StrCat("joint ", joint, " of the controller is '", controllerJoints[joint],
                                                     "', but of the robot backend's description (", robotDescription.getURDFName(), ") '",
                                                     robotJoints[joint], "': build both from the same URDF"));
    }
  }
  return absl::OkStatus();
}

}  // namespace ocs2::humanoid
