/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

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

#include "humanoid_common_mpc_ros2/gait/GaitScheduleUpdaterRos2.h"

#include <functional>
#include <memory>
#include <mutex>
#include <utility>

#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc_ros2/gait/ModeSequenceTemplateRos.h"

namespace ocs2::humanoid {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
GaitScheduleUpdaterRos2::GaitScheduleUpdaterRos2(rclcpp::Node::SharedPtr& nodeHandle,
                                                 std::shared_ptr<GaitSchedule> gaitSchedulePtr,
                                                 absl::string_view robotName)
    : GaitScheduleUpdater(std::move(gaitSchedulePtr)) {
  rclcpp::QoS qos(1);
  qos.best_effort();
  mpcModeSequenceSubscriber_ = nodeHandle->create_subscription<ocs2_ros2_msgs::msg::ModeSchedule>(
      absl::StrCat(robotName, "_mpc_mode_schedule"), qos,
      std::bind(&GaitScheduleUpdaterRos2::mpcModeSequenceCallback, this, std::placeholders::_1));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
ModeSequenceTemplate GaitScheduleUpdaterRos2::getReceivedGait() {
  std::lock_guard<std::mutex> lock(receivedGaitMutex_);
  return receivedGait_;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void GaitScheduleUpdaterRos2::mpcModeSequenceCallback(const ocs2_ros2_msgs::msg::ModeSchedule::SharedPtr msg) {
  // The template under the mutex getReceivedGait() takes, the flag through the base class: this subclass keeps no flag
  // of its own, so the base class's reset() drops a gait that arrived before it.
  std::lock_guard<std::mutex> lock(receivedGaitMutex_);
  updateModeSequence(readModeSequenceTemplateMsg(*msg));
}

}  // namespace ocs2::humanoid
