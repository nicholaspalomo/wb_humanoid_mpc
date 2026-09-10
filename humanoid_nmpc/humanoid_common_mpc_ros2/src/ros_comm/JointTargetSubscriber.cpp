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

#include "humanoid_common_mpc_ros2/ros_comm/JointTargetSubscriber.h"

#include <absl/log/log.h>
#include <yaml-cpp/yaml.h>

namespace ocs2::humanoid {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void JointTargetSubscriber::subscribe(rclcpp::Node::SharedPtr node) {
  auto qos = rclcpp::QoS(1).best_effort();
  subscription_ = node->create_subscription<std_msgs::msg::String>(kTopicName, qos, [this](const std_msgs::msg::String::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(pendingMutex_);
    pendingYamlContent_ = msg->data;
    hasNewData_.store(true);
  });
  LOG(INFO) << "[JointTargetSubscriber] Subscribed to " << kTopicName << " topic.";
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

bool JointTargetSubscriber::applyPendingUpdates(std::vector<scalar_t>& nominalPositions,
                                                const std::vector<std::string>& jointNames,
                                                const std::vector<size_t>& jointIndices) {
  if (!hasNewData_.load()) {
    return false;
  }

  std::string yamlContent;
  {
    std::lock_guard<std::mutex> lock(pendingMutex_);
    yamlContent = std::move(pendingYamlContent_);
    pendingYamlContent_.clear();
    hasNewData_.store(false);
  }

  if (yamlContent.empty()) {
    return false;
  }

  bool updated = false;
  try {
    YAML::Node root = YAML::Load(yamlContent);
    if (!root.IsMap()) {
      LOG(WARNING) << "[JointTargetSubscriber] Expected YAML map, got type " << root.Type();
      return false;
    }

    // Build a name→index lookup from the provided joint name/index vectors
    for (const auto& kv : root) {
      const std::string jointName = kv.first.as<std::string>();
      const scalar_t targetPos = kv.second.as<scalar_t>();

      // Search for matching joint name and update corresponding index
      for (size_t i = 0; i < jointNames.size(); ++i) {
        if (jointNames[i] == jointName) {
          size_t robotIndex = jointIndices[i];
          if (robotIndex < nominalPositions.size()) {
            nominalPositions[robotIndex] = targetPos;
            updated = true;
          }
          break;
        }
      }
    }
  } catch (const std::exception& e) {
    LOG(WARNING) << "[JointTargetSubscriber] Failed to parse YAML: " << e.what();
  }

  return updated;
}

}  // namespace ocs2::humanoid
