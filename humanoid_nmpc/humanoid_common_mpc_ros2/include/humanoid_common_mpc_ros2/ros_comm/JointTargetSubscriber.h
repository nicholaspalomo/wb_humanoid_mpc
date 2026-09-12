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

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include <humanoid_common_mpc/common/Types.h>

namespace ocs2::humanoid {

/**
 * @brief ROS 2 subscriber for real-time joint target position updates from the GUI.
 *
 * Receives YAML-encoded joint name→position maps on `/joint_pd_target_positions`
 * and merges them into the nominal joint position vector used by JOINT_PD mode.
 * Designed to be composed into any class that owns nominal joint positions
 * (e.g. SimFsmBridge).
 */
class JointTargetSubscriber {
 public:
  /**
   * @brief Create the ROS 2 subscription on `/joint_pd_target_positions`.
   * @param node Active ROS 2 node handle.
   */
  void subscribe(rclcpp::Node::SharedPtr node);

  /**
   * @brief Apply any pending target position updates to `nominalPositions`.
   *
   * Call this from the control loop. If new data arrived via the ROS topic,
   * parses the pending YAML string and updates matching entries in
   * `nominalPositions` using the joint-name → index mapping.
   *
   * @param[in,out] nominalPositions Vector of nominal joint positions (indexed by robot joint index).
   * @param[in] jointNames Ordered list of joint names whose indices correspond to `jointIndices`.
   * @param[in] jointIndices Robot joint indices corresponding to each name in `jointNames`.
   * @return true if any positions were updated, false otherwise.
   */
  bool applyPendingUpdates(std::vector<scalar_t>& nominalPositions,
                           const std::vector<std::string>& jointNames,
                           const std::vector<size_t>& jointIndices);

 private:
  // LINT.IfChange(joint_target_topic_name)
  static constexpr const char* kTopicName = "/joint_pd_target_positions";
  // LINT.ThenChange(//humanoid_nmpc/remote_control/remote_control/tk_app/joint_targets_tab.py:joint_target_topic_name)

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
  std::mutex pendingMutex_;
  std::string pendingYamlContent_;
  std::atomic<bool> hasNewData_{false};
};

}  // namespace ocs2::humanoid
