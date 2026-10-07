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

#include "robot_model/RobotDescription.h"

#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "urdfdom/urdf_parser/urdf_parser.h"

namespace robot::model {
namespace {

/** urdf::parseURDF() of `content`; InvalidArgument naming `urdfPath` when it does not parse. */
absl::StatusOr<urdf::ModelInterfaceSharedPtr> parseUrdf(const std::string& content, absl::string_view urdfPath) {
  urdf::ModelInterfaceSharedPtr model;
  try {  // NOLINT(exceptions): urdfdom is not documented not to throw from its parser; converted to a Status at once
    model = urdf::parseURDF(content);
  } catch (const std::exception& error) {  // NOLINT(exceptions): urdfdom; converted to a Status at once
    return absl::InvalidArgumentError(absl::StrCat("Failed to parse URDF file: ", urdfPath, ": ", error.what()));
  }
  if (model == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat("Failed to parse URDF file: ", urdfPath));
  }
  return model;
}

}  // namespace

absl::StatusOr<RobotDescription> RobotDescription::Create(absl::string_view urdfPath) {
  const std::string path(urdfPath);
  std::error_code error;
  if (!std::filesystem::exists(path, error) || error) {
    return absl::NotFoundError(absl::StrCat("URDF file not found: ", urdfPath));
  }
  std::ifstream urdfFile(path);
  if (!urdfFile.is_open()) {
    return absl::NotFoundError(absl::StrCat("URDF file cannot be read: ", urdfPath));
  }
  const std::string content((std::istreambuf_iterator<char>(urdfFile)), std::istreambuf_iterator<char>());

  absl::StatusOr<urdf::ModelInterfaceSharedPtr> model = parseUrdf(content, urdfPath);
  if (!model.ok()) return model.status();

  // The controllable joints, numbered in the order of their names (urdf's joint map is sorted by name); fixed,
  // continuous, floating and planar joints are skipped.
  std::vector<std::string> jointNames;
  std::vector<JointDescription> joints;
  for (const std::pair<const std::string, urdf::JointSharedPtr>& entry : (*model)->joints_) {
    const urdf::Joint& joint = *entry.second;
    if (joint.type != urdf::Joint::REVOLUTE && joint.type != urdf::Joint::PRISMATIC) {
      continue;
    }
    JointDescription description;
    description.id = joints.size();
    if (joint.limits != nullptr) {
      description.min_angle = joint.limits->lower;
      description.max_angle = joint.limits->upper;
      description.max_velocity = joint.limits->velocity;
      description.max_effort = joint.limits->effort;
    }
    jointNames.push_back(entry.first);
    joints.push_back(description);
  }
  if (joints.empty()) {
    return absl::InvalidArgumentError(absl::StrCat("No valid joints found in URDF: ", urdfPath));
  }
  if (joints.size() < kMinJoints || joints.size() > kMaxJoints) {
    return absl::InvalidArgumentError(absl::StrCat("The URDF ", urdfPath, " has ", joints.size(),
                                                   " revolute and prismatic joints; a robot description takes ", kMinJoints, " to ",
                                                   kMaxJoints, "."));
  }
  return RobotDescription(path, std::move(jointNames), std::move(joints));
}

RobotDescription::RobotDescription(std::string urdfPath, std::vector<std::string> jointNames, std::vector<JointDescription> joints)
    : urdf_path_(std::move(urdfPath)), joint_names_(std::move(jointNames)), joints_(std::move(joints)) {
  joint_indices_.reserve(joint_names_.size());
  joint_index_by_name_.reserve(joint_names_.size());
  for (joint_index_t index = 0; index < joint_names_.size(); ++index) {
    joint_indices_.push_back(index);
    joint_index_by_name_.emplace(joint_names_[index], index);
  }
}

bool RobotDescription::containsJoint(absl::string_view jointName) const {
  return joint_index_by_name_.contains(jointName);
}

const JointDescription* absl_nullable RobotDescription::findJointDescription(absl::string_view jointName) const {
  const std::optional<joint_index_t> index = findJointIndex(jointName);
  return index.has_value() ? &joints_[*index] : nullptr;
}

std::optional<joint_index_t> RobotDescription::findJointIndex(absl::string_view jointName) const {
  const absl::flat_hash_map<std::string, joint_index_t>::const_iterator found = joint_index_by_name_.find(jointName);
  if (found == joint_index_by_name_.end()) return std::nullopt;
  return found->second;
}

absl::StatusOr<std::vector<joint_index_t>> RobotDescription::findJointIndices(const std::vector<std::string>& jointNames) const {
  std::vector<joint_index_t> jointIndices;
  jointIndices.reserve(jointNames.size());
  for (const std::string& jointName : jointNames) {
    const std::optional<joint_index_t> index = findJointIndex(jointName);
    if (!index.has_value()) {
      return absl::NotFoundError(absl::StrCat("The URDF ", urdf_path_, " has no revolute or prismatic joint '", jointName, "'"));
    }
    jointIndices.push_back(*index);
  }
  return jointIndices;
}

joint_index_t RobotDescription::getJointIndex(absl::string_view jointName) const {
  const std::optional<joint_index_t> index = findJointIndex(jointName);
  ABSL_CHECK(index.has_value()) << "The URDF " << urdf_path_ << " has no revolute or prismatic joint '" << jointName << "'";
  return *index;
}

std::vector<joint_index_t> RobotDescription::getJointIndices(const std::vector<std::string>& jointNames) const {
  std::vector<joint_index_t> jointIndices;
  jointIndices.reserve(jointNames.size());
  for (const std::string& jointName : jointNames) {
    jointIndices.push_back(getJointIndex(jointName));
  }
  return jointIndices;
}

const std::string& RobotDescription::getJointName(joint_index_t jointIndex) const {
  ABSL_CHECK_LT(jointIndex, joint_names_.size()) << "The URDF " << urdf_path_ << " has no joint of index " << jointIndex;
  return joint_names_[jointIndex];
}

std::string RobotDescription::getURDFName() const {
  return std::filesystem::path(urdf_path_).filename().string();
}

std::ostream& operator<<(std::ostream& os, const JointDescription& joint) {
  os << "JointDescription { " << "id: " << joint.id << ", min_angle: " << joint.min_angle << ", max_angle: " << joint.max_angle
     << ", max_velocity: " << joint.max_velocity << ", max_effort: " << joint.max_effort << " }";
  return os;
}

std::ostream& operator<<(std::ostream& os, const RobotDescription& robot) {
  os << "RobotDescription {\n";
  os << "Generated from URDF: " << robot.getURDFPath() << '\n';
  os << " Joint names and descriptions:\n";
  for (joint_index_t index = 0; index < robot.getNumJoints(); ++index) {
    os << "  {" << robot.joint_names_[index] << ": " << robot.joints_[index] << " }\n";
  }
  os << "}";
  return os;
}

}  // namespace robot::model
