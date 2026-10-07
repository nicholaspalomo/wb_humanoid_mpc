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

#pragma once

#include <cstddef>
#include <limits>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "robot_core/Types.h"

namespace robot::model {

/** The limits of one revolute or prismatic joint of the URDF, and its joint index. */
struct JointDescription {
  joint_index_t id = 0;
  scalar_t min_angle = std::numeric_limits<scalar_t>::lowest();
  scalar_t max_angle = std::numeric_limits<scalar_t>::max();
  scalar_t max_velocity = std::numeric_limits<scalar_t>::max();
  scalar_t max_effort = std::numeric_limits<scalar_t>::max();

  friend std::ostream& operator<<(std::ostream& os, const JointDescription& joint);
};

/**
 * The joints of a robot as its URDF describes them: the revolute and prismatic joints, numbered 0 to getNumJoints() - 1
 * in the order of their names, which is the joint index of RobotState and RobotJointAction. Built once during setup;
 * the name lookups hash, so the realtime loop works with the indices instead. Immutable, and so safe to share between
 * threads.
 */
class RobotDescription {
 public:
  /** The fewest and the most joints a description may have: the bounds of a JointIdMap. */
  static constexpr size_t kMinJoints = 2;
  static constexpr size_t kMaxJoints = 255;

  /**
   * Reads the URDF at `urdfPath`. NotFound when there is no such file; InvalidArgument when it does not parse or has
   * fewer than kMinJoints or more than kMaxJoints revolute and prismatic joints.
   */
  static absl::StatusOr<RobotDescription> Create(absl::string_view urdfPath);

  RobotDescription() = delete;
  RobotDescription(const RobotDescription&) = delete;
  RobotDescription& operator=(const RobotDescription&) = delete;
  /** Movable, so that Create() can return one: a moved-from description is left empty. */
  RobotDescription(RobotDescription&&) = default;
  RobotDescription& operator=(RobotDescription&&) = delete;
  virtual ~RobotDescription() = default;

  /** 0 to getNumJoints() - 1, in order. */
  const std::vector<joint_index_t>& getJointIndices() const { return joint_indices_; }
  /** The joint names in joint-index order: getJointNames()[i] is the joint of index i. */
  const std::vector<std::string>& getJointNames() const { return joint_names_; }

  bool containsJoint(absl::string_view jointName) const;

  /** The joint called `jointName`, or nullptr when the URDF has no revolute or prismatic joint of that name. */
  const JointDescription* absl_nullable findJointDescription(absl::string_view jointName) const;

  size_t getNumJoints() const { return joint_names_.size(); }
  const std::string& getURDFPath() const { return urdf_path_; }
  /** The file name of the URDF, without its directory. */
  std::string getURDFName() const;

  /** The index of `jointName`, or nullopt when the URDF has no joint of that name. */
  std::optional<joint_index_t> findJointIndex(absl::string_view jointName) const;
  /** The indices of `jointNames`, in their order; NotFound naming the first joint the URDF does not have. */
  absl::StatusOr<std::vector<joint_index_t>> findJointIndices(const std::vector<std::string>& jointNames) const;

  /**
   * The index of `jointName`, which must be a joint of the description (containsJoint()); a name it does not have ends
   * the process. For names that come from a file, use findJointIndex().
   */
  joint_index_t getJointIndex(absl::string_view jointName) const;
  /** getJointIndex() of each name, in their order. For names that come from a file, use findJointIndices(). */
  std::vector<joint_index_t> getJointIndices(const std::vector<std::string>& jointNames) const;

  /** The name of joint `jointIndex`, which must be below getNumJoints(); another index ends the process. */
  const std::string& getJointName(joint_index_t jointIndex) const;

  friend std::ostream& operator<<(std::ostream& os, const RobotDescription& robot);

 private:
  /** The description of `urdfPath`'s joints, given in joint-index order. */
  RobotDescription(std::string urdfPath, std::vector<std::string> jointNames, std::vector<JointDescription> joints);

  std::string urdf_path_;
  std::vector<std::string> joint_names_;      // by joint index
  std::vector<JointDescription> joints_;      // by joint index
  std::vector<joint_index_t> joint_indices_;  // 0 to getNumJoints() - 1
  absl::flat_hash_map<std::string, joint_index_t> joint_index_by_name_;

  // Additional sensors like IMUs can be added here later.
};

}  // namespace robot::model
