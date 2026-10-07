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

#include <array>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "absl/status/statusor.h"

#include "humanoid_mpc_validation/io/GoldenIo.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotState.h"

namespace ocs2::humanoid::validation {

/**
 * The measured robot at one solve of a closed-loop run, with the GUI command of that moment: what the solve benchmark
 * replays. It is the RobotState the MRT joint controller reads, so it does not depend on the MPC's state layout, and the
 * states recorded from the Euler formulation drive the quaternion one through the same observation path.
 */
struct RobotStateRecord {
  double time = 0.0;                                                   ///< [s]
  Eigen::Vector3d basePosition = Eigen::Vector3d::Zero();              ///< [m] world frame
  Eigen::Vector4d baseQuaternion = Eigen::Vector4d::UnitW();           ///< base to world, coefficients (x, y, z, w)
  Eigen::Vector3d baseLinearVelocityLocal = Eigen::Vector3d::Zero();   ///< [m/s] base frame
  Eigen::Vector3d baseAngularVelocityLocal = Eigen::Vector3d::Zero();  ///< [rad/s] base frame
  Eigen::VectorXd jointPositions;                                      ///< every joint of the description, in its order
  Eigen::VectorXd jointVelocities;
  std::array<bool, 2> contactFlags{false, false};
  Eigen::Vector4d guiCommand = Eigen::Vector4d::Zero();  ///< GuiVelocityCommand::message
};

/** A recording: the joint names of the description, in the order of the records' joint vectors. */
struct RecordedRobotStates {
  std::vector<std::string> jointNames;
  std::vector<RobotStateRecord> records;
};

/** The record of `state` (every joint of `description`) with the GUI command `guiCommand`. */
RobotStateRecord recordRobotState(const robot::model::RobotState& state,
                                  const robot::model::RobotDescription& description,
                                  const Eigen::Vector4d& guiCommand);

/**
 * The RobotState of `record` for `description`, its joints matched by name (`jointNames`, the order of the record's
 * joint vectors): the order of a RobotDescription's joints differs from one process to the next. InvalidArgument when
 * the recording's joints are not exactly the description's.
 */
absl::StatusOr<robot::model::RobotState> toRobotState(const RobotStateRecord& record,
                                                      const std::vector<std::string>& jointNames,
                                                      const robot::model::RobotDescription& description);

/** The recording as a golden file: one matrix per field, a row per record, the joint names in a note. */
GoldenFile toGoldenFile(const RecordedRobotStates& states, GoldenProvenance provenance);

/** The recording of a golden file written by toGoldenFile(); InvalidArgument naming what is missing or mis-sized. */
absl::StatusOr<RecordedRobotStates> fromGoldenFile(const GoldenFile& golden);

}  // namespace ocs2::humanoid::validation
