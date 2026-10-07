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

#include "humanoid_mpc_validation/closed_loop/RecordedRobotStates.h"

#include <string>
#include <utility>
#include <vector>

#include "Eigen/Geometry"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"

namespace ocs2::humanoid::validation {
namespace {

// LINT.IfChange(recorded_state_labels)
constexpr char kJointNamesNote[] = "joint_names";
constexpr char kTime[] = "time";
constexpr char kBasePosition[] = "base_position";
constexpr char kBaseQuaternion[] = "base_quaternion_xyzw";
constexpr char kBaseLinearVelocity[] = "base_linear_velocity_local";
constexpr char kBaseAngularVelocity[] = "base_angular_velocity_local";
constexpr char kJointPositions[] = "joint_positions";
constexpr char kJointVelocities[] = "joint_velocities";
constexpr char kContactFlags[] = "contact_flags";
constexpr char kGuiCommand[] = "gui_command";
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_validation/README.md:recorded_state_labels)

absl::StatusOr<const golden_matrix_t* absl_nonnull> requireMatrix(const GoldenFile& golden,
                                                                  absl::string_view label,
                                                                  Eigen::Index rows,
                                                                  Eigen::Index cols) {
  const golden_matrix_t* absl_nullable matrix = golden.find(label);
  if (matrix == nullptr) return absl::InvalidArgumentError(absl::StrCat("[fromGoldenFile] no matrix '", label, "'"));
  if (matrix->rows() != rows || matrix->cols() != cols) {
    return absl::InvalidArgumentError(
        absl::StrCat("[fromGoldenFile] '", label, "' is ", matrix->rows(), " x ", matrix->cols(), ", expected ", rows, " x ", cols));
  }
  return matrix;
}

}  // namespace

RobotStateRecord recordRobotState(const robot::model::RobotState& state,
                                  const robot::model::RobotDescription& description,
                                  const Eigen::Vector4d& guiCommand) {
  RobotStateRecord record;
  record.time = state.getTime();
  record.basePosition = state.getRootPositionInWorldFrame();
  record.baseQuaternion = state.getRootRotationLocalToWorldFrame().coeffs();
  record.baseLinearVelocityLocal = state.getRootLinearVelocityInLocalFrame();
  record.baseAngularVelocityLocal = state.getRootAngularVelocityInLocalFrame();
  const std::vector<robot::joint_index_t>& joints = description.getJointIndices();
  record.jointPositions.resize(static_cast<Eigen::Index>(joints.size()));
  record.jointVelocities.resize(static_cast<Eigen::Index>(joints.size()));
  for (size_t i = 0; i < joints.size(); ++i) {
    record.jointPositions(static_cast<Eigen::Index>(i)) = state.getJointPosition(joints[i]);
    record.jointVelocities(static_cast<Eigen::Index>(i)) = state.getJointVelocity(joints[i]);
  }
  const std::vector<bool>& flags = state.getContactFlags();
  for (size_t contact = 0; contact < 2 && contact < flags.size(); ++contact) record.contactFlags[contact] = flags[contact];
  record.guiCommand = guiCommand;
  return record;
}

absl::StatusOr<robot::model::RobotState> toRobotState(const RobotStateRecord& record,
                                                      const std::vector<std::string>& jointNames,
                                                      const robot::model::RobotDescription& description) {
  // The joints are matched by name: the order of a RobotDescription's joints is not the same in every process.
  if (jointNames.size() != description.getNumJoints()) {
    return absl::InvalidArgumentError(absl::StrCat("[toRobotState] the recording has ", jointNames.size(), " joints, ",
                                                   description.getURDFPath(), " has ", description.getNumJoints()));
  }
  if (record.jointPositions.size() != static_cast<Eigen::Index>(jointNames.size()) ||
      record.jointVelocities.size() != static_cast<Eigen::Index>(jointNames.size())) {
    return absl::InvalidArgumentError("[toRobotState] the record's joint vectors do not match its joint names");
  }
  absl::flat_hash_set<std::string> seen;
  for (const std::string& name : jointNames) {
    if (!description.containsJoint(name)) {
      return absl::InvalidArgumentError(absl::StrCat("[toRobotState] ", description.getURDFPath(), " has no joint '", name,
                                                     "' of the recording (", absl::StrJoin(jointNames, ","), ")"));
    }
    if (!seen.insert(name).second) {
      return absl::InvalidArgumentError(absl::StrCat("[toRobotState] the joint '", name, "' is recorded twice"));
    }
  }
  robot::model::RobotState state(description, /*contactSize=*/2);
  state.setTime(record.time);
  state.setRootPositionInWorldFrame(record.basePosition);
  Eigen::Quaterniond rotation;
  rotation.coeffs() = record.baseQuaternion;
  state.setRootRotationLocalToWorldFrame(rotation);
  state.setRootLinearVelocityInLocalFrame(record.baseLinearVelocityLocal);
  state.setRootAngularVelocityInLocalFrame(record.baseAngularVelocityLocal);
  for (size_t i = 0; i < jointNames.size(); ++i) {
    const robot::joint_index_t joint = description.getJointIndex(jointNames[i]);
    state.setJointPosition(joint, record.jointPositions(static_cast<Eigen::Index>(i)));
    state.setJointVelocity(joint, record.jointVelocities(static_cast<Eigen::Index>(i)));
  }
  state.setContactFlag(/*index=*/0, record.contactFlags[0]);
  state.setContactFlag(/*index=*/1, record.contactFlags[1]);
  return state;
}

GoldenFile toGoldenFile(const RecordedRobotStates& states, GoldenProvenance provenance) {
  const Eigen::Index rows = static_cast<Eigen::Index>(states.records.size());
  const Eigen::Index numJoints = static_cast<Eigen::Index>(states.jointNames.size());
  golden_matrix_t time(rows, 1);
  golden_matrix_t position(rows, 3);
  golden_matrix_t quaternion(rows, 4);
  golden_matrix_t linearVelocity(rows, 3);
  golden_matrix_t angularVelocity(rows, 3);
  golden_matrix_t jointPositions(rows, numJoints);
  golden_matrix_t jointVelocities(rows, numJoints);
  golden_matrix_t contactFlags(rows, 2);
  golden_matrix_t guiCommand(rows, 4);
  for (Eigen::Index row = 0; row < rows; ++row) {
    const RobotStateRecord& record = states.records[static_cast<size_t>(row)];
    time(row, /*col=*/0) = record.time;
    position.row(row) = record.basePosition.transpose();
    quaternion.row(row) = record.baseQuaternion.transpose();
    linearVelocity.row(row) = record.baseLinearVelocityLocal.transpose();
    angularVelocity.row(row) = record.baseAngularVelocityLocal.transpose();
    jointPositions.row(row) = record.jointPositions.transpose();
    jointVelocities.row(row) = record.jointVelocities.transpose();
    contactFlags(row, /*col=*/0) = record.contactFlags[0] ? 1.0 : 0.0;
    contactFlags(row, /*col=*/1) = record.contactFlags[1] ? 1.0 : 0.0;
    guiCommand.row(row) = record.guiCommand.transpose();
  }
  GoldenFile golden;
  golden.provenance = std::move(provenance);
  golden.provenance.notes.emplace_back(kJointNamesNote, absl::StrJoin(states.jointNames, ","));
  golden.entries = {{kTime, time},
                    {kBasePosition, position},
                    {kBaseQuaternion, quaternion},
                    {kBaseLinearVelocity, linearVelocity},
                    {kBaseAngularVelocity, angularVelocity},
                    {kJointPositions, jointPositions},
                    {kJointVelocities, jointVelocities},
                    {kContactFlags, contactFlags},
                    {kGuiCommand, guiCommand}};
  return golden;
}

absl::StatusOr<RecordedRobotStates> fromGoldenFile(const GoldenFile& golden) {
  RecordedRobotStates states;
  bool haveJointNames = false;
  for (const std::pair<std::string, std::string>& note : golden.provenance.notes) {
    if (note.first != kJointNamesNote) continue;
    const std::vector<std::string> names = absl::StrSplit(note.second, ',', absl::SkipEmpty());
    states.jointNames = names;
    haveJointNames = true;
  }
  if (!haveJointNames) return absl::InvalidArgumentError(absl::StrCat("[fromGoldenFile] no '", kJointNamesNote, "' note"));
  const golden_matrix_t* absl_nullable time = golden.find(kTime);
  if (time == nullptr || time->cols() != 1) return absl::InvalidArgumentError(absl::StrCat("[fromGoldenFile] no '", kTime, "' column"));
  const Eigen::Index rows = time->rows();
  const Eigen::Index numJoints = static_cast<Eigen::Index>(states.jointNames.size());
  absl::StatusOr<const golden_matrix_t* absl_nonnull> position = requireMatrix(golden, kBasePosition, rows, /*cols=*/3);
  absl::StatusOr<const golden_matrix_t* absl_nonnull> quaternion = requireMatrix(golden, kBaseQuaternion, rows, /*cols=*/4);
  absl::StatusOr<const golden_matrix_t* absl_nonnull> linearVelocity = requireMatrix(golden, kBaseLinearVelocity, rows, /*cols=*/3);
  absl::StatusOr<const golden_matrix_t* absl_nonnull> angularVelocity = requireMatrix(golden, kBaseAngularVelocity, rows, /*cols=*/3);
  absl::StatusOr<const golden_matrix_t* absl_nonnull> jointPositions = requireMatrix(golden, kJointPositions, rows, numJoints);
  absl::StatusOr<const golden_matrix_t* absl_nonnull> jointVelocities = requireMatrix(golden, kJointVelocities, rows, numJoints);
  absl::StatusOr<const golden_matrix_t* absl_nonnull> contactFlags = requireMatrix(golden, kContactFlags, rows, /*cols=*/2);
  absl::StatusOr<const golden_matrix_t* absl_nonnull> guiCommand = requireMatrix(golden, kGuiCommand, rows, /*cols=*/4);
  for (const absl::StatusOr<const golden_matrix_t* absl_nonnull>* absl_nonnull matrix :
       {&position, &quaternion, &linearVelocity, &angularVelocity, &jointPositions, &jointVelocities, &contactFlags, &guiCommand}) {
    if (!matrix->ok()) return matrix->status();
  }
  states.records.resize(static_cast<size_t>(rows));
  for (Eigen::Index row = 0; row < rows; ++row) {
    RobotStateRecord& record = states.records[static_cast<size_t>(row)];
    record.time = (*time)(row, /*col=*/0);
    record.basePosition = (*position)->row(row).transpose();
    record.baseQuaternion = (*quaternion)->row(row).transpose();
    record.baseLinearVelocityLocal = (*linearVelocity)->row(row).transpose();
    record.baseAngularVelocityLocal = (*angularVelocity)->row(row).transpose();
    record.jointPositions = (*jointPositions)->row(row).transpose();
    record.jointVelocities = (*jointVelocities)->row(row).transpose();
    record.contactFlags = {(**contactFlags)(row, /*col=*/0) != 0.0, (**contactFlags)(row, /*col=*/1) != 0.0};
    record.guiCommand = (*guiCommand)->row(row).transpose();
  }
  return states;
}

}  // namespace ocs2::humanoid::validation
