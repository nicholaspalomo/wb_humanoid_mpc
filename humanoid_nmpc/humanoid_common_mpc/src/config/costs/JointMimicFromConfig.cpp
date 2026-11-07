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

#include "humanoid_common_mpc/config/costs/JointMimicFromConfig.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/Types.h"

#include "humanoid_common_mpc/common/BipedFeet.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/TwoContactCount.h"
#include "humanoid_mpc_config/mimic_joints_config.nproto.h"

namespace ocs2::humanoid {
namespace {

using MimicJoint = mpc_config::MimicJointsConfig::MimicJoint;

/** A number of a mimic joint block and the setting it becomes. */
struct MimicNumberField {
  const char* absl_nonnull name;
  double MimicJoint::*absl_nonnull field;
  scalar_t JointMimicSettings::*absl_nonnull setting;
};

constexpr std::array<MimicNumberField, 3> kMimicNumberFields = {{
    {.name = "multiplier",
     .field = &MimicJoint::multiplier,
     .setting = &JointMimicSettings::multiplier},
    {.name = "position_gain",
     .field = &MimicJoint::position_gain,
     .setting = &JointMimicSettings::positionGain},
    {.name = "velocity_gain",
     .field = &MimicJoint::velocity_gain,
     .setting = &JointMimicSettings::velocityGain},
}};

/** The settings of the mimic joint `block` (mimic_joints.<name>). */
absl::StatusOr<JointMimicSettings> mimicJointFromConfig(
    const MimicJoint& block, absl::string_view name) {
  JointMimicSettings settings;
  settings.parentJointName = block.parent_joint_name;
  settings.childJointName = block.child_joint_name;
  for (const MimicNumberField& entry : kMimicNumberFields) {
    const double value = block.*entry.field;
    if (!std::isfinite(value)) {
      return absl::InvalidArgumentError(
          absl::StrCat("mimic_joints.", name, ".", entry.name, " is ", value,
                       ", but it must be finite."));
    }
    settings.*entry.setting = value;
  }
  return settings;
}

}  // namespace

absl::StatusOr<BipedFeetArray<JointMimicSettings>> kneeMimicJointsFromConfig(
    const mpc_config::MimicJointsConfig& mimicJoints, size_t numContacts) {
  // One knee mimic block per leg: left_knee and right_knee.
  RETURN_IF_ERROR(
      checkTwoContactCount(
          /*component=*/"The knee joint mimic (mimic_joints.left_knee and "
                        "right_knee)",
          numContacts));
  ASSIGN_OR_RETURN(JointMimicSettings left,
                   mimicJointFromConfig(mimicJoints.left_knee, "left_knee"));
  ASSIGN_OR_RETURN(JointMimicSettings right,
                   mimicJointFromConfig(mimicJoints.right_knee, "right_knee"));
  return BipedFeetArray<JointMimicSettings>{std::move(left), std::move(right)};
}

absl::StatusOr<BipedFeetArray<JointMimicSettings>>
kneeMimicKinematicJointsFromConfig(
    const mpc_config::MimicJointsConfig& mimicJoints, size_t numContacts) {
  constexpr BipedFeetArray<absl::string_view> kBlocks = {"left_knee",
                                                         "right_knee"};
  ASSIGN_OR_RETURN(BipedFeetArray<JointMimicSettings> settings,
                   kneeMimicJointsFromConfig(mimicJoints, numContacts));
  for (size_t leg = 0; leg < settings.size(); ++leg) {
    if (settings[leg].velocityGain != 0.0) {
      return absl::InvalidArgumentError(
          absl::StrCat("mimic_joints.", kBlocks[leg], ".velocity_gain is ",
                       settings[leg].velocityGain,
                       ", but the centroidal MPC's mimic constraint holds the "
                       "joint position alone: set "
                       "it to 0 or leave it out."));
    }
  }
  return settings;
}

}  // namespace ocs2::humanoid
