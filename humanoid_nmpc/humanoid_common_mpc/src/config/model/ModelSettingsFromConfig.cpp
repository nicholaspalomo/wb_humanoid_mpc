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

#include "humanoid_common_mpc/config/model/ModelSettingsFromConfig.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_mpc_config/contact_implicit_config.nproto.h"
#include "humanoid_mpc_config/model_settings_config.nproto.h"
#include "humanoid_mpc_config/model_settings_config.nproto.pb.h"
#include "humanoid_mpc_config/model_settings_config.pb.h"
#include "humanoid_mpc_config/nominal_foothold_config.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "nproto/Textproto.h"

namespace ocs2::humanoid {
namespace {

// ContactImplicitConfig holds nothing but its seven scalars, so a field added to it without a line in
// contactImplicitFromConfig() - which would leave it at its default whatever the file says - changes its size and fails
// to compile here.
static_assert(sizeof(ModelSettings::ContactImplicitConfig) == 7 * sizeof(scalar_t),
              "every field of ModelSettings::ContactImplicitConfig needs its line in contactImplicitFromConfig()");

/** A stance_constraint name, and the two settings of the zero_velocity constraint it selects. */
struct StanceConstraintEntry {
  // NOLINTNEXTLINE(totw-view-member): every entry is a string literal of the constexpr table below, alive for the whole program.
  absl::string_view name;
  bool constrainOrientation = true;
  bool constrainYawRateAboutContactNormal = false;
};

// The single place a stance constraint is added; stanceConstraintName() is its inverse.
constexpr std::array<StanceConstraintEntry, 3> kStanceConstraints = {{
    {.name = kPositionStanceConstraint, .constrainOrientation = false, .constrainYawRateAboutContactNormal = false},
    {.name = kPositionAndTiltStanceConstraint, .constrainOrientation = true, .constrainYawRateAboutContactNormal = false},
    {.name = kPositionAndOrientationStanceConstraint, .constrainOrientation = true, .constrainYawRateAboutContactNormal = true},
}};

/** The index of each name of `names` in it. */
absl::flat_hash_map<std::string, size_t> indexMap(const std::vector<std::string>& names) {
  absl::flat_hash_map<std::string, size_t> indices;
  for (size_t i = 0; i < names.size(); ++i) {
    indices[names[i]] = i;
  }
  return indices;
}

/** The joints of the URDF the MPC keeps: every joint that model_settings.fixed_joint_names does not name, in order. */
absl::StatusOr<std::vector<std::string>> activeJointNames(const std::vector<std::string>& fullJointNames,
                                                          const std::vector<std::string>& fixedJointNames,
                                                          bool verbose) {
  // Compared before subtracting: more fixed joints than the URDF has would wrap the unsigned difference around.
  if (fullJointNames.size() <= fixedJointNames.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("[ModelSettings] the URDF has ", fullJointNames.size(), " joints and ", fixedJointNames.size(),
                     " are fixed by model_settings.fixed_joint_names; the MPC needs at least one active joint."));
  }
  std::vector<std::string> active;
  active.reserve(fullJointNames.size() - fixedJointNames.size());
  for (const std::string& joint : fullJointNames) {
    if (std::find(fixedJointNames.begin(), fixedJointNames.end(), joint) == fixedJointNames.end()) {
      active.push_back(joint);
    }
  }
  if (verbose) {
    LOG(INFO) << "[ModelSettings] the " << active.size() << " active MPC joints: " << absl::StrJoin(active, ", ");
  }
  return active;
}

/** Where each MPC joint is among the URDF's joints. */
std::vector<size_t> mpcToFullJointIndices(const std::vector<std::string>& fullJointNames, const std::vector<std::string>& mpcJointNames) {
  const absl::flat_hash_map<std::string, size_t> fullIndices = indexMap(fullJointNames);
  std::vector<size_t> indices;
  indices.reserve(mpcJointNames.size());
  for (const std::string& joint : mpcJointNames) {
    // The MPC joints are the URDF's joints less the fixed ones (activeJointNames()), so every one of them is found.
    const absl::flat_hash_map<std::string, size_t>::const_iterator found = fullIndices.find(joint);
    ABSL_CHECK(found != fullIndices.end()) << "MPC joint " << joint << " is not a joint of the URDF";
    indices.push_back(found->second);
  }
  return indices;
}

/** The index of the arm joint `name` (model_settings.arm_joint_names.`key`) among the MPC joints, or InvalidArgument. */
absl::StatusOr<size_t> armJointIndex(const absl::flat_hash_map<std::string, size_t>& jointIndexMap,
                                     absl::string_view key,
                                     const std::string& name) {
  const absl::flat_hash_map<std::string, size_t>::const_iterator found = jointIndexMap.find(name);
  if (found == jointIndexMap.end()) {
    return absl::InvalidArgumentError(absl::StrCat("[ModelSettings] model_settings.arm_joint_names.", key, " is '", name,
                                                   "', which is not an active joint of the MPC model: a misspelled joint, or one that "
                                                   "model_settings.fixed_joint_names fixes. Name all four arm joints, or none."));
  }
  return found->second;
}

/**
 * InvalidArgument naming model_settings.foot_constraint.`field` unless `weight`, the weight of a soft foot term, is a
 * finite positive number: a weight of 0 switches the term off, a negative one rewards what it holds, and a NaN poisons
 * every solve.
 */
absl::Status checkSoftTermWeight(absl::string_view field, double weight) {
  // Negated so that NaN is refused too.
  if (!(weight > 0.0) || !std::isfinite(weight)) {
    return absl::InvalidArgumentError(absl::StrCat("[ModelSettings] model_settings.foot_constraint.", field, " is ", weight,
                                                   ", but the weight of a soft foot term must be a finite positive number."));
  }
  return absl::OkStatus();
}

}  // namespace

std::vector<std::string> stanceConstraintNames() {
  std::vector<std::string> names;
  names.reserve(kStanceConstraints.size());
  for (const StanceConstraintEntry& entry : kStanceConstraints) names.emplace_back(entry.name);
  return names;
}

absl::string_view stanceConstraintName(const ModelSettings::FootConstraintConfig& config) {
  // Three rows have no yaw-rate row, so the yaw-rate setting counts only beside the orientation rows.
  const bool yawRate = config.constrainOrientation && config.constrainYawRateAboutContactNormal;
  for (const StanceConstraintEntry& entry : kStanceConstraints) {
    if (entry.constrainOrientation == config.constrainOrientation && entry.constrainYawRateAboutContactNormal == yawRate) {
      return entry.name;
    }
  }
  // Unreachable while kStanceConstraints names all three rows; the registry test walks them.
  return kPositionAndTiltStanceConstraint;
}

absl::StatusOr<ModelSettings::FootConstraintConfig> footConstraintFromConfig(
    const mpc_config::ModelSettingsConfig::FootConstraintConfig& config) {
  RETURN_IF_ERROR(checkSoftTermWeight("soft_constraint_weight", config.soft_constraint_weight));
  RETURN_IF_ERROR(checkSoftTermWeight("normal_velocity_soft_constraint_weight", config.normal_velocity_soft_constraint_weight));
  ModelSettings::FootConstraintConfig converted = {
      .positionErrorGain_z = config.position_error_gain_z,
      .orientationErrorGain = config.orientation_error_gain,
      .linearVelocityErrorGain_z = config.linear_velocity_error_gain_z,
      .linearVelocityErrorGain_xy = config.linear_velocity_error_gain_xy,
      .angularVelocityErrorGain = config.angular_velocity_error_gain,
      .linearAccelerationErrorGain_z = config.linear_acceleration_error_gain_z,
      .linearAccelerationErrorGain_xy = config.linear_acceleration_error_gain_xy,
      .angularAccelerationErrorGain = config.angular_acceleration_error_gain,
      .softConstraintWeight = config.soft_constraint_weight,
      .normalVelocitySoftConstraintWeight = config.normal_velocity_soft_constraint_weight,
  };
  for (const StanceConstraintEntry& entry : kStanceConstraints) {
    if (entry.name == config.stance_constraint) {
      converted.constrainOrientation = entry.constrainOrientation;
      converted.constrainYawRateAboutContactNormal = entry.constrainYawRateAboutContactNormal;
      return converted;
    }
  }
  return absl::InvalidArgumentError(
      absl::StrCat("[ModelSettings] model_settings.foot_constraint.stance_constraint is '", config.stance_constraint,
                   "', which is not a stance constraint; valid names are: ", absl::StrJoin(stanceConstraintNames(), ", "), "."));
}

ModelSettings::ContactImplicitConfig contactImplicitFromConfig(const mpc_config::ContactImplicitConfig& config) {
  return {
      .complementarityWeight = config.complementarity_weight,
      .slipWeight = config.slip_weight,
      .heightReference = config.height_reference,
      .velocityReference = config.velocity_reference,
      .angularVelocityReference = config.angular_velocity_reference,
      .penetrationWeight = config.penetration_weight,
      .gapSmoothing = config.gap_smoothing,
  };
}

ModelSettings::NominalFootholdConfig nominalFootholdFromConfig(const mpc_config::NominalFootholdConfig& config) {
  return {.stepWidth = config.step_width};
}

absl::StatusOr<ModelSettings> ModelSettings::Create(const mpc_config::TaskFile& taskFile,
                                                    const std::string& urdfFile,
                                                    const std::string& mpcName,
                                                    bool verbose) {
  const mpc_config::ModelSettingsConfig& config = taskFile.model_settings;
  if (verbose) {
    humanoid_mpc_config::ModelSettingsConfig message;
    mpc_config::ToProto(config, &message);
    LOG(INFO) << "[ModelSettings] Robot Model Settings, the model_settings of the task file:\n" << nproto::WriteTextproto(message);
  }

  ModelSettings settings;
  settings.robotName = config.robot_name;
  settings.verboseCppAd = config.verbose_cpp_ad;
  settings.recompileLibrariesCppAd = config.recompile_libraries_cpp_ad;
  settings.modelFolderCppAd = absl::StrCat("cppad_code_gen/cppad_", mpcName, settings.robotName);
  settings.phaseTransitionStanceTime = config.phase_transition_stance_time;
  settings.fixedJointNames = config.fixed_joint_names;
  settings.contactNames6DoF = config.contact_names_6dof;
  settings.contactParentJointNames = config.contact_parent_joint_names;

  // NOLINTNEXTLINE(exceptions): Pinocchio's URDF parser reports a file it cannot read by throwing; converted to a Status here, once.
  try {
    settings.loadFullJointNames(urdfFile, verbose);
  } catch (const std::exception& error) {  // NOLINT(exceptions): the boundary of the try above.
    return absl::InvalidArgumentError(absl::StrCat("[ModelSettings] cannot read the joints of the URDF ", urdfFile, ": ", error.what()));
  }
  ASSIGN_OR_RETURN(settings.mpcModelJointNames, activeJointNames(settings.fullJointNames, settings.fixedJointNames, verbose));
  settings.mpcModelToFullJointsIndices = mpcToFullJointIndices(settings.fullJointNames, settings.mpcModelJointNames);
  settings.jointIndexMap = indexMap(settings.mpcModelJointNames);
  settings.contactNames = settings.contactNames3DoF;
  settings.contactNames.insert(settings.contactNames.end(), settings.contactNames6DoF.begin(), settings.contactNames6DoF.end());
  settings.mpc_joint_dim = settings.mpcModelJointNames.size();
  settings.full_joint_dim = settings.fullJointNames.size();

  // The arm joints of the procedural arm swing: all four, or none (a legs-only robot such as the EngineAI SA01, whose
  // swing is then disabled). Any other combination is refused, because a robot that names three of the four, misspells
  // one or fixes one out would otherwise walk with a half-built arm swing.
  const mpc_config::ModelSettingsConfig::ArmJointNames& arms = config.arm_joint_names;
  settings.hasArmSwingJoints =
      !(arms.left_shoulder_y.empty() && arms.right_shoulder_y.empty() && arms.left_elbow_y.empty() && arms.right_elbow_y.empty());
  if (settings.hasArmSwingJoints) {
    ASSIGN_OR_RETURN(settings.j_l_shoulder_y_index, armJointIndex(settings.jointIndexMap, "left_shoulder_y", arms.left_shoulder_y));
    ASSIGN_OR_RETURN(settings.j_r_shoulder_y_index, armJointIndex(settings.jointIndexMap, "right_shoulder_y", arms.right_shoulder_y));
    ASSIGN_OR_RETURN(settings.j_l_elbow_y_index, armJointIndex(settings.jointIndexMap, "left_elbow_y", arms.left_elbow_y));
    ASSIGN_OR_RETURN(settings.j_r_elbow_y_index, armJointIndex(settings.jointIndexMap, "right_elbow_y", arms.right_elbow_y));
  } else if (verbose) {
    LOG(INFO) << "[ModelSettings] model_settings.arm_joint_names is not set: the procedural arm swing reference is disabled.";
  }

  ASSIGN_OR_RETURN(settings.footConstraintConfig, footConstraintFromConfig(config.foot_constraint));
  settings.terrainHeight = taskFile.terrain_height;
  settings.contactImplicitConfig = contactImplicitFromConfig(taskFile.contact_implicit);
  settings.nominalFootholdConfig = nominalFootholdFromConfig(taskFile.nominal_foothold);
  return settings;
}

}  // namespace ocs2::humanoid
