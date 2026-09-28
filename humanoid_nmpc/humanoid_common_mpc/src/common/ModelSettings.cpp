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

#include "humanoid_common_mpc/common/ModelSettings.h"

#include <boost/optional.hpp>
#include <boost/property_tree/info_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <exception>
#include <iterator>
#include <stdexcept>

#include <ocs2_core/misc/LoadData.h>
#include <cassert>

#ifndef CHECK
#define CHECK(cond)                                     \
  do {                                                  \
    if (!(cond)) {                                      \
      throw std::runtime_error("Check failed: " #cond); \
    }                                                   \
  } while (0)
#endif

#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace ocs2::humanoid {

/******************************************************************************************************/
/// Helper functions contained in a local anonymous namespace
/******************************************************************************************************/
namespace {

/**
 * @brief Creates a joint Index map from a list of joint names.
 */

static std::unordered_map<std::string, size_t> createJointIndexMap(const std::vector<std::string>& jointNames, size_t offset = 0) {
  std::unordered_map<std::string, size_t> jointIndexMap;
  for (size_t i = 0; i < jointNames.size(); ++i) {
    jointIndexMap[jointNames[i]] = i + offset;
  }
  return jointIndexMap;
}

static std::vector<std::string> initializeJointNames(const std::vector<std::string>& fullJointNames,
                                                     const std::vector<std::string>& fixedJointNames,
                                                     bool verbose) {
  if (verbose) LOG(INFO) << "Initialize the following active MPC joints: ";
  size_t n_joints = fullJointNames.size() - fixedJointNames.size();
  if (verbose) LOG(INFO) << "Num active joints: " << n_joints;
  std::vector<std::string> mpcModelJointNames;
  if (n_joints > 0) {
    mpcModelJointNames.reserve(n_joints);
  } else {
    throw std::invalid_argument("Number of joints must be greater than zero");
  }
  for (const std::string& joint : fullJointNames) {
    if (std::find(fixedJointNames.begin(), fixedJointNames.end(), joint) == fixedJointNames.end()) {
      // If the joint is not found in fixedJointNames, add it to mpcModelJointNames
      if (verbose) LOG(INFO) << joint;
      mpcModelJointNames.emplace_back(joint);
    }
  }
  return mpcModelJointNames;
}

std::vector<size_t> initializeMpcToFullJointIndices(const std::vector<std::string>& fullJointNames,
                                                    const std::vector<std::string>& mpcModelJointNames) {
  std::unordered_map<std::string, size_t> fullJointIndexMap = createJointIndexMap(fullJointNames);
  // resize, not reserve: reserve only grows the capacity, so indexing the vector below would write outside its
  // (zero) size and the function would return an empty mapping.
  std::vector<size_t> mpcModelJointIndices(mpcModelJointNames.size());
  for (size_t i = 0; i < mpcModelJointNames.size(); ++i) {
    CHECK(fullJointIndexMap.find(mpcModelJointNames[i]) != fullJointIndexMap.end());
    mpcModelJointIndices[i] = fullJointIndexMap[mpcModelJointNames[i]];
  }
  return mpcModelJointIndices;
}

std::vector<std::string> concatenateStringVectors(const std::vector<std::string>& a, const std::vector<std::string>& b) {
  std::vector<std::string> temp_vec(a);
  temp_vec.insert(temp_vec.end(), b.begin(), b.end());
  return temp_vec;
}

// The keys of the task file's `contact_implicit` block. Renaming one here renames it for the loader below, for
// validateContactImplicitConfig() and for the parameter updater at once; the task files have to follow, and
// checkContactImplicitBlockKeys() refuses a file that still carries the old name.
// LINT.IfChange(contact_implicit_yaml_path)
constexpr ModelSettings::ContactImplicitKey kContactImplicitKeys[] = {
    {"complementarityWeight", &ModelSettings::ContactImplicitConfig::complementarityWeight, /*isWeight=*/true},
    {"slipWeight", &ModelSettings::ContactImplicitConfig::slipWeight, /*isWeight=*/true},
    {"penetrationWeight", &ModelSettings::ContactImplicitConfig::penetrationWeight, /*isWeight=*/true},
    {"heightReference", &ModelSettings::ContactImplicitConfig::heightReference, /*isWeight=*/false},
    {"velocityReference", &ModelSettings::ContactImplicitConfig::velocityReference, /*isWeight=*/false},
    {"angularVelocityReference", &ModelSettings::ContactImplicitConfig::angularVelocityReference, /*isWeight=*/false},
    {"gapSmoothing", &ModelSettings::ContactImplicitConfig::gapSmoothing, /*isWeight=*/false},
};
// clang-format off
// LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml:contact_implicit_config, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.yaml:contact_implicit_config, //humanoid_nmpc/humanoid_centroidal_mpc/src/mrt/MpcParameterUpdaterModule.cpp:contact_implicit_updater_keys)
// clang-format on

// ContactImplicitConfig holds nothing but these scalars, so a field added to it without a key here - which would be
// loaded by nothing, validated by nothing and hot-reloaded by nothing - changes its size and fails to compile.
static_assert(sizeof(ModelSettings::ContactImplicitConfig) == std::size(kContactImplicitKeys) * sizeof(scalar_t),
              "every field of ModelSettings::ContactImplicitConfig needs a key in kContactImplicitKeys");

}  // namespace

absl::Span<const ModelSettings::ContactImplicitKey> ModelSettings::contactImplicitKeys() {
  return kContactImplicitKeys;
}

absl::StatusOr<bool> ModelSettings::loadInterfaceVerbose(absl::string_view configFile) {
  const std::string file(configFile);
  boost::property_tree::ptree pt;
  try {
    loadData::readPropertyTree(file, pt);
  } catch (const std::exception& error) {
    return absl::NotFoundError(absl::StrCat("[ModelSettings] cannot read ", kInterfaceVerboseKey, " from ", file, ": ", error.what()));
  }
  const boost::optional<boost::property_tree::ptree&> child = pt.get_child_optional(std::string(kInterfaceVerboseKey));
  if (!child) {
    return false;
  }
  const boost::optional<bool> verbose = child->get_value_optional<bool>();
  if (!verbose) {
    return absl::InvalidArgumentError(absl::StrCat("[ModelSettings] ", kInterfaceVerboseKey, " in ", file, " is '", child->data(),
                                                   "', which is not a bool: write true or false."));
  }
  return *verbose;
}

ModelSettings::ModelSettings(const std::string& configFile, const std::string& urdfFile, const std::string& mpcName, bool verbose) {
  boost::property_tree::ptree pt;
  loadData::readPropertyTree(configFile, pt);

  std::string prefix{"model_settings."};

  if (verbose) {
    LOG(INFO) << "\n #### Robot Model Settings:";
    LOG(INFO) << "\n #### "
                 "============================================================="
                 "================\n";
  }

  loadData::loadPtreeValue(pt, this->robotName, prefix + "robotName", verbose);
  loadData::loadPtreeValue(pt, this->verboseCppAd, prefix + "verboseCppAd", verbose);
  loadData::loadPtreeValue(pt, this->recompileLibrariesCppAd, prefix + "recompileLibrariesCppAd", verbose);
  loadData::loadPtreeValue(pt, this->phaseTransitionStanceTime, prefix + "phaseTransitionStanceTime", verbose);

  loadData::loadPtreeValue(pt, this->j_l_shoulder_y_name, prefix + "armJointNames.left_shoulder_y", verbose);
  loadData::loadPtreeValue(pt, this->j_r_shoulder_y_name, prefix + "armJointNames.right_shoulder_y", verbose);
  loadData::loadPtreeValue(pt, this->j_l_elbow_y_name, prefix + "armJointNames.left_elbow_y", verbose);
  loadData::loadPtreeValue(pt, this->j_r_elbow_y_name, prefix + "armJointNames.right_elbow_y", verbose);
  modelFolderCppAd = absl::StrCat("cppad_code_gen/cppad_", mpcName, robotName);
  // The folder is derived, and the task files used to carry a model_settings.modelFolderCppAd that nothing read. Say so
  // rather than ignore it, so that nobody edits it expecting the libraries to move.
  if (pt.get_child_optional(absl::StrCat(prefix, "modelFolderCppAd"))) {
    LOG(WARNING) << "[ModelSettings] " << prefix << "modelFolderCppAd is not read: the CppAD libraries are built under " << modelFolderCppAd
                 << " (the centroidal MPC adds a sub-folder per contact input parameterization). Delete the key from " << configFile << ".";
  }

  loadData::loadStdVector(configFile, prefix + "fixedJointNames", fixedJointNames, verbose);
  loadData::loadStdVector(configFile, prefix + "contactNames6DoF", contactNames6DoF, verbose);
  loadData::loadStdVector(configFile, prefix + "contactParentJointNames", contactParentJointNames, verbose);

  if (verbose) {
    LOG(INFO) << "Initializing MPC by fixing joints: ";
    for (std::string fixedJoint : fixedJointNames) LOG(INFO) << fixedJoint;
  }

  // Get full joint order from a full pinocchio interface, this removes any joints marked as fix in the urdf.
  PinocchioInterface fullPinocchioInterface = createDefaultPinocchioInterface(urdfFile);
  const pinocchio::Model& model = fullPinocchioInterface.getModel();
  if (verbose) LOG(INFO) << "Full URDF joints: ";
  fullJointNames.reserve(model.njoints - 2);  // Substract universe and root joint
  for (pinocchio::JointIndex joint_id = 2; joint_id < (pinocchio::JointIndex)model.njoints; ++joint_id) {
    if (verbose) LOG(INFO) << model.names[joint_id];
    fullJointNames.emplace_back(model.names[joint_id]);
  }

  this->mpcModelJointNames = initializeJointNames(this->fullJointNames, this->fixedJointNames, verbose);
  this->mpcModelToFullJointsIndices = initializeMpcToFullJointIndices(this->fullJointNames, this->mpcModelJointNames);
  this->jointIndexMap = createJointIndexMap(this->mpcModelJointNames);
  this->contactNames = concatenateStringVectors(this->contactNames3DoF, this->contactNames6DoF);
  this->mpc_joint_dim = this->mpcModelJointNames.size();
  this->full_joint_dim = this->fullJointNames.size();
  // The arm joints of the procedural arm swing. A legs-only robot (the EngineAI SA01) has none and omits
  // model_settings.armJointNames entirely, which leaves all four names empty: the swing is then disabled rather than
  // being a load-time failure. Any other combination is a configuration error and still throws, because a robot that
  // names three of the four, or misspells one, or fixes one out through fixedJointNames, would otherwise walk with a
  // half-built arm swing.
  const bool armJointNamesOmitted =
      j_l_shoulder_y_name.empty() && j_r_shoulder_y_name.empty() && j_l_elbow_y_name.empty() && j_r_elbow_y_name.empty();
  this->hasArmSwingJoints = !armJointNamesOmitted;
  if (this->hasArmSwingJoints) {
    CHECK(this->jointIndexMap.find(j_l_shoulder_y_name) != this->jointIndexMap.end());
    j_l_shoulder_y_index = this->jointIndexMap.at(j_l_shoulder_y_name);
    CHECK(this->jointIndexMap.find(j_r_shoulder_y_name) != this->jointIndexMap.end());
    j_r_shoulder_y_index = this->jointIndexMap.at(j_r_shoulder_y_name);
    CHECK(this->jointIndexMap.find(j_l_elbow_y_name) != this->jointIndexMap.end());
    j_l_elbow_y_index = this->jointIndexMap.at(j_l_elbow_y_name);
    CHECK(this->jointIndexMap.find(j_r_elbow_y_name) != this->jointIndexMap.end());
    j_r_elbow_y_index = this->jointIndexMap.at(j_r_elbow_y_name);
  } else {
    j_l_shoulder_y_index = 0;
    j_r_shoulder_y_index = 0;
    j_l_elbow_y_index = 0;
    j_r_elbow_y_index = 0;
    if (verbose) {
      LOG(INFO) << "\n #### model_settings.armJointNames is not set: the procedural arm swing reference is disabled.\n";
    }
  }

  const std::string footConstraintPrefix = prefix + "foot_constraint.";

  if (verbose) {
    LOG(INFO) << "\n #### Robot Model Foot Constraint Config:";
    LOG(INFO) << "\n #### "
                 "============================================================="
                 "================\n";
  }

  loadData::loadPtreeValue(pt, this->footConstraintConfig.positionErrorGain_z, footConstraintPrefix + "positionErrorGain_z", verbose);
  loadData::loadPtreeValue(pt, this->footConstraintConfig.orientationErrorGain, footConstraintPrefix + "orientationErrorGain", verbose);
  loadData::loadPtreeValue(pt, this->footConstraintConfig.linearVelocityErrorGain_z, footConstraintPrefix + "linearVelocityErrorGain_z",
                           verbose);
  loadData::loadPtreeValue(pt, this->footConstraintConfig.linearVelocityErrorGain_xy, footConstraintPrefix + "linearVelocityErrorGain_xy",
                           verbose);
  loadData::loadPtreeValue(pt, this->footConstraintConfig.angularVelocityErrorGain, footConstraintPrefix + "angularVelocityErrorGain",
                           verbose);
  loadData::loadPtreeValue(pt, this->footConstraintConfig.linearAccelerationErrorGain_z,
                           footConstraintPrefix + "linearAccelerationErrorGain_z", verbose);
  loadData::loadPtreeValue(pt, this->footConstraintConfig.linearAccelerationErrorGain_xy,
                           footConstraintPrefix + "linearAccelerationErrorGain_xy", verbose);
  loadData::loadPtreeValue(pt, this->footConstraintConfig.angularAccelerationErrorGain,
                           footConstraintPrefix + "angularAccelerationErrorGain", verbose);
  loadData::loadPtreeValue(pt, this->footConstraintConfig.softConstraintWeight, footConstraintPrefix + "softConstraintWeight", verbose);
  loadData::loadPtreeValue(pt, this->footConstraintConfig.normalVelocitySoftConstraintWeight,
                           footConstraintPrefix + "normalVelocitySoftConstraintWeight", verbose);
  loadData::loadPtreeValue(pt, this->footConstraintConfig.constrainOrientation, footConstraintPrefix + "constrainOrientation", verbose);
  loadData::loadPtreeValue(pt, this->footConstraintConfig.constrainYawRateAboutContactNormal,
                           footConstraintPrefix + "constrainYawRateAboutContactNormal", verbose);

  // LINT.IfChange(terrain_height_yaml_path)
  loadData::loadPtreeValue(pt, this->terrainHeight, "terrainHeight", verbose);
  // clang-format off
  // LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml:terrain_height_config, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.yaml:terrain_height_config)
  // clang-format on

  // The contact_implicit block, key by key from the one list every reader of it shares.
  for (const ContactImplicitKey& key : contactImplicitKeys()) {
    loadData::loadPtreeValue(pt, this->contactImplicitConfig.*key.field, absl::StrCat(kContactImplicitBlock, ".", key.name), verbose);
  }

  // LINT.IfChange(nominal_foothold_yaml_path)
  const std::string nominalFootholdPrefix = "nominal_foothold.";
  // clang-format off
  // LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml:nominal_foothold_config, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.yaml:nominal_foothold_config)
  // clang-format on
  loadData::loadPtreeValue(pt, this->nominalFootholdConfig.stepWidth, nominalFootholdPrefix + "stepWidth", verbose);

  if (verbose) {
    LOG(INFO) << " #### "
                 "============================================================="
                 "================";
    LOG(INFO) << " #### "
                 "============================================================="
                 "================";
  }
}

}  // namespace ocs2::humanoid
