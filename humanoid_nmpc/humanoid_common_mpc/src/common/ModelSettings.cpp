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

#include <iterator>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_mpc_config/task_file.nproto.h"

namespace ocs2::humanoid {

/******************************************************************************************************/
/// Helper functions contained in a local anonymous namespace
/******************************************************************************************************/
namespace {

// The fields of the task file's `contact_implicit` block, for validateContactImplicitConfig() and the parameter updater.
constexpr ModelSettings::ContactImplicitKey kContactImplicitKeys[] = {
    {.fieldName = "complementarity_weight", .field = &ModelSettings::ContactImplicitConfig::complementarityWeight, .isWeight = true},
    {.fieldName = "slip_weight", .field = &ModelSettings::ContactImplicitConfig::slipWeight, .isWeight = true},
    {.fieldName = "penetration_weight", .field = &ModelSettings::ContactImplicitConfig::penetrationWeight, .isWeight = true},
    {.fieldName = "height_reference", .field = &ModelSettings::ContactImplicitConfig::heightReference, .isWeight = false},
    {.fieldName = "velocity_reference", .field = &ModelSettings::ContactImplicitConfig::velocityReference, .isWeight = false},
    {.fieldName = "angular_velocity_reference",
     .field = &ModelSettings::ContactImplicitConfig::angularVelocityReference,
     .isWeight = false},
    {.fieldName = "gap_smoothing", .field = &ModelSettings::ContactImplicitConfig::gapSmoothing, .isWeight = false},
};

// ContactImplicitConfig holds nothing but these scalars, so a field added to it without a key here - which would be
// loaded by nothing, validated by nothing and hot-reloaded by nothing - changes its size and fails to compile.
static_assert(sizeof(ModelSettings::ContactImplicitConfig) == std::size(kContactImplicitKeys) * sizeof(scalar_t),
              "every field of ModelSettings::ContactImplicitConfig needs a key in kContactImplicitKeys");

}  // namespace

absl::Span<const ModelSettings::ContactImplicitKey> ModelSettings::contactImplicitKeys() {
  return kContactImplicitKeys;
}

absl::StatusOr<ModelSettings> ModelSettings::Create(const std::string& configFile,
                                                    const std::string& urdfFile,
                                                    const std::string& mpcName,
                                                    bool verbose) {
  ASSIGN_OR_RETURN(const mpc_config::TaskFile taskFile, loadTaskFile(configFile));
  absl::StatusOr<ModelSettings> settings = Create(taskFile, urdfFile, mpcName, verbose);
  if (!settings.ok()) {
    return withConfigFile(settings.status(), configFile);
  }
  return settings;
}

void ModelSettings::loadFullJointNames(const std::string& urdfFile, bool verbose) {
  // Get full joint order from a full pinocchio interface, this removes any joints marked as fix in the urdf.
  const PinocchioInterface fullPinocchioInterface = createDefaultPinocchioInterface(urdfFile);
  const pinocchio::Model& model = fullPinocchioInterface.getModel();
  if (verbose) LOG(INFO) << "Full URDF joints: ";
  fullJointNames.reserve(model.njoints - 2);  // Subtract universe and root joint
  for (pinocchio::JointIndex joint_id = 2; joint_id < static_cast<pinocchio::JointIndex>(model.njoints); ++joint_id) {
    if (verbose) LOG(INFO) << model.names[joint_id];
    fullJointNames.emplace_back(model.names[joint_id]);
  }
}

}  // namespace ocs2::humanoid
