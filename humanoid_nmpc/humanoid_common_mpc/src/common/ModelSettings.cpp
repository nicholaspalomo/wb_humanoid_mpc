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

#include <climits>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "ocs2_core/Types.h"

#include "humanoid_common_mpc/common/BipedFeet.h"
#include "humanoid_common_mpc/common/StateLayout.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "nproto/Fnv1a64Constants.h"

namespace ocs2::humanoid {

/******************************************************************************************************/
/// Helper functions contained in a local anonymous namespace
/******************************************************************************************************/
namespace {

// The fields of the task file's `contact_implicit` block, for
// validateContactImplicitConfig() and the parameter updater.
constexpr ModelSettings::ContactImplicitKey kContactImplicitKeys[] = {
    {.fieldName = "complementarity_weight",
     .field = &ModelSettings::ContactImplicitConfig::complementarityWeight,
     .isWeight = true},
    {.fieldName = "slip_weight",
     .field = &ModelSettings::ContactImplicitConfig::slipWeight,
     .isWeight = true},
    {.fieldName = "penetration_weight",
     .field = &ModelSettings::ContactImplicitConfig::penetrationWeight,
     .isWeight = true},
    {.fieldName = "scale_pos_contact_rt_surface_ewrt_world_z",
     .field = &ModelSettings::ContactImplicitConfig::
                  scalePosContactRtSurfaceEwrtWorldZ,
     .isWeight = false},
    {.fieldName = "scale_linvel_contact_rt_world_ewrt_world_xy",
     .field = &ModelSettings::ContactImplicitConfig::
                  scaleLinvelContactRtWorldEwrtWorldXy,
     .isWeight = false},
    {.fieldName = "scale_angvel_contact_rt_world_ewrt_world_z",
     .field = &ModelSettings::ContactImplicitConfig::
                  scaleAngvelContactRtWorldEwrtWorldZ,
     .isWeight = false},
    {.fieldName = "gap_smoothing",
     .field = &ModelSettings::ContactImplicitConfig::gapSmoothing,
     .isWeight = false},
};

// ContactImplicitConfig holds nothing but these scalars, so a field added to it
// without a key here - which would be loaded by nothing, validated by nothing
// and hot-reloaded by nothing - changes its size and fails to compile.
static_assert(sizeof(ModelSettings::ContactImplicitConfig) ==
                  std::size(kContactImplicitKeys) * sizeof(scalar_t),
              "every field of ModelSettings::ContactImplicitConfig needs a key "
              "in kContactImplicitKeys");

}  // namespace

absl::Span<const ModelSettings::ContactImplicitKey>
ModelSettings::contactImplicitKeys() {
  return kContactImplicitKeys;
}

absl::StatusOr<ModelSettings> ModelSettings::Create(
    const std::string& configFile,
    const std::string& urdfFile,
    const std::string& mpcName,
    bool verbose) {
  ASSIGN_OR_RETURN(const mpc_config::TaskFile taskFile,
                   loadTaskFile(configFile));
  absl::StatusOr<ModelSettings> settings =
      Create(taskFile, urdfFile, mpcName, verbose);
  if (!settings.ok()) {
    return withConfigFile(settings.status(), configFile);
  }
  return settings;
}

namespace {

/**
 * FNV-1a 64 over words and length-prefixed strings, so that ("ab", "c") and
 * ("a", "bc") differ: add, then read value().
 *
 * Not thread-safe.
 */
class LayoutHash {
 public:
  void addWord(uint64_t word) {
    for (int octet = 0; octet < nproto::kFnvWordOctets; ++octet) {
      hash_ ^= (word >> (CHAR_BIT * octet)) & nproto::kFnvOctetMask;
      hash_ *= nproto::kFnvPrime;
    }
  }
  void addString(absl::string_view text) {
    addWord(text.size());
    for (const char character : text) {
      hash_ ^= static_cast<uint64_t>(static_cast<unsigned char>(character));
      hash_ *= nproto::kFnvPrime;
    }
  }
  void addStrings(const std::vector<std::string>& texts) {
    addWord(texts.size());
    for (const std::string& text : texts) addString(text);
  }
  uint64_t value() const { return hash_; }

 private:
  uint64_t hash_ = nproto::kFnvOffsetBasis;
};

}  // namespace

std::vector<std::string> ModelSettings::contactParentJointNamesInContactOrder()
    const {
  std::vector<std::string> parents;
  parents.reserve(contactNames.size());
  for (const PointContact& point : pointContacts) {
    parents.push_back(point.parentJointName);
  }
  for (size_t contact = 0; contact < contactNames6DoF.size(); ++contact) {
    parents.push_back(contact < contactParentJointNames.size()
                          ? contactParentJointNames[contact]
                          : std::string());
  }
  return parents;
}

std::optional<size_t> ModelSettings::legOfContact(size_t contactIndex) const {
  for (size_t leg = 0; leg < legs.size(); ++leg) {
    for (const size_t contact : legs[leg].contactIndices) {
      if (contact == contactIndex) return leg;
    }
  }
  return std::nullopt;
}

bool ModelSettings::isLeggedContactSet() const {
  if (legs.empty()) return false;
  for (size_t contact = 0; contact < numContacts(); ++contact) {
    if (!legOfContact(contact).has_value()) return false;
  }
  return true;
}

bool ModelSettings::hasBipedStance() const {
  const bool leggedNonBiped = isLeggedContactSet() && !isBipedContactSet();
  return !leggedNonBiped &&
         contactNames6DoF.size() >= static_cast<size_t>(kBipedFootCount);
}

uint64_t modelLayoutFingerprint(const ModelSettings& settings) {
  LayoutHash hash;
  hash.addString(kStateLayoutTag);
  hash.addString(settings.robotName);
  hash.addStrings(settings.mpcModelJointNames);
  hash.addStrings(settings.contactNames3DoF);
  hash.addStrings(settings.contactNames6DoF);
  // The value an unstamped message carries is never a layout's.
  return hash.value() == kUnstampedModelFingerprint
             ? kUnstampedModelFingerprint + 1
             : hash.value();
}

}  // namespace ocs2::humanoid
