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

#include "humanoid_common_mpc/common/ContactInputParameterization.h"

#include <yaml-cpp/yaml.h>

#include <array>
#include <exception>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace ocs2::humanoid {

namespace {

struct ParameterizationEntry {
  absl::string_view name;
  ContactInputParameterization parameterization;
};

// The single place a parameterization is added.
// LINT.IfChange(contact_input_parameterization_registry)
constexpr std::array<ParameterizationEntry, 2> kParameterizationRegistry = {{
    {kWrenchContactInputParameterization, ContactInputParameterization::kWrench},
    {kBasisVectorsContactInputParameterization, ContactInputParameterization::kBasisVectors},
}};
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/common/ContactInputParameterization.h:contact_input_parameterization_names)
// clang-format on

}  // namespace

std::vector<std::string> contactInputParameterizationNames() {
  std::vector<std::string> names;
  names.reserve(kParameterizationRegistry.size());
  for (const ParameterizationEntry& entry : kParameterizationRegistry) {
    names.emplace_back(entry.name);
  }
  return names;
}

absl::string_view contactInputParameterizationName(ContactInputParameterization parameterization) {
  for (const ParameterizationEntry& entry : kParameterizationRegistry) {
    if (entry.parameterization == parameterization) {
      return entry.name;
    }
  }
  // Unreachable while every enumerator is registered; the registry test walks the enum to keep it so.
  return "unregistered";
}

absl::StatusOr<ContactInputParameterization> contactInputParameterizationFromName(absl::string_view name) {
  for (const ParameterizationEntry& entry : kParameterizationRegistry) {
    if (entry.name == name) {
      return entry.parameterization;
    }
  }
  return absl::InvalidArgumentError(absl::StrCat("[ContactInputParameterization] unknown ", kContactInputParameterizationKey, " '", name,
                                                 "'; valid names are: ", absl::StrJoin(contactInputParameterizationNames(), ", "),
                                                 " (humanoid_nmpc/docs/contact_basis_vectors/README.md)."));
}

absl::StatusOr<ContactInputParameterization> loadContactInputParameterization(absl::string_view taskFile) {
  const std::string taskFilePath(taskFile);
  YAML::Node root;
  try {
    root = YAML::LoadFile(taskFilePath);
  } catch (const std::exception& error) {
    return absl::NotFoundError(
        absl::StrCat("[ContactInputParameterization] failed to read the task file '", taskFilePath, "': ", error.what()));
  }
  // A key lookup on a scalar or a sequence throws rather than finding nothing; an empty file is an empty map.
  if (!root.IsMap() && !root.IsNull()) {
    return absl::InvalidArgumentError(absl::StrCat("[ContactInputParameterization] the task file '", taskFilePath,
                                                   "' is not a map of keys, so it cannot name a ", kContactInputParameterizationKey, "."));
  }

  // Present at all, whatever its value: `false` used to mean the wrench parameterization, which is also the default, but
  // a file that still spells the switch the old way was written against the old loader and has to be looked at.
  if (root[std::string(kRetiredContactBasisVectorInputsKey)]) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[ContactInputParameterization] '", kRetiredContactBasisVectorInputsKey, "' in ", taskFilePath,
        " is no longer read: the contact input parameterization is selected by name. Delete the key and write '",
        kContactInputParameterizationKey, ": ", kBasisVectorsContactInputParameterization, "' where it was true, '",
        kContactInputParameterizationKey, ": ", kWrenchContactInputParameterization, "' (the default) where it was false. Valid names: ",
        absl::StrJoin(contactInputParameterizationNames(), ", "), " (humanoid_nmpc/docs/contact_basis_vectors/README.md)."));
  }

  const YAML::Node node = root[std::string(kContactInputParameterizationKey)];
  if (!node) {
    return kDefaultContactInputParameterization;
  }
  if (!node.IsScalar()) {
    return absl::InvalidArgumentError(
        absl::StrCat("[ContactInputParameterization] ", kContactInputParameterizationKey, " in ", taskFilePath,
                     " must be one name; valid names are: ", absl::StrJoin(contactInputParameterizationNames(), ", "), "."));
  }
  return contactInputParameterizationFromName(node.Scalar());
}

}  // namespace ocs2::humanoid
