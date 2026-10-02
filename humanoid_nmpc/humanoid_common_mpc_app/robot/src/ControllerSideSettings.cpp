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

#include "humanoid_common_mpc_app/robot/ControllerSideSettings.h"

#include <exception>
#include <fstream>
#include <sstream>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

#include <ocs2_core/misc/LoadData.h>
#include <ocs2_core/misc/PropertyTree.h>

namespace ocs2::humanoid {
namespace {

/** The optional `key` of `pt` into `value`; absent leaves it. A value that is not a T is InvalidArgument naming the key. */
template <typename T>
absl::Status readOptionalValue(const PropertyTree& pt, absl::string_view key, T& value) {
  const PropertyTree* child = pt.findChild(key);
  if (child == nullptr) {
    return absl::OkStatus();
  }
  const std::optional<T> parsed = child->getValueOptional<T>();
  if (!parsed.has_value()) {
    return absl::InvalidArgumentError(absl::StrCat(key, " is '", child->data(), "', which is not a value of the expected type."));
  }
  value = *parsed;
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<ControllerSideSettings> parseControllerSideSettings(absl::string_view yamlText) {
  PropertyTree pt;
  try {
    loadData::readPropertyTreeFromString(yamlText, pt);
  } catch (const std::exception& error) {
    return absl::InvalidArgumentError(absl::StrCat("the document is not YAML: ", error.what()));
  }
  ControllerSideSettings settings;
  // LINT.IfChange(controller_side_keys)
  settings.contactEstimator = pt.getOptional<std::string>("contactEstimator");
  if (pt.findChild("contact_wrench_gate") != nullptr) {
    ContactWrenchGate::Config config;
    absl::Status status = readOptionalValue(pt, "contact_wrench_gate.debounceTime", config.debounceTime);
    if (status.ok()) {
      status = readOptionalValue(pt, "contact_wrench_gate.rampTime", config.rampTime);
    }
    if (!status.ok()) {
      settings.problems.push_back(absl::StrCat("contact_wrench_gate was not applied, the running gate is kept: ", status.message()));
    } else if (config.debounceTime >= 0.0 && config.rampTime >= 0.0) {
      settings.contactWrenchGate = config;
    } else {
      settings.problems.push_back(
          "contact_wrench_gate.debounceTime and contact_wrench_gate.rampTime must be non-negative; the block was not applied, the "
          "running gate is kept.");
    }
  }
  // LINT.ThenChange(//humanoid_nmpc/humanoid_centroidal_mpc/src/mrt/MpcParameterUpdaterModule.cpp:controller_side_keys)
  return settings;
}

absl::StatusOr<ControllerSideSettings> loadControllerSideSettings(const std::string& yamlFile) {
  std::ifstream stream(yamlFile);
  if (!stream.is_open()) {
    return absl::NotFoundError(absl::StrCat("cannot read ", yamlFile));
  }
  std::stringstream text;
  text << stream.rdbuf();
  absl::StatusOr<ControllerSideSettings> settings = parseControllerSideSettings(text.str());
  if (!settings.ok()) {
    return absl::InvalidArgumentError(absl::StrCat(yamlFile, ": ", settings.status().message()));
  }
  return settings;
}

}  // namespace ocs2::humanoid
