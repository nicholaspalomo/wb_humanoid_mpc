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

#include "humanoid_common_mpc_app/visualization/VisualizationConfig.h"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/StatusMacros.h"

namespace ocs2::humanoid::visualization {

namespace {

/** True for a name the Rerun bridge accepts as one part of an entity path: letters, digits, '_', '-' and '.'. */
bool isEntityPathName(absl::string_view name) {
  if (name.empty() || name == "." || name == "..") {
    return false;
  }
  for (const char character : name) {
    if (!absl::ascii_isalnum(static_cast<unsigned char>(character)) && character != '_' && character != '-' && character != '.') {
      return false;
    }
  }
  return true;
}

/** The list of frame names at `key`, or `fallback` when the document does not carry the key or the list is empty. */
absl::StatusOr<std::vector<std::string>> loadFrameList(const YAML::Node& document,
                                                       absl::string_view key,
                                                       absl::string_view source,
                                                       const std::vector<std::string>& fallback) {
  const YAML::Node node = document[std::string(key)];
  if (!node || node.IsNull()) {
    return fallback;
  }
  if (!node.IsSequence()) {
    return absl::InvalidArgumentError(absl::StrCat(source, ": ", key, " must be a list of frame names."));
  }
  std::vector<std::string> frames;
  absl::flat_hash_set<std::string> seen;
  for (const YAML::Node& entry : node) {
    if (!entry.IsScalar()) {
      return absl::InvalidArgumentError(absl::StrCat(source, ": ", key, " must be a list of frame names, but an entry is not a name."));
    }
    const std::string& frame = entry.Scalar();
    if (!isEntityPathName(frame)) {
      return absl::InvalidArgumentError(absl::StrCat(source, ": ", key, " names the frame '", frame,
                                                     "'; a frame name of the plots and the 3D scene may only contain letters, "
                                                     "digits, '_', '-' and '.'."));
    }
    if (!seen.insert(frame).second) {
      return absl::InvalidArgumentError(absl::StrCat(source, ": ", key, " names the frame '", frame, "' twice."));
    }
    frames.push_back(frame);
  }
  if (frames.empty()) {
    return fallback;
  }
  return frames;
}

}  // namespace

absl::StatusOr<VisualizationConfig> parseVisualizationConfig(absl::string_view yamlText,
                                                             absl::string_view source,
                                                             const ModelSettings& modelSettings) {
  YAML::Node document;
  try {
    document = YAML::Load(std::string(yamlText));
  } catch (const YAML::Exception& e) {
    return absl::InvalidArgumentError(absl::StrCat(source, ": the task file does not parse: ", e.what()));
  }
  if (document && !document.IsNull() && !document.IsMap()) {
    return absl::InvalidArgumentError(absl::StrCat(source, ": the task file is not a map of keys."));
  }

  VisualizationConfig config;
  const YAML::Node frequency = document[std::string(kRerunSceneFrequencyKey)];
  if (frequency && !frequency.IsNull()) {
    scalar_t value = 0.0;
    try {
      value = frequency.as<scalar_t>();
    } catch (const YAML::Exception&) {
      return absl::InvalidArgumentError(absl::StrCat(source, ": ", kRerunSceneFrequencyKey, " is '",
                                                     frequency.IsScalar() ? frequency.Scalar() : "", "', which is not a number of Hz."));
    }
    if (!std::isfinite(value) || value <= 0.0) {
      return absl::InvalidArgumentError(
          absl::StrCat(source, ": ", kRerunSceneFrequencyKey, " is ", value, "; it must be a positive number of Hz."));
    }
    config.sceneFrequency = value;
    config.sceneFrequencyIsDefault = false;
  }
  ASSIGN_OR_RETURN(config.telemetryFrames, loadFrameList(document, kTelemetryFramesKey, source, modelSettings.contactNames));
  ASSIGN_OR_RETURN(config.planFrames, loadFrameList(document, kRerunPlanFramesKey, source, modelSettings.contactNames));
  return config;
}

absl::StatusOr<VisualizationConfig> loadVisualizationConfig(const std::string& taskFile, const ModelSettings& modelSettings) {
  std::ifstream stream(taskFile);
  if (!stream) {
    return absl::NotFoundError(absl::StrCat("the task file '", taskFile, "' cannot be read."));
  }
  std::stringstream text;
  text << stream.rdbuf();
  return parseVisualizationConfig(text.str(), taskFile, modelSettings);
}

std::string describeVisualizationConfig(const VisualizationConfig& config) {
  return absl::StrCat("viz/scene at ", config.sceneFrequency, " Hz (", kRerunSceneFrequencyKey,
                      config.sceneFrequencyIsDefault ? ", the default" : "", "), viz/telemetry once per robot/state sample; ",
                      kTelemetryFramesKey, ": ", absl::StrJoin(config.telemetryFrames, ", "), "; ", kRerunPlanFramesKey, ": ",
                      absl::StrJoin(config.planFrames, ", "));
}

}  // namespace ocs2::humanoid::visualization
