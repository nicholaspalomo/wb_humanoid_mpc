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

#include "web_viewer/ViewerInputParsing.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "web_viewer/ViewerCameraMode.h"
#include "web_viewer/ViewerInput.h"

namespace robot::web_viewer {
namespace {

using Parameters = absl::flat_hash_map<std::string, std::string>;

// LINT.IfChange(web_viewer_routes)
// The ranges of README.md, "Routes" (a drag's, ViewerInput::kMaxDragFraction, with ViewerInput).
constexpr double kMaxNotches = 10.0;
constexpr int kMinWidth = 160;
constexpr int kMaxWidth = 3840;
constexpr int kMinHeight = 120;
constexpr int kMaxHeight = 2160;
constexpr int kMinFps = 1;
constexpr int kMaxFps = 60;
constexpr int kMinQuality = 10;
constexpr int kMaxQuality = 95;

// The parameters each input type takes besides `type`, all required.
constexpr std::array<absl::string_view, 4> kDragParameters = {"button", "shift", "dx", "dy"};
constexpr std::array<absl::string_view, 1> kScrollParameters = {"notches"};
constexpr std::array<absl::string_view, 1> kKeyParameters = {"key"};
constexpr std::array<absl::string_view, 1> kCameraParameters = {"mode"};
constexpr std::array<absl::string_view, 1> kVisualizationParameters = {"name"};
// The parameters of POST /settings, each optional.
constexpr std::array<absl::string_view, 4> kSettingsParameters = {"width", "height", "fps", "quality"};
// The input types of POST /input?type=.
constexpr std::array<absl::string_view, 6> kInputTypes = {"drag", "scroll", "key", "reset_camera", "camera", "visualization"};
// LINT.ThenChange(//robot_runtime/web_viewer/page/index.html:web_viewer_routes, //robot_runtime/web_viewer/README.md:web_viewer_routes)

// Refuses a parameter of `parameters` that is neither `type` nor one of `allowed`.
absl::Status checkKnownParameters(const Parameters& parameters, absl::Span<const absl::string_view> allowed, absl::string_view request) {
  for (const std::pair<const std::string, std::string>& parameter : parameters) {
    if (parameter.first == "type") continue;
    if (std::find(allowed.begin(), allowed.end(), parameter.first) == allowed.end()) {
      return absl::InvalidArgumentError(absl::StrCat("unknown parameter '", parameter.first, "' for ", request,
                                                     " (it takes: ", allowed.empty() ? "nothing" : absl::StrJoin(allowed, ", "), ")"));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<absl::string_view> requiredParameter(const Parameters& parameters, absl::string_view name, absl::string_view request) {
  const Parameters::const_iterator found = parameters.find(name);
  if (found == parameters.end()) return absl::InvalidArgumentError(absl::StrCat(request, " needs the parameter '", name, "'"));
  return absl::string_view(found->second);
}

// A finite number, clamped to [-limit, limit].
absl::StatusOr<double> clampedNumber(const Parameters& parameters, absl::string_view name, double limit, absl::string_view request) {
  absl::StatusOr<absl::string_view> text = requiredParameter(parameters, name, request);
  if (!text.ok()) return text.status();
  double value = 0.0;
  if (!absl::SimpleAtod(*text, &value) || !std::isfinite(value)) {
    return absl::InvalidArgumentError(absl::StrCat("parameter '", name, "' of ", request, " is not a finite number: '", *text, "'"));
  }
  return std::clamp(value, -limit, limit);
}

// An optional integer, clamped to [low, high].
absl::StatusOr<std::optional<int>> clampedInteger(const Parameters& parameters, absl::string_view name, int low, int high) {
  const Parameters::const_iterator found = parameters.find(name);
  if (found == parameters.end()) return std::optional<int>();
  int value = 0;
  if (!absl::SimpleAtoi(found->second, &value)) {
    return absl::InvalidArgumentError(absl::StrCat("parameter '", name, "' of /settings is not an integer: '", found->second, "'"));
  }
  return std::optional<int>(std::clamp(value, low, high));
}

absl::StatusOr<ViewerInput> parseDrag(const Parameters& parameters) {
  constexpr absl::string_view kRequest = "type=drag";
  ViewerInput input = {.type = ViewerInputType::kDrag};
  absl::StatusOr<absl::string_view> button = requiredParameter(parameters, "button", kRequest);
  if (!button.ok()) return button.status();
  if (*button == "left") {
    input.button = ViewerMouseButton::kLeft;
  } else if (*button == "middle") {
    input.button = ViewerMouseButton::kMiddle;
  } else if (*button == "right") {
    input.button = ViewerMouseButton::kRight;
  } else {
    return absl::InvalidArgumentError(absl::StrCat("parameter 'button' of type=drag is left, middle or right, not '", *button, "'"));
  }
  absl::StatusOr<absl::string_view> shift = requiredParameter(parameters, "shift", kRequest);
  if (!shift.ok()) return shift.status();
  if (*shift != "0" && *shift != "1") {
    return absl::InvalidArgumentError(absl::StrCat("parameter 'shift' of type=drag is 0 or 1, not '", *shift, "'"));
  }
  input.shift = *shift == "1";
  absl::StatusOr<double> dx = clampedNumber(parameters, "dx", ViewerInput::kMaxDragFraction, kRequest);
  if (!dx.ok()) return dx.status();
  absl::StatusOr<double> dy = clampedNumber(parameters, "dy", ViewerInput::kMaxDragFraction, kRequest);
  if (!dy.ok()) return dy.status();
  input.dx = *dx;
  input.dy = *dy;
  return input;
}

absl::StatusOr<ViewerInput> parseKey(const Parameters& parameters) {
  absl::StatusOr<absl::string_view> key = requiredParameter(parameters, "key", "type=key");
  if (!key.ok()) return key.status();
  const bool valid = key->size() == 1 && (((*key)[0] >= '0' && (*key)[0] <= '5') || ((*key)[0] >= 'a' && (*key)[0] <= 'z'));
  if (!valid) {
    return absl::InvalidArgumentError(absl::StrCat("parameter 'key' of type=key is one of 0-5 or a-z, not '", *key, "'"));
  }
  return ViewerInput{.type = ViewerInputType::kKey, .key = (*key)[0]};
}

absl::StatusOr<ViewerInput> parseCamera(const Parameters& parameters) {
  absl::StatusOr<absl::string_view> mode = requiredParameter(parameters, "mode", "type=camera");
  if (!mode.ok()) return mode.status();
  const std::optional<ViewerCameraMode> cameraMode = parseViewerCameraMode(*mode);
  if (!cameraMode.has_value()) {
    return absl::InvalidArgumentError(absl::StrCat("parameter 'mode' of type=camera is free or tracking, not '", *mode, "'"));
  }
  return ViewerInput{.type = ViewerInputType::kCameraMode, .cameraMode = *cameraMode};
}

absl::StatusOr<ViewerInput> parseVisualization(const Parameters& parameters, absl::Span<const std::string> visualizationNames) {
  absl::StatusOr<absl::string_view> name = requiredParameter(parameters, "name", "type=visualization");
  if (!name.ok()) return name.status();
  if (std::find(visualizationNames.begin(), visualizationNames.end(), *name) == visualizationNames.end()) {
    return absl::InvalidArgumentError(absl::StrCat("parameter 'name' of type=visualization is not a visualization of this viewer: '", *name,
                                                   "' (it has: ", absl::StrJoin(visualizationNames, ", "), ")"));
  }
  return ViewerInput{.type = ViewerInputType::kToggleVisualization, .visualizationName = std::string(*name)};
}

// The parser of one input type, after its parameters were checked against `allowed`.
absl::StatusOr<ViewerInput> parseKnownType(absl::string_view type,
                                           const Parameters& parameters,
                                           absl::Span<const std::string> visualizationNames) {
  if (type == "drag") return parseDrag(parameters);
  if (type == "scroll") {
    absl::StatusOr<double> notches = clampedNumber(parameters, "notches", kMaxNotches, "type=scroll");
    if (!notches.ok()) return notches.status();
    return ViewerInput{.type = ViewerInputType::kScroll, .notches = *notches};
  }
  if (type == "key") return parseKey(parameters);
  if (type == "reset_camera") return ViewerInput{.type = ViewerInputType::kResetCamera};
  if (type == "camera") return parseCamera(parameters);
  return parseVisualization(parameters, visualizationNames);
}

absl::Span<const absl::string_view> parametersOf(absl::string_view type) {
  if (type == "drag") return kDragParameters;
  if (type == "scroll") return kScrollParameters;
  if (type == "key") return kKeyParameters;
  if (type == "camera") return kCameraParameters;
  if (type == "visualization") return kVisualizationParameters;
  return {};
}

}  // namespace

absl::StatusOr<ViewerInput> parseViewerInput(const absl::flat_hash_map<std::string, std::string>& parameters,
                                             absl::Span<const std::string> visualizationNames) {
  absl::StatusOr<absl::string_view> type = requiredParameter(parameters, "type", "/input");
  if (!type.ok()) return type.status();
  if (std::find(kInputTypes.begin(), kInputTypes.end(), *type) == kInputTypes.end()) {
    return absl::InvalidArgumentError(
        absl::StrCat("parameter 'type' of /input is one of ", absl::StrJoin(kInputTypes, ", "), ", not '", *type, "'"));
  }
  absl::Status known = checkKnownParameters(parameters, parametersOf(*type), absl::StrCat("type=", *type));
  if (!known.ok()) return known;
  return parseKnownType(*type, parameters, visualizationNames);
}

absl::StatusOr<ViewerInput> parseViewerSettings(const absl::flat_hash_map<std::string, std::string>& parameters) {
  for (const std::pair<const std::string, std::string>& parameter : parameters) {
    if (std::find(kSettingsParameters.begin(), kSettingsParameters.end(), parameter.first) == kSettingsParameters.end()) {
      return absl::InvalidArgumentError(absl::StrCat("unknown parameter '", parameter.first,
                                                     "' for /settings (it takes: ", absl::StrJoin(kSettingsParameters, ", "), ")"));
    }
  }
  if (parameters.empty()) {
    return absl::InvalidArgumentError(absl::StrCat("/settings needs at least one of ", absl::StrJoin(kSettingsParameters, ", ")));
  }
  ViewerInput input = {.type = ViewerInputType::kSettings};
  absl::StatusOr<std::optional<int>> width = clampedInteger(parameters, "width", kMinWidth, kMaxWidth);
  if (!width.ok()) return width.status();
  absl::StatusOr<std::optional<int>> height = clampedInteger(parameters, "height", kMinHeight, kMaxHeight);
  if (!height.ok()) return height.status();
  absl::StatusOr<std::optional<int>> fps = clampedInteger(parameters, "fps", kMinFps, kMaxFps);
  if (!fps.ok()) return fps.status();
  absl::StatusOr<std::optional<int>> quality = clampedInteger(parameters, "quality", kMinQuality, kMaxQuality);
  if (!quality.ok()) return quality.status();
  input.width = *width;
  input.height = *height;
  input.fps = *fps;
  input.quality = *quality;
  return input;
}

}  // namespace robot::web_viewer
