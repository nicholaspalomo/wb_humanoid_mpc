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

#include "web_viewer/ViewerStatus.h"

#include <cmath>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/Dimensions.h"

#include "web_viewer/ViewerCameraMode.h"

namespace robot::web_viewer {
namespace {

// The quotes around a JSON string.
constexpr size_t kJsonStringQuotes = 2;
// JSON escapes the control characters, U+0000 to U+001F: the characters below this.
constexpr int kJsonControlCharacterEnd = 0x20;

// A JSON number, or null for one JSON cannot spell (NaN, infinity).
std::string jsonNumber(double value) {
  if (!std::isfinite(value)) return "null";
  return absl::StrFormat("%.9g", value);
}

absl::string_view jsonBool(bool value) {
  return value ? "true" : "false";
}

std::string cameraJson(const ViewerStatus::Camera& camera) {
  return absl::StrCat(R"({"mode":)", jsonString(viewerCameraModeName(camera.mode)), R"(,"azimuth":)", jsonNumber(camera.azimuth),
                      R"(,"elevation":)", jsonNumber(camera.elevation), R"(,"distance":)", jsonNumber(camera.distance), R"(,"lookat":[)",
                      jsonNumber(camera.lookat[ocs2::kXAxisIndex]), ",", jsonNumber(camera.lookat[ocs2::kYAxisIndex]), ",",
                      jsonNumber(camera.lookat[ocs2::kZAxisIndex]), "]}");
}

std::string renderJson(const ViewerStatus::Render& render) {
  return absl::StrCat(R"({"width":)", render.width, R"(,"height":)", render.height, R"(,"targetFps":)", render.targetFps,
                      R"(,"measuredFps":)", jsonNumber(render.measuredFps), R"(,"quality":)", render.quality, "}");
}

std::string visualizationJson(const ViewerStatus::Visualization& visualization) {
  const std::string hotkey = visualization.hotkey == 0 ? "null" : jsonString(std::string{visualization.hotkey});
  return absl::StrCat(R"({"name":)", jsonString(visualization.name), R"(,"hotkey":)", hotkey, R"(,"enabled":)",
                      jsonBool(visualization.enabled), R"(,"description":)", jsonString(visualization.description), "}");
}

}  // namespace

std::string jsonString(absl::string_view text) {
  std::string json = "\"";
  json.reserve(text.size() + kJsonStringQuotes);
  for (const char character : text) {
    switch (character) {
      case '"':
        json += R"(\")";
        break;
      case '\\':
        json += R"(\\)";
        break;
      case '\n':
        json += R"(\n)";
        break;
      case '\r':
        json += R"(\r)";
        break;
      case '\t':
        json += R"(\t)";
        break;
      default:
        if (static_cast<unsigned char>(character) < kJsonControlCharacterEnd) {
          absl::StrAppendFormat(&json, R"(\u%04x)", static_cast<int>(character));
        } else {
          json += character;
        }
    }
  }
  json += '"';
  return json;
}

std::string viewerStatusJson(const ViewerStatus& status) {
  std::string json =
      absl::StrCat(R"({"camera":)", cameraJson(status.camera), R"(,"render":)", renderJson(status.render), R"(,"geomGroups":[)");
  for (size_t group = 0; group < status.geomGroups.size(); ++group) {
    if (group > 0) json += ',';
    json += jsonBool(status.geomGroups[group]);
  }
  json += R"(],"visualizations":[)";
  for (size_t index = 0; index < status.visualizations.size(); ++index) {
    if (index > 0) json += ',';
    json += visualizationJson(status.visualizations[index]);
  }
  json += "]}";
  return json;
}

}  // namespace robot::web_viewer
