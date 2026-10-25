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

#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "ocs2_core/Dimensions.h"

#include "web_viewer/ViewerCameraMode.h"

namespace robot::web_viewer {

/**
 * What the viewer shows and how, as the render thread reports it with
 * ViewerFrameMailbox::publishStatus() and GET /state answers it
 * (viewerStatusJson()).
 *
 * Passive data: the render thread fills in a whole value after each frame, and
 * the server reads a copy.
 */
struct ViewerStatus {
  // The camera, as mjvCamera holds it.
  struct Camera {
    ViewerCameraMode mode = ViewerCameraMode::kTracking;
    // Degrees.
    double azimuth = 0.0;
    double elevation = 0.0;
    // Meters from the look-at point.
    double distance = 0.0;
    // The point the camera looks at, in world coordinates (meters).
    std::array<double, ocs2::kPositionDim> lookat = {0.0, 0.0, 0.0};
  };

  // The render settings and the measured rate.
  struct Render {
    // The size of the frames, in pixels.
    int width = 0;
    int height = 0;
    // The frame rate the render loop aims at while a client pulls frames.
    int targetFps = 0;
    // The frame rate it reached over the last second.
    double measuredFps = 0.0;
    // The JPEG quality, 10 to 95.
    int quality = 0;
  };

  // One marker family of the simulator's visualization registry.
  struct Visualization {
    // The registry name, which POST /input?type=visualization&name= toggles.
    std::string name = "";
    // The key that toggles it, 'a' to 'z', or 0 for none.
    char hotkey = 0;
    bool enabled = false;
    std::string description = "";
  };

  Camera camera = {};
  Render render = {};
  // Whether each MuJoCo geom group (index 0 to mjNGROUP - 1) is drawn; the keys
  // '0' to '5' toggle them.
  std::vector<bool> geomGroups = {};
  std::vector<Visualization> visualizations = {};
};

/**
 * Returns `status` as a JSON object: {"camera": {"mode", "azimuth",
 * "elevation", "distance", "lookat"}, "render":
 * {"width", "height", "targetFps", "measuredFps", "quality"}, "geomGroups":
 * [bool], "visualizations": [{"name", "hotkey", "enabled", "description"}]}. A
 * hotkey of 0 is null, and so is a number that is not finite.
 */
std::string viewerStatusJson(const ViewerStatus& status);

/** Returns `text` as a JSON string literal, quotes included, with the
 * characters JSON requires escaped. */
std::string jsonString(absl::string_view text);

}  // namespace robot::web_viewer
