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

#include <optional>
#include <string>

#include "web_viewer/ViewerCameraMode.h"

namespace robot::web_viewer {

// What a ViewerInput asks the render thread to do; POST /input?type=<name> and POST /settings make them
// (ViewerInputParsing.h).
enum class ViewerInputType {
  kDrag,                 // move the camera as a mouse drag does: type=drag
  kScroll,               // zoom as the mouse wheel does: type=scroll
  kKey,                  // the key the GLFW viewer binds to the same character: type=key
  kResetCamera,          // back to the camera of start-up: type=reset_camera
  kCameraMode,           // switch between the free and the tracking camera: type=camera
  kToggleVisualization,  // switch one marker family on or off: type=visualization
  kSettings,             // change the frame size, rate or quality: POST /settings
};

// The mouse button of a drag, which picks the camera motion as in the GLFW viewer: left rotates, right moves, middle
// zooms.
enum class ViewerMouseButton {
  kLeft,
  kMiddle,
  kRight,
};

/**
 * One input event from a browser, as the server queues it (ViewerInputQueue) and the render thread applies it.
 *
 * Passive data. `type` says which fields are meaningful; the parser (parseViewerInput(), parseViewerSettings()) has
 * checked and clamped them, so the render thread applies them as they are.
 */
struct ViewerInput {
  /** The bound of a drag's motion: dx and dy are each in [-kMaxDragFraction, kMaxDragFraction], also once merged. */
  // LINT.IfChange(drag_range)
  static constexpr double kMaxDragFraction = 2.0;
  // LINT.ThenChange(//robot_runtime/web_viewer/README.md:web_viewer_routes, //robot_runtime/web_viewer/page/index.html:drag_range)

  ViewerInputType type = ViewerInputType::kResetCamera;

  // kDrag: the button, whether shift was held (the horizontal variants of rotate and move, as in GLFW), and the
  // motion on both axes as fractions of the displayed height, as mjv_moveCamera takes it (the GLFW viewer's
  // cameraDragFractions()), each in [-2, 2].
  ViewerMouseButton button = ViewerMouseButton::kLeft;
  bool shift = false;
  double dx = 0.0;
  double dy = 0.0;

  // kScroll: wheel notches with GLFW's sign, in [-10, 10]: positive is the wheel turned away from the user (GLFW's
  // yoffset > 0) and zooms out, as in MuJoCo's own viewers.
  double notches = 0.0;

  // kKey: '0' to '5' (geom groups) or 'a' to 'z' (camera tracking and the visualizations' hotkeys).
  char key = 0;

  // kCameraMode.
  ViewerCameraMode cameraMode = ViewerCameraMode::kFree;

  // kToggleVisualization: a name of WebViewerServerOptions::visualizationNames.
  std::string visualizationName = "";

  // kSettings: the values to change, each clamped to its range; an absent one stays as it is. Width 160 to 3840 and
  // height 120 to 2160 pixels (the renderer clamps further to its maximum size), fps 1 to 60, quality 10 to 95.
  std::optional<int> width = std::nullopt;
  std::optional<int> height = std::nullopt;
  std::optional<int> fps = std::nullopt;
  std::optional<int> quality = std::nullopt;
};

}  // namespace robot::web_viewer
