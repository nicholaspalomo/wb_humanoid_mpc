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

#include "absl/strings/string_view.h"

/**
 * The scene contract with the Rerun bridge (humanoid_nmpc/humanoid_rerun_viewer/README.md, "The 3D scene"): the names
 * of the robot model instances and the paths of the markers, which the bridge resolves under world/, and the marker
 * colors the publisher sets explicitly. A marker the publisher leaves uncolored takes the bridge's default.
 */
namespace ocs2::humanoid::visualization::scene {

// LINT.IfChange(robot_instances)
/** The robot as robot/state measures it (the MPC observation's state when no sample has arrived recently). */
inline constexpr absl::string_view kMeasuredInstance = "measured";
/** The last node of the MPC plan. */
inline constexpr absl::string_view kTerminalStateInstance = "terminal_state";
/** The reference at the plan's last time. */
inline constexpr absl::string_view kTerminalTargetInstance = "terminal_target";
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/python/humanoid_rerun_viewer/scene_contract.py:robot_instances)
// clang-format on

// LINT.IfChange(markers)
inline constexpr absl::string_view kContactForces = "markers/contact_forces";
inline constexpr absl::string_view kCenterOfPressure = "markers/center_of_pressure";
inline constexpr absl::string_view kCornerForces = "markers/corner_forces";
inline constexpr absl::string_view kPlanEndEffectors = "plan/end_effectors";
inline constexpr absl::string_view kPlanBase = "plan/base";
inline constexpr absl::string_view kPlanCom = "plan/com";
inline constexpr absl::string_view kPlanFootholds = "plan/footholds";
inline constexpr absl::string_view kCollisionSpheres = "markers/collision_spheres";
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/python/humanoid_rerun_viewer/scene_contract.py:markers)
// clang-format on

/** An RGBA color in [0, 1]. */
struct Rgba {
  float r = 0.0f;
  float g = 0.0f;
  float b = 0.0f;
  float a = 1.0f;
};

// OCS2's MATLAB-like palette (ocs2::Color), which the bridge draws its defaults in.
// LINT.IfChange(marker_palette)
inline constexpr Rgba kBlue{0.0f, 0.4470f, 0.7410f, 1.0f};
inline constexpr Rgba kOrange{0.8500f, 0.3250f, 0.0980f, 1.0f};
inline constexpr Rgba kYellow{0.9290f, 0.6940f, 0.1250f, 1.0f};
inline constexpr Rgba kPurple{0.4940f, 0.1840f, 0.5560f, 1.0f};
inline constexpr Rgba kGreen{0.4660f, 0.6740f, 0.1880f, 1.0f};
inline constexpr Rgba kRed{0.6350f, 0.0780f, 0.1840f, 1.0f};
/** Contact i (and plan frame i) is drawn in kContactColors[i % 5]: left foot purple, right foot orange. */
inline constexpr std::array<Rgba, 5> kContactColors = {kPurple, kOrange, kBlue, kGreen, kYellow};
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/python/humanoid_rerun_viewer/palette.py:marker_palette)
// clang-format on

/** [N/m] Force per meter of arrow of a contact force. */
inline constexpr double kForceScale = 200.0;

}  // namespace ocs2::humanoid::visualization::scene
