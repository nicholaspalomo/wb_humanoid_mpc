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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mujoco/mujoco.h"

#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "mujoco_sim_interface/MujocoUtils.h"
#include "mujoco_sim_interface/viewer/EglContext.h"
#include "mujoco_sim_interface/viewer/MujocoFrameRenderer.h"
#include "mujoco_sim_interface/viewer/WebViewerOptions.h"

/*
 * Scenes, simulators and an offscreen renderer shared by the viewer tests of this package.
 */

namespace robot::mujoco_sim_interface {

inline constexpr char kAtlasScenePath[] = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.xml";
inline constexpr char kAtlasUrdfPath[] = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";

/** Writes `content` to `name` in the test's temporary directory and returns its path. */
std::string writeTestFile(absl::string_view name, absl::string_view content);

/** The URDF of the hinge robot: a floating base with an arm on two hinges. */
std::string hingeUrdfPath();

/** A simulator configuration of the hinge robot's scene (no gantry weld, no contact points) with the viewers `viewerNames`. */
MujocoSimConfig hingeSimConfig(std::vector<std::string> viewerNames);

/**
 * A simulator configuration of the shipped Atlas scene, contact detection on both feet (so the contact timeline draws),
 * held by the gantry weld, with the viewers `viewerNames` and the visualizations `visualizations`.
 */
MujocoSimConfig atlasSimConfig(std::vector<std::string> viewerNames, std::vector<std::string> visualizations);

/** The ids of this process's threads, from /proc/self/task, sorted. */
std::vector<int> processThreadIds();

/** The share of pixels that differ between two RGB images of one size, in [0, 1]; 1 when the sizes differ. */
double differingPixelFraction(absl::Span<const uint8_t> first, absl::Span<const uint8_t> second);

/** The variance of the bytes of an RGB image; zero for a uniform one. */
double byteVariance(absl::Span<const uint8_t> rgb);

/**
 * Writes a bottom-up RGB image of `width` x `height` as `name`.ppm into the test's undeclared outputs
 * (TEST_UNDECLARED_OUTPUTS_DIR, Bazel's outputs.zip), for a failing image comparison; nothing outside Bazel.
 */
void writeTestImage(absl::string_view name, absl::Span<const uint8_t> rgb, int width, int height);

/**
 * An offscreen renderer of a simulator for the tests: an EGL context, a private copy of the simulator's model sized by
 * `options` and a MujocoFrameRenderer drawing into the offscreen buffer. Every method makes its context current on the
 * calling thread first, so that several renderers may take turns. Not thread-safe. `sim` must outlive it.
 */
class OffscreenTestRenderer {
 public:
  static absl::StatusOr<std::unique_ptr<OffscreenTestRenderer>> Create(const MujocoSimInterface* absl_nonnull sim,
                                                                       const WebViewerOptions& options);

  OffscreenTestRenderer(const OffscreenTestRenderer&) = delete;
  OffscreenTestRenderer& operator=(const OffscreenTestRenderer&) = delete;
  OffscreenTestRenderer(OffscreenTestRenderer&&) = delete;
  OffscreenTestRenderer& operator=(OffscreenTestRenderer&&) = delete;

  /** Frees MuJoCo's GPU context in its EGL context, then releases that. */
  ~OffscreenTestRenderer();

  /** Draws the simulator's newest snapshot at `width` x `height` and returns it: RGB, rows bottom-up. */
  absl::StatusOr<std::vector<uint8_t>> render(int width, int height);

  MujocoFrameRenderer& frameRenderer() { return frameRenderer_; }
  const EglContext& context() const { return *context_; }
  /** The private model copy the renderer draws with. */
  const mjModel& model() const { return *model_; }

 private:
  OffscreenTestRenderer(const MujocoSimInterface* absl_nonnull sim, std::unique_ptr<EglContext> context, MjModelPtr model);

  std::unique_ptr<EglContext> context_;
  MjModelPtr model_;
  MujocoFrameRenderer frameRenderer_;
};

}  // namespace robot::mujoco_sim_interface
