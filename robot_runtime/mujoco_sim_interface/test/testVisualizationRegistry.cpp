/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.

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

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "gtest/gtest.h"
#include "mujoco/mujoco.h"

#include "mujoco_sim_interface/MujocoContactPatch.h"
#include "mujoco_sim_interface/MujocoUtils.h"
#include "mujoco_sim_interface/visualization/ContactTimelineVisualization.h"
#include "mujoco_sim_interface/visualization/ExternalForceVisualization.h"
#include "mujoco_sim_interface/visualization/MetricsOverlay.h"
#include "mujoco_sim_interface/visualization/MujocoOptionFlagVisualization.h"
#include "mujoco_sim_interface/visualization/TargetContactPatchVisualization.h"
#include "mujoco_sim_interface/visualization/VisualizationRegistry.h"

namespace robot::mujoco_sim_interface {
namespace {

TEST(VisualizationRegistry, NamesAndHotkeysAreUnique) {
  const std::vector<VisualizationInfo> infos = availableVisualizations();
  ASSERT_FALSE(infos.empty());
  std::set<std::string> names;
  std::set<char> hotkeys;
  for (const VisualizationInfo& info : infos) {
    EXPECT_FALSE(info.name.empty());
    EXPECT_FALSE(info.description.empty());
    EXPECT_TRUE(names.insert(info.name).second)
        << "duplicate name " << info.name;
    if (info.hotkey != 0) {
      EXPECT_TRUE(info.hotkey >= 'a' && info.hotkey <= 'z')
          << info.name << " must use a lower-case letter";
      EXPECT_TRUE(hotkeys.insert(info.hotkey).second)
          << "duplicate hotkey " << info.hotkey << " for " << info.name;
    }
  }
  // Hotkeys the renderer keeps for itself: camera tracking, the cheatsheet and
  // the geom groups.
  EXPECT_EQ(hotkeys.count('k'), 0u);
  EXPECT_EQ(hotkeys.count('p'), 0u);
}

TEST(VisualizationRegistry, EveryNameCreatesItsVisualization) {
  for (const VisualizationInfo& info : availableVisualizations()) {
    const std::unique_ptr<MujocoVisualization> visualization =
        createVisualization(info.name);
    ASSERT_NE(visualization, nullptr) << info.name;
    EXPECT_EQ(visualization->name(), info.name);
    EXPECT_EQ(visualization->hotkey(), info.hotkey);
  }
  EXPECT_EQ(createVisualization("no_such_marker"), nullptr);
}

TEST(VisualizationRegistry, DefaultSetIsTheHistoricalViewer) {
  const std::vector<std::string> defaults = defaultVisualizationNames();
  const std::set<std::string> set(defaults.begin(), defaults.end());
  for (const char* absl_nonnull name :
       {"metrics", "external_forces", "contact_forces", "base_velocity",
        "contact_timeline", "target_contact_patches"}) {
    EXPECT_EQ(set.count(name), 1u) << name << " is on by default";
  }
  for (const char* absl_nonnull name :
       {"mj_contact_points", "mj_contact_forces", "mj_com", "mj_inertia",
        "mj_convex_hull", "mj_transparent"}) {
    EXPECT_EQ(set.count(name), 0u)
        << name << " is off by default, like in MuJoCo";
  }
  for (const char* absl_nonnull name : {"center_of_mass", "zmp", "dcm"}) {
    EXPECT_EQ(set.count(name), 0u)
        << name << " is off unless listed in the task file";
    EXPECT_NE(createVisualization(name), nullptr) << name;
  }
}

TEST(VisualizationRegistry, ListedNamesAreCreatedEnabledInRegistryOrder) {
  std::vector<std::string> errors;
  const std::vector<std::unique_ptr<MujocoVisualization>> visualizations =
      createVisualizations({"target_contact_patches", "metrics", "mj_com"},
                           &errors);
  EXPECT_TRUE(errors.empty());
  ASSERT_EQ(visualizations.size(), 3u);
  EXPECT_EQ(visualizations[0]->name(), "metrics")
      << "drawing order is the registry's, not the list's";
  EXPECT_EQ(visualizations[1]->name(), "target_contact_patches");
  EXPECT_EQ(visualizations[2]->name(), "mj_com");
  for (const std::unique_ptr<MujocoVisualization>& visualization :
       visualizations) {
    EXPECT_TRUE(visualization->enabled()) << visualization->name();
  }

  EXPECT_TRUE(createVisualizations(/*names=*/{}, &errors).empty());
  EXPECT_TRUE(errors.empty());
}

TEST(VisualizationRegistry, UnknownAndRepeatedNamesAreReportedAndSkipped) {
  std::vector<std::string> errors;
  const std::vector<std::unique_ptr<MujocoVisualization>> visualizations =
      createVisualizations({"metrics", "contact_barcode", "metrics"}, &errors);
  ASSERT_EQ(visualizations.size(), 1u);
  EXPECT_EQ(visualizations[0]->name(), "metrics");
  ASSERT_EQ(errors.size(), 2u);
  EXPECT_NE(errors[0].find("contact_barcode"), std::string::npos);
  EXPECT_NE(errors[0].find("contact_timeline"), std::string::npos)
      << "the message lists the available names";
  EXPECT_NE(errors[1].find("more than once"), std::string::npos);
  // No error sink: still skipped, no crash.
  EXPECT_EQ(
      createVisualizations({"contact_barcode"}, /*errors=*/nullptr).size(), 0u);
}

TEST(VisualizationRegistry, ToggleAndEnable) {
  std::unique_ptr<MujocoVisualization> timeline =
      createVisualization("contact_timeline");
  ASSERT_NE(timeline, nullptr);
  EXPECT_TRUE(timeline->enabled());
  timeline->toggle();
  EXPECT_FALSE(timeline->enabled());
  timeline->setEnabled(true);
  EXPECT_TRUE(timeline->enabled());
  EXPECT_EQ(timeline->hotkey(), 'b');
  EXPECT_EQ(createVisualization("target_contact_patches")->hotkey(), 'g');
}

TEST(MujocoOptionFlagVisualization, FlagFollowsTheEnabledStateEveryFrame) {
  std::unique_ptr<MujocoOptionFlagVisualization> com =
      MujocoOptionFlagVisualization::centerOfMass();
  EXPECT_FALSE(com->enabled()) << "MuJoCo's own markers start off";
  EXPECT_EQ(com->flag(), mjVIS_COM);

  mjvOption options;
  mjv_defaultOption(&options);
  options.flags[mjVIS_COM] = 1;
  VisualizationFrame frame;
  frame.options = &options;
  com->beforeSceneUpdate(frame);
  EXPECT_EQ(options.flags[mjVIS_COM], 0)
      << "a disabled visualization clears its flag";
  com->setEnabled(true);
  com->beforeSceneUpdate(frame);
  EXPECT_EQ(options.flags[mjVIS_COM], 1);
  com->toggle();
  com->beforeSceneUpdate(frame);
  EXPECT_EQ(options.flags[mjVIS_COM], 0);

  // Without options the hook is a no-op.
  VisualizationFrame empty;
  com->beforeSceneUpdate(empty);
  MujocoOptionFlagVisualization::transparency()->beforeSceneUpdate(empty);
}

// Two geoms of their own transparency: an opaque box and a translucent
// collision shell, as Atlas's collision class.
constexpr char kTwoAlphas[] = R"(
<mujoco>
  <worldbody>
    <body name="robot">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
      <geom type="sphere" size="0.2" rgba="0.5 0.5 0.5 0.3"/>
    </body>
  </worldbody>
</mujoco>
)";

// The x-ray is MuJoCo's own mjVIS_TRANSPARENT flag, which leaves the model
// alone. It used to write one alpha into every geom of the model (0.3,
// then 1.0), so switching it off turned the translucent collision shells opaque
// for good.
TEST(MujocoOptionFlagVisualization, TransparencyLeavesTheModelsAlphasAlone) {
  char error[1000] = "";
  const MjSpecPtr spec(
      mj_parseXMLString(kTwoAlphas, /*vfs=*/nullptr, error, sizeof(error)));
  ASSERT_NE(spec, nullptr) << "mj_parseXMLString: " << error;
  const MjModelPtr model(mj_compile(spec.get(), /*vfs=*/nullptr));
  ASSERT_NE(model, nullptr);
  ASSERT_EQ(model->ngeom, 2);
  std::unique_ptr<MujocoOptionFlagVisualization> transparency =
      MujocoOptionFlagVisualization::transparency();
  EXPECT_EQ(transparency->flag(), mjVIS_TRANSPARENT);
  mjvOption options;
  mjv_defaultOption(&options);
  VisualizationFrame frame;
  frame.model = model.get();
  frame.options = &options;
  for (const bool enabled : {true, false, true, false}) {
    transparency->setEnabled(enabled);
    transparency->beforeSceneUpdate(frame);
    EXPECT_EQ(options.flags[mjVIS_TRANSPARENT], enabled ? 1 : 0);
    EXPECT_FLOAT_EQ(model->geom_rgba[kMjRgbaChannels * 0 + kAlphaChannel],
                    1.0f);
    EXPECT_FLOAT_EQ(model->geom_rgba[kMjRgbaChannels * 1 + kAlphaChannel],
                    0.3f);
  }
}

// A free body whose center of mass is 0.2 m ahead of its frame origin, as a
// pelvis's is off its frame.
constexpr char kOffsetInertia[] = R"(
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint/>
      <inertial pos="0.2 0 0" mass="10" diaginertia="1 1 1"/>
      <geom type="box" size="0.1 0.1 0.1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
)";

// The push is held for kPushHoldTime after the step it was applied at, starts
// at the body's center of mass (where MuJoCo applies xfrc_applied), and stays
// on screen: a 0.45 kg ball at 25 m/s used to draw an arrow of the one-step
// force, 0.05 m/N x 22.5 kN, a kilometer long, from the frame origin, and only
// in the one frame that held it.
TEST(ExternalForceVisualization, AHeldPushIsDrawnAtTheCenterOfMassAndClamped) {
  char error[1000] = "";
  const MjSpecPtr spec(
      mj_parseXMLString(kOffsetInertia, /*vfs=*/nullptr, error, sizeof(error)));
  ASSERT_NE(spec, nullptr) << "mj_parseXMLString: " << error;
  const MjModelPtr model(mj_compile(spec.get(), /*vfs=*/nullptr));
  ASSERT_NE(model, nullptr);
  MjState state(model.get());
  mj_forward(model.get(), state.data);
  const int base = 1;
  state.push = AppliedPush{.bodyId = base,
                           .impulseWorld = {-11.25, 0.0, 0.0},
                           .simTimeApplied = 0.0};
  mjvScene scene;
  mjv_defaultScene(&scene);
  mjv_makeScene(model.get(), &scene, /*maxgeom=*/16);

  state.data->time = 0.25 * ExternalForceVisualization::kPushHoldTime;
  ExternalForceVisualization::addArrows(model.get(), state, scene);
  ASSERT_EQ(scene.ngeom, 1);
  const mjvGeom& arrow = scene.geoms[0];
  // mjvGeom stores single-precision floats.
  // NOLINTBEGIN(frame-naming): MuJoCo's mjvGeom::pos
  EXPECT_NEAR(arrow.pos[0], state.data->xipos[3 * base + 0], 1.0e-6);
  EXPECT_NEAR(arrow.pos[2], state.data->xipos[3 * base + 2], 1.0e-6);
  // NOLINTEND(frame-naming)
  EXPECT_NEAR(state.data->xipos[3 * base] - state.data->xpos[3 * base], 0.2,
              1.0e-9);
  EXPECT_LT(arrow.mat[2], -0.9)
      << "the arrow does not point along the push, -x";
  EXPECT_LE(arrow.size[2], 1.5 + 1.0e-9)
      << "an arrow of " << arrow.size[2] << " m";

  scene.ngeom = 0;
  state.data->time = 1.5 * ExternalForceVisualization::kPushHoldTime;
  ExternalForceVisualization::addArrows(model.get(), state, scene);
  EXPECT_EQ(scene.ngeom, 0) << "the push outlived its hold time";
  mjv_freeScene(&scene);
}

TEST(MetricsOverlay, TextShowsTheRatesAsIntegersAndTheRestToAMillisecond) {
  Metrics metrics;
  metrics.fpsSim = 1999.7;
  metrics.rtfSmoothed = 0.98765;
  metrics.driftTick = -0.0000234;
  metrics.driftCumulative = 0.25;
  EXPECT_EQ(MetricsOverlay::text(metrics, /*renderFps=*/59.9,
                                 /*elapsedRealTime=*/12.3456, /*simTime=*/12.0),
            "Render FPS: 59\n"
            "Sim FPS: 1999\n"
            "Real Time[s]: 12.346\n"
            "Sim  Time[s]: 12.000\n\n"
            "RTF: 0.988\n"
            "Drift[ms]: -0.023\n"
            "Cumulative Drift[ms]: 250.000");
}

TEST(ContactTimelineVisualization, TickLabelsCountTheSecondsBackFromNow) {
  EXPECT_EQ(ContactTimelineVisualization::tickLabel(/*secondsAgo=*/0), "now");
  EXPECT_EQ(ContactTimelineVisualization::tickLabel(/*secondsAgo=*/1), "-1 s");
  EXPECT_EQ(ContactTimelineVisualization::tickLabel(/*secondsAgo=*/12),
            "-12 s");
}

TEST(TargetContactPatchVisualization,
     AKindOutsideTheEnumeratorsIsDrawnLikeAStance) {
  TargetContactPatch patch;
  patch.kind = TargetContactPatch::Kind::kStance;
  const ContactPatchStyle stance =
      TargetContactPatchVisualization::styleFor(/*contact=*/1, patch);
  EXPECT_FALSE(stance.fill);
  EXPECT_FALSE(stance.arrow);
  patch.kind = static_cast<TargetContactPatch::Kind>(
      7);  // what a patch decoded from a newer controller can carry
  const ContactPatchStyle unknown =
      TargetContactPatchVisualization::styleFor(/*contact=*/1, patch);
  EXPECT_EQ(unknown.rgba, stance.rgba);
  EXPECT_EQ(unknown.fill, stance.fill);
  EXPECT_EQ(unknown.arrow, stance.arrow);
  EXPECT_EQ(unknown.emission, stance.emission);

  patch.kind = TargetContactPatch::Kind::kSwingInFlight;
  const ContactPatchStyle swing =
      TargetContactPatchVisualization::styleFor(/*contact=*/1, patch);
  EXPECT_TRUE(swing.fill);
  EXPECT_TRUE(swing.arrow);
  EXPECT_GT(swing.rgba[3], stance.rgba[3])
      << "the step being executed is drawn brighter than a foot held in place";
  patch.kind = TargetContactPatch::Kind::kNextSwing;
  const ContactPatchStyle next =
      TargetContactPatchVisualization::styleFor(/*contact=*/1, patch);
  EXPECT_TRUE(next.fill);
  EXPECT_LT(next.rgba[3], swing.rgba[3]);
}

}  // namespace
}  // namespace robot::mujoco_sim_interface
