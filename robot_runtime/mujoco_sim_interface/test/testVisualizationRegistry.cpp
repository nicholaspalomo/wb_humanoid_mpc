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

#include <gtest/gtest.h>
#include <mujoco/mujoco.h>

#include <set>
#include <string>
#include <vector>

#include "mujoco_sim_interface/visualization/MujocoOptionFlagVisualization.h"
#include "mujoco_sim_interface/visualization/VisualizationRegistry.h"

using namespace robot::mujoco_sim_interface;

TEST(VisualizationRegistry, NamesAndHotkeysAreUnique) {
  const std::vector<VisualizationInfo> infos = availableVisualizations();
  ASSERT_FALSE(infos.empty());
  std::set<std::string> names;
  std::set<char> hotkeys;
  for (const VisualizationInfo& info : infos) {
    EXPECT_FALSE(info.name.empty());
    EXPECT_FALSE(info.description.empty());
    EXPECT_TRUE(names.insert(info.name).second) << "duplicate name " << info.name;
    if (info.hotkey != 0) {
      EXPECT_TRUE(info.hotkey >= 'a' && info.hotkey <= 'z') << info.name << " must use a lower-case letter";
      EXPECT_TRUE(hotkeys.insert(info.hotkey).second) << "duplicate hotkey " << info.hotkey << " for " << info.name;
    }
  }
  // Hotkeys the renderer keeps for itself: camera tracking, the cheatsheet and the geom groups.
  EXPECT_EQ(hotkeys.count('k'), 0u);
  EXPECT_EQ(hotkeys.count('p'), 0u);
}

TEST(VisualizationRegistry, EveryNameCreatesItsVisualization) {
  for (const VisualizationInfo& info : availableVisualizations()) {
    const std::unique_ptr<MujocoVisualization> visualization = createVisualization(info.name);
    ASSERT_NE(visualization, nullptr) << info.name;
    EXPECT_EQ(visualization->name(), info.name);
    EXPECT_EQ(visualization->hotkey(), info.hotkey);
  }
  EXPECT_EQ(createVisualization("no_such_marker"), nullptr);
}

TEST(VisualizationRegistry, DefaultSetIsTheHistoricalViewer) {
  const std::vector<std::string> defaults = defaultVisualizationNames();
  const std::set<std::string> set(defaults.begin(), defaults.end());
  for (const char* name : {"metrics", "external_forces", "contact_forces", "base_velocity", "contact_timeline", "target_contact_patches"}) {
    EXPECT_EQ(set.count(name), 1u) << name << " is on by default";
  }
  for (const char* name : {"mj_contact_points", "mj_contact_forces", "mj_com", "mj_inertia", "mj_convex_hull", "mj_transparent"}) {
    EXPECT_EQ(set.count(name), 0u) << name << " is off by default, like in MuJoCo";
  }
}

TEST(VisualizationRegistry, ListedNamesAreCreatedEnabledInRegistryOrder) {
  std::vector<std::string> errors;
  const auto visualizations = createVisualizations({"target_contact_patches", "metrics", "mj_com"}, &errors);
  EXPECT_TRUE(errors.empty());
  ASSERT_EQ(visualizations.size(), 3u);
  EXPECT_EQ(visualizations[0]->name(), "metrics") << "drawing order is the registry's, not the list's";
  EXPECT_EQ(visualizations[1]->name(), "target_contact_patches");
  EXPECT_EQ(visualizations[2]->name(), "mj_com");
  for (const auto& visualization : visualizations) EXPECT_TRUE(visualization->enabled()) << visualization->name();

  EXPECT_TRUE(createVisualizations({}, &errors).empty());
  EXPECT_TRUE(errors.empty());
}

TEST(VisualizationRegistry, UnknownAndRepeatedNamesAreReportedAndSkipped) {
  std::vector<std::string> errors;
  const auto visualizations = createVisualizations({"metrics", "contact_barcode", "metrics"}, &errors);
  ASSERT_EQ(visualizations.size(), 1u);
  EXPECT_EQ(visualizations[0]->name(), "metrics");
  ASSERT_EQ(errors.size(), 2u);
  EXPECT_NE(errors[0].find("contact_barcode"), std::string::npos);
  EXPECT_NE(errors[0].find("contact_timeline"), std::string::npos) << "the message lists the available names";
  EXPECT_NE(errors[1].find("more than once"), std::string::npos);
  // No error sink: still skipped, no crash.
  EXPECT_EQ(createVisualizations({"contact_barcode"}, nullptr).size(), 0u);
}

TEST(VisualizationRegistry, ToggleAndEnable) {
  std::unique_ptr<MujocoVisualization> timeline = createVisualization("contact_timeline");
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
  std::unique_ptr<MujocoOptionFlagVisualization> com = MujocoOptionFlagVisualization::centerOfMass();
  EXPECT_FALSE(com->enabled()) << "MuJoCo's own markers start off";
  EXPECT_EQ(com->flag(), mjVIS_COM);

  mjvOption options;
  mjv_defaultOption(&options);
  options.flags[mjVIS_COM] = 1;
  VisualizationFrame frame;
  frame.options = &options;
  com->beforeSceneUpdate(frame);
  EXPECT_EQ(options.flags[mjVIS_COM], 0) << "a disabled visualization clears its flag";
  com->setEnabled(true);
  com->beforeSceneUpdate(frame);
  EXPECT_EQ(options.flags[mjVIS_COM], 1);
  com->toggle();
  com->beforeSceneUpdate(frame);
  EXPECT_EQ(options.flags[mjVIS_COM], 0);

  // Without options (or a simulator, for the transparency) the hook is a no-op.
  VisualizationFrame empty;
  com->beforeSceneUpdate(empty);
  MujocoOptionFlagVisualization::transparency()->beforeSceneUpdate(empty);
  EXPECT_EQ(MujocoOptionFlagVisualization::transparency()->flag(), MujocoOptionFlagVisualization::kTransparency);
}
