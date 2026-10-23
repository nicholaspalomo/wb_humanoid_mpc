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

#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"

#include "web_viewer/ViewerCameraMode.h"
#include "web_viewer/ViewerInput.h"
#include "web_viewer/ViewerInputParsing.h"

namespace robot::web_viewer {
namespace {

using Parameters = absl::flat_hash_map<std::string, std::string>;

const std::vector<std::string>& visualizationNames() {
  static const std::vector<std::string>& kNames = *new std::vector<std::string>{"contact_timeline", "zmp"};
  return kNames;
}

absl::StatusOr<ViewerInput> parse(const Parameters& parameters) {
  return parseViewerInput(parameters, visualizationNames());
}

void expectRefused(const Parameters& parameters, const std::string& messagePart) {
  const absl::StatusOr<ViewerInput> input = parse(parameters);
  ASSERT_FALSE(input.ok());
  EXPECT_EQ(input.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(input.status().message().find(messagePart), std::string::npos) << input.status();
}

TEST(ViewerInputParsingTest, ParsesADrag) {
  const absl::StatusOr<ViewerInput> input = parse({{"type", "drag"}, {"button", "right"}, {"shift", "1"}, {"dx", "0.25"}, {"dy", "-0.5"}});
  ASSERT_TRUE(input.ok()) << input.status();
  EXPECT_EQ(input->type, ViewerInputType::kDrag);
  EXPECT_EQ(input->button, ViewerMouseButton::kRight);
  EXPECT_TRUE(input->shift);
  EXPECT_DOUBLE_EQ(input->dx, 0.25);
  EXPECT_DOUBLE_EQ(input->dy, -0.5);
}

TEST(ViewerInputParsingTest, ClampsDragsAndScrolls) {
  const absl::StatusOr<ViewerInput> dragInput = parse({{"type", "drag"}, {"button", "left"}, {"shift", "0"}, {"dx", "7"}, {"dy", "-9"}});
  ASSERT_TRUE(dragInput.ok()) << dragInput.status();
  EXPECT_DOUBLE_EQ(dragInput->dx, 2.0);
  EXPECT_DOUBLE_EQ(dragInput->dy, -2.0);
  const absl::StatusOr<ViewerInput> up = parse({{"type", "scroll"}, {"notches", "50"}});
  ASSERT_TRUE(up.ok()) << up.status();
  EXPECT_EQ(up->type, ViewerInputType::kScroll);
  EXPECT_DOUBLE_EQ(up->notches, 10.0);
  const absl::StatusOr<ViewerInput> down = parse({{"type", "scroll"}, {"notches", "-50"}});
  ASSERT_TRUE(down.ok()) << down.status();
  EXPECT_DOUBLE_EQ(down->notches, -10.0);
}

TEST(ViewerInputParsingTest, RefusesNumbersThatAreNotFinite) {
  for (const char* absl_nonnull const text : {"nan", "NaN", "inf", "-inf", "infinity", "1e999", "", "abc", "0.5x"}) {
    expectRefused({{"type", "drag"}, {"button", "left"}, {"shift", "0"}, {"dx", text}, {"dy", "0"}}, "'dx'");
    expectRefused({{"type", "scroll"}, {"notches", text}}, "'notches'");
  }
}

TEST(ViewerInputParsingTest, RefusesMalformedDrags) {
  expectRefused({{"type", "drag"}, {"button", "back"}, {"shift", "0"}, {"dx", "0"}, {"dy", "0"}}, "'button'");
  expectRefused({{"type", "drag"}, {"button", "left"}, {"shift", "yes"}, {"dx", "0"}, {"dy", "0"}}, "'shift'");
  expectRefused({{"type", "drag"}, {"button", "left"}, {"shift", "0"}, {"dx", "0"}}, "'dy'");
  expectRefused({{"type", "drag"}, {"shift", "0"}, {"dx", "0"}, {"dy", "0"}}, "'button'");
  expectRefused({{"type", "drag"}, {"button", "left"}, {"shift", "0"}, {"dx", "0"}, {"dy", "0"}, {"dz", "0"}}, "unknown parameter 'dz'");
}

TEST(ViewerInputParsingTest, AcceptsOnlyTheKeysOfTheGlfwViewer) {
  for (const char* absl_nonnull const accepted : {"0", "5", "a", "k", "z"}) {
    const absl::StatusOr<ViewerInput> input = parse({{"type", "key"}, {"key", accepted}});
    ASSERT_TRUE(input.ok()) << accepted << ": " << input.status();
    EXPECT_EQ(input->type, ViewerInputType::kKey);
    EXPECT_EQ(input->key, accepted[0]);
  }
  for (const char* absl_nonnull const refused : {"6", "9", "A", "Z", "", "ab", " ", "/", "\x7f"}) {
    expectRefused({{"type", "key"}, {"key", refused}}, "'key'");
  }
}

TEST(ViewerInputParsingTest, ParsesTheCameraInputs) {
  const absl::StatusOr<ViewerInput> reset = parse({{"type", "reset_camera"}});
  ASSERT_TRUE(reset.ok()) << reset.status();
  EXPECT_EQ(reset->type, ViewerInputType::kResetCamera);
  const absl::StatusOr<ViewerInput> free = parse({{"type", "camera"}, {"mode", "free"}});
  ASSERT_TRUE(free.ok()) << free.status();
  EXPECT_EQ(free->type, ViewerInputType::kCameraMode);
  EXPECT_EQ(free->cameraMode, ViewerCameraMode::kFree);
  const absl::StatusOr<ViewerInput> tracking = parse({{"type", "camera"}, {"mode", "tracking"}});
  ASSERT_TRUE(tracking.ok()) << tracking.status();
  EXPECT_EQ(tracking->cameraMode, ViewerCameraMode::kTracking);
  expectRefused({{"type", "camera"}, {"mode", "fixed"}}, "'mode'");
  expectRefused({{"type", "camera"}}, "'mode'");
  expectRefused({{"type", "reset_camera"}, {"mode", "free"}}, "unknown parameter 'mode'");
}

TEST(ViewerInputParsingTest, TogglesOnlyKnownVisualizations) {
  const absl::StatusOr<ViewerInput> input = parse({{"type", "visualization"}, {"name", "zmp"}});
  ASSERT_TRUE(input.ok()) << input.status();
  EXPECT_EQ(input->type, ViewerInputType::kToggleVisualization);
  EXPECT_EQ(input->visualizationName, "zmp");
  expectRefused({{"type", "visualization"}, {"name", "gantry_weld"}}, "contact_timeline, zmp");
}

TEST(ViewerInputParsingTest, RefusesAMissingOrUnknownType) {
  expectRefused(/*parameters=*/{}, "'type'");
  expectRefused({{"type", "teleport"}}, "drag, scroll, key, reset_camera, camera, visualization");
}

TEST(ViewerInputParsingTest, ParsesAndClampsSettings) {
  const absl::StatusOr<ViewerInput> settings =
      parseViewerSettings({{"width", "99999"}, {"height", "10"}, {"fps", "0"}, {"quality", "100"}});
  ASSERT_TRUE(settings.ok()) << settings.status();
  EXPECT_EQ(settings->type, ViewerInputType::kSettings);
  EXPECT_EQ(settings->width, 3840);
  EXPECT_EQ(settings->height, 120);
  EXPECT_EQ(settings->fps, 1);
  EXPECT_EQ(settings->quality, 95);
  const absl::StatusOr<ViewerInput> fpsOnly = parseViewerSettings({{"fps", "24"}});
  ASSERT_TRUE(fpsOnly.ok()) << fpsOnly.status();
  EXPECT_EQ(fpsOnly->fps, 24);
  EXPECT_FALSE(fpsOnly->width.has_value());
  EXPECT_FALSE(fpsOnly->quality.has_value());
}

TEST(ViewerInputParsingTest, RefusesBadSettings) {
  const absl::StatusOr<ViewerInput> empty = parseViewerSettings({});
  EXPECT_EQ(empty.status().code(), absl::StatusCode::kInvalidArgument);
  const absl::StatusOr<ViewerInput> fraction = parseViewerSettings({{"width", "1280.5"}});
  ASSERT_FALSE(fraction.ok());
  EXPECT_NE(fraction.status().message().find("'width'"), std::string::npos) << fraction.status();
  const absl::StatusOr<ViewerInput> unknown = parseViewerSettings({{"gamma", "2"}});
  ASSERT_FALSE(unknown.ok());
  EXPECT_NE(unknown.status().message().find("'gamma'"), std::string::npos) << unknown.status();
}

}  // namespace
}  // namespace robot::web_viewer
