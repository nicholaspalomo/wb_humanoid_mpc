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

#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mujoco/mujoco.h"

#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "mujoco_sim_interface/viewer/EglContext.h"
#include "mujoco_sim_interface/viewer/EglLibrary.h"
#include "mujoco_sim_interface/viewer/GlCapabilities.h"
#include "mujoco_sim_interface/viewer/ViewerNames.h"
#include "mujoco_sim_interface/viewer/WebViewerOptions.h"
#include "robot_runtime/mujoco_sim_interface/test/ViewerTestSupport.h"

/*
 * The EGL context of the web viewer, on the machine's own EGL (Mesa's llvmpipe in the dev container): which display it
 * picks, that it works on a thread that never bound the OpenGL API, that contexts are independent of each other, and
 * what a missing library or device says.
 */

namespace robot::mujoco_sim_interface {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::AnyOf;
using ::testing::HasSubstr;

const EglLibrary* absl_nonnull egl() {
  absl::StatusOr<const EglLibrary* absl_nonnull> library = EglLibrary::Get("libEGL.so.1");
  EXPECT_TRUE(library.ok()) << library.status();
  return *library;
}

TEST(EglContext, CreatesADesktopOpenGlContextAndDescribesItsDisplay) {
  absl::StatusOr<std::unique_ptr<EglContext>> context = EglContext::Create(egl(), EglContext::Options{});
  ASSERT_TRUE(context.ok()) << context.status();
  EXPECT_THAT((*context)->description(), AnyOf(HasSubstr("surfaceless"), HasSubstr("EGL device"))) << (*context)->description();
  ASSERT_THAT((*context)->makeCurrent(), IsOk());
  const absl::StatusOr<GlCapabilities> capabilities = (*context)->queryGlCapabilities();
  ASSERT_TRUE(capabilities.ok()) << capabilities.status();
  EXPECT_FALSE(capabilities->renderer.empty());
  EXPECT_GT(capabilities->maxRenderbufferSize, 0);
  EXPECT_THAT(checkGlCapabilities(*capabilities, WebViewerOptions{}), IsOk())
      << capabilities->version << " " << capabilities->renderer << " " << capabilities->maxRenderbufferSize;
  EXPECT_THAT((*context)->release(), IsOk());
}

TEST(EglContext, WorksOnAThreadThatNeverBoundTheOpenGlApi) {
  absl::StatusOr<std::unique_ptr<EglContext>> context = EglContext::Create(egl(), EglContext::Options{});
  ASSERT_TRUE(context.ok()) << context.status();
  absl::Status made = absl::UnknownError("not run");
  absl::StatusOr<GlCapabilities> capabilities = absl::UnknownError("not run");
  absl::Status released = absl::UnknownError("not run");
  // EGL's bound API is per thread: this one starts on EGL_OPENGL_ES_API, which makeCurrent() and release() replace.
  std::thread other([&] {
    made = (*context)->makeCurrent();
    capabilities = (*context)->queryGlCapabilities();
    released = (*context)->release();
  });
  other.join();
  EXPECT_THAT(made, IsOk());
  EXPECT_TRUE(capabilities.ok()) << capabilities.status();
  EXPECT_THAT(released, IsOk());
}

TEST(EglContext, TwoContextsAreIndependentAndOneSurvivesTheOther) {
  absl::StatusOr<std::unique_ptr<MujocoSimInterface>> sim = MujocoSimInterface::Create(hingeSimConfig({kNoViewerName}), hingeUrdfPath());
  ASSERT_TRUE(sim.ok()) << sim.status();
  absl::StatusOr<std::unique_ptr<OffscreenTestRenderer>> first = OffscreenTestRenderer::Create(sim->get(), WebViewerOptions{});
  ASSERT_TRUE(first.ok()) << first.status();
  absl::StatusOr<std::unique_ptr<OffscreenTestRenderer>> second = OffscreenTestRenderer::Create(sim->get(), WebViewerOptions{});
  ASSERT_TRUE(second.ok()) << second.status();
  // The free camera stays where it is; the tracking one eases towards the robot from frame to frame.
  (*first)->frameRenderer().setCameraType(mjCAMERA_FREE);
  (*second)->frameRenderer().setCameraType(mjCAMERA_FREE);
  const absl::StatusOr<std::vector<uint8_t>> firstFrame = (*first)->render(/*width=*/320, /*height=*/240);
  const absl::StatusOr<std::vector<uint8_t>> secondFrame = (*second)->render(/*width=*/320, /*height=*/240);
  ASSERT_TRUE(firstFrame.ok()) << firstFrame.status();
  ASSERT_TRUE(secondFrame.ok()) << secondFrame.status();
  EXPECT_EQ(differingPixelFraction(*firstFrame, *secondFrame), 0.0) << "two contexts drew the same state differently";
  if (HasFailure()) {
    writeTestImage("two_contexts_first", *firstFrame, /*width=*/320, /*height=*/240);
    writeTestImage("two_contexts_second", *secondFrame, /*width=*/320, /*height=*/240);
  }

  first->reset();  // destroys its MuJoCo context and its EGL context; the display stays initialized
  const absl::StatusOr<std::vector<uint8_t>> afterwards = (*second)->render(/*width=*/320, /*height=*/240);
  ASSERT_TRUE(afterwards.ok()) << afterwards.status();
  EXPECT_GT(byteVariance(*afterwards), 10.0) << "the surviving context drew a uniform frame";
  EXPECT_EQ(differingPixelFraction(*afterwards, *secondFrame), 0.0) << "destroying one context changed what the other draws";
  if (HasFailure()) writeTestImage("two_contexts_afterwards", *afterwards, /*width=*/320, /*height=*/240);
}

TEST(EglLibrary, AMissingLibraryIsUnavailableAndSaysHowToGetOne) {
  EXPECT_THAT(
      EglLibrary::Get("libEGL_missing.so.1").status(),
      StatusIs(absl::StatusCode::kUnavailable, AllOf(HasSubstr("libEGL_missing.so.1"), HasSubstr("libegl1"), HasSubstr("libegl-mesa0"),
                                                     HasSubstr("--viewer=none"), HasSubstr("--viewer=web,none"))));
  // A failure is not remembered.
  EXPECT_THAT(EglLibrary::Get("libEGL_missing.so.1").status(), StatusIs(absl::StatusCode::kUnavailable));
}

TEST(EglLibrary, IsLoadedOncePerName) {
  const absl::StatusOr<const EglLibrary* absl_nonnull> first = EglLibrary::Get("libEGL.so.1");
  const absl::StatusOr<const EglLibrary* absl_nonnull> second = EglLibrary::Get("libEGL.so.1");
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_EQ(*first, *second);
  EXPECT_EQ((*first)->soname(), "libEGL.so.1");
}

TEST(EglContext, ADeviceThatDoesNotExistIsAnInvalidArgumentNamingTheCount) {
  EXPECT_THAT(EglContext::Create(egl(), EglContext::Options{.deviceId = 99}).status(),
              StatusIs(absl::StatusCode::kInvalidArgument, AllOf(HasSubstr("EGL device 99"), HasSubstr("device(s)"))));
}

}  // namespace
}  // namespace robot::mujoco_sim_interface
