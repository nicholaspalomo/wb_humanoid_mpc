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

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "mujoco_sim_interface/viewer/GlCapabilities.h"
#include "mujoco_sim_interface/viewer/WebViewerOptions.h"

/*
 * The GL preflight of the web viewer on synthetic capabilities: every condition
 * under which MuJoCo would end the process while creating or resizing its
 * offscreen buffer or its shadow map is refused first, with a message that
 * names the limit and the fix. The live case (a real context asked for too many
 * samples) is testWebMujocoViewer.cpp's.
 */

namespace robot::mujoco_sim_interface {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::HasSubstr;

/** A context that offers everything: llvmpipe's numbers. */
GlCapabilities capable() {
  GlCapabilities capabilities;
  capabilities.vendor = "Mesa";
  capabilities.renderer = "llvmpipe (LLVM 20.1.2, 256 bits)";
  capabilities.version = "4.5 (Compatibility Profile) Mesa 25.2.8";
  capabilities.extensions =
      "GL_ARB_multisample GL_ARB_framebuffer_object GL_EXT_texture";
  capabilities.maxRenderbufferSize = 16384;
  capabilities.maxSamples = 8;
  capabilities.maxTextureSize = 16384;
  return capabilities;
}

/** The end every refusal shares: how to run without the web viewer. */
::testing::Matcher<absl::string_view> namesTheFix() {
  return AllOf(HasSubstr("--viewer=none"), HasSubstr("--viewer=web,none"));
}

TEST(GlCapabilities, AModernContextIsAccepted) {
  EXPECT_THAT(checkGlCapabilities(capable(), WebViewerOptions{}), IsOk());
}

TEST(GlCapabilities, OpenGl14IsRefused) {
  GlCapabilities capabilities = capable();
  capabilities.version = "1.4 Mesa 6.5";
  EXPECT_THAT(checkGlCapabilities(capabilities, WebViewerOptions{}),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       AllOf(HasSubstr("OpenGL 1.5 or newer"), HasSubstr("1.4"),
                             namesTheFix())));
}

TEST(GlCapabilities, AVersionThatDoesNotParseIsRefused) {
  GlCapabilities capabilities = capable();
  capabilities.version = "";
  EXPECT_THAT(checkGlCapabilities(capabilities, WebViewerOptions{}),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(GlCapabilities, OpenGl15NeedsTheFramebufferObjectExtension) {
  GlCapabilities capabilities = capable();
  capabilities.version = "1.5.0 NVIDIA 1.0";
  capabilities.extensions = "GL_ARB_multisample GL_ARB_framebuffer_object_ish";
  EXPECT_THAT(
      checkGlCapabilities(capabilities, WebViewerOptions{}),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               AllOf(HasSubstr("GL_ARB_framebuffer_object"), namesTheFix())));
  capabilities.extensions = "GL_ARB_multisample GL_ARB_framebuffer_object";
  EXPECT_THAT(checkGlCapabilities(capabilities, WebViewerOptions{}), IsOk());
}

TEST(GlCapabilities, OpenGl30HasFramebufferObjectsWithoutTheExtension) {
  GlCapabilities capabilities = capable();
  capabilities.version = "3.0 Mesa 10.0";
  capabilities.extensions = "";
  EXPECT_THAT(checkGlCapabilities(capabilities, WebViewerOptions{}), IsOk());
}

TEST(GlCapabilities, AnOffscreenBufferPastTheRenderbufferLimitIsRefused) {
  GlCapabilities capabilities = capable();
  capabilities.maxRenderbufferSize = 2048;
  WebViewerOptions options;
  options.maxWidth = 3840;
  options.maxHeight = 2160;
  EXPECT_THAT(checkGlCapabilities(capabilities, options),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       AllOf(HasSubstr("maxWidth"), HasSubstr("3840"),
                             HasSubstr("GL_MAX_RENDERBUFFER_SIZE of 2048"),
                             namesTheFix())));
  options.maxWidth = 2048;
  EXPECT_THAT(checkGlCapabilities(capabilities, options),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       AllOf(HasSubstr("maxHeight"), HasSubstr("2160"))));
  options.maxHeight = 2048;
  EXPECT_THAT(checkGlCapabilities(capabilities, options), IsOk());
}

TEST(GlCapabilities, MoreSamplesThanTheContextOffersAreRefused) {
  WebViewerOptions options;
  options.msaaSamples = 9;
  EXPECT_THAT(checkGlCapabilities(capable(), options),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       AllOf(HasSubstr("msaaSamples of 9"),
                             HasSubstr("GL_MAX_SAMPLES of 8"), namesTheFix())));
  options.msaaSamples = 8;
  EXPECT_THAT(checkGlCapabilities(capable(), options), IsOk());
  // No multisampling needs none, even from a context that reports no
  // GL_MAX_SAMPLES at all.
  GlCapabilities noSamples = capable();
  noSamples.maxSamples = 0;
  options.msaaSamples = 0;
  EXPECT_THAT(checkGlCapabilities(noSamples, options), IsOk());
}

// MuJoCo makes the shadow map a depth texture of shadowSize x shadowSize and
// ends the process ("Shadow framebuffer is not complete") when the context
// cannot make one that large; the preflight did not look at it.
TEST(GlCapabilities, AShadowMapPastTheTextureLimitIsRefused) {
  GlCapabilities capabilities = capable();
  capabilities.maxTextureSize = 4096;
  WebViewerOptions options;
  options.shadowSize = 8192;
  EXPECT_THAT(
      checkGlCapabilities(capabilities, options),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               AllOf(HasSubstr("shadowSize of 8192"),
                     HasSubstr("GL_MAX_TEXTURE_SIZE of 4096"), namesTheFix())));
  options.shadowSize = 4096;
  EXPECT_THAT(checkGlCapabilities(capabilities, options), IsOk());
  // No shadows need no texture, whatever the context reports.
  capabilities.maxTextureSize = 0;
  options.shadowSize = 0;
  EXPECT_THAT(checkGlCapabilities(capabilities, options), IsOk());
}

}  // namespace
}  // namespace robot::mujoco_sim_interface
