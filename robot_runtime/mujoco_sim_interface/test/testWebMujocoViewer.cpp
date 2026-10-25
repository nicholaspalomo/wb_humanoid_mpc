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

#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/log_severity.h"
#include "absl/base/nullability.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "mujoco_sim_interface/viewer/EglContext.h"
#include "mujoco_sim_interface/viewer/EglLibrary.h"
#include "mujoco_sim_interface/viewer/GlCapabilities.h"
#include "mujoco_sim_interface/viewer/ViewerNames.h"
#include "mujoco_sim_interface/viewer/WebMujocoViewer.h"
#include "mujoco_sim_interface/viewer/WebViewerOptions.h"
#include "robot_runtime/mujoco_sim_interface/test/ViewerTestSupport.h"
#include "robot_runtime/mujoco_sim_interface/test/WebViewerTestClient.h"
#include "web_viewer/JpegDecoder.h"
#include "web_viewer/ViewerThreadPriority.h"

/*
 * The `web` viewer end to end: EGL, the render thread and the HTTP server
 * around a simulator (an ephemeral port). It serves frames of its size, follows
 * the camera input and keys a browser sends, draws nothing without a client,
 * places every thread it starts, stops its render thread on a start-up timeout
 * (and says it waits for a set-up step that cannot be interrupted), refuses
 * what the GL preflight refuses, and is destroyed promptly with a stream open.
 */

namespace robot::mujoco_sim_interface {

/** Reaches into a WebMujocoViewer for the tests: the start-up delay hooks, its
 * render thread and its newest frame. */
class WebMujocoViewerTestPeer {
 public:
  static void setInitDelay(WebMujocoViewer& viewer, absl::Duration delay) {
    viewer.initDelayForTesting_ = delay;
  }
  /** A set-up step that, like an EGL or llvmpipe call, does not look for a stop
   * request while it runs. */
  static void setUninterruptibleInitDelay(WebMujocoViewer& viewer,
                                          absl::Duration delay) {
    viewer.uninterruptibleInitDelayForTesting_ = delay;
  }
  static bool renderThreadJoinable(const WebMujocoViewer& viewer) {
    return viewer.renderThread_.joinable();
  }
  /** Makes the next frame's renderAndPublish() fail, as an internal
   * libjpeg-turbo error would. */
  static void failNextFrame(WebMujocoViewer& viewer) {
    viewer.failNextFrameForTesting_.store(true);
  }
  /** The sequence of the newest frame, 0 before the first; asking does not
   * count as demand. */
  static int64_t latestSequence(const WebMujocoViewer& viewer) {
    const std::optional<web_viewer::EncodedFrame> frame =
        viewer.mailbox_.latestFrameMetadata();
    return frame.has_value() ? frame->sequence : 0;
  }
};

namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::HasSubstr;

/** Web viewer options for a test: an ephemeral port, a small frame. */
WebViewerOptions testOptions() {
  WebViewerOptions options;
  options.port = 0;
  options.width = 320;
  options.height = 240;
  options.fps = 30;
  // Not SCHED_IDLE, which would leave these viewers no time on a machine busy
  // with other tests.
  options.threadPriority = web_viewer::ViewerThreadPriority::kInherited;
  return options;
}

std::unique_ptr<MujocoSimInterface> makeHingeSim(
    const WebViewerOptions& options) {
  MujocoSimConfig config = hingeSimConfig({kWebViewerName});
  config.viewer.web = options;
  config.visualizations = {"contact_forces", "metrics"};
  absl::StatusOr<std::unique_ptr<MujocoSimInterface>> sim =
      MujocoSimInterface::Create(config, hingeUrdfPath());
  EXPECT_TRUE(sim.ok()) << sim.status();
  return sim.ok() ? *std::move(sim) : nullptr;
}

std::unique_ptr<WebMujocoViewer> makeViewer(
    const MujocoSimInterface* absl_nonnull sim,
    const WebViewerOptions& options) {
  absl::StatusOr<std::unique_ptr<WebMujocoViewer>> viewer =
      WebMujocoViewer::Create(sim, options);
  EXPECT_TRUE(viewer.ok()) << viewer.status();
  return viewer.ok() ? *std::move(viewer) : nullptr;
}

/** Polls /state until `ready` holds for its body or `timeout` passes; returns
 * the last body. */
std::string awaitState(int port,
                       const std::function<bool(const std::string&)>& ready,
                       absl::Duration timeout = absl::Seconds(3)) {
  const absl::Time deadline = absl::Now() + timeout;
  std::string body;
  while (absl::Now() < deadline) {
    body = httpGet(port, "/state").body;
    if (ready(body)) return body;
    absl::SleepFor(absl::Milliseconds(20));
  }
  return body;
}

/** Polls `condition` every 10 ms until it holds (true) or 3 s pass (false). */
bool eventually(const std::function<bool()>& condition) {
  const absl::Time deadline = absl::Now() + absl::Seconds(3);
  while (absl::Now() < deadline) {
    if (condition()) return true;
    absl::SleepFor(absl::Milliseconds(10));
  }
  return condition();
}

web_viewer::JpegDecoder::Image decode(const std::string& jpeg) {
  absl::StatusOr<std::unique_ptr<web_viewer::JpegDecoder>> decoder =
      web_viewer::JpegDecoder::Create();
  EXPECT_TRUE(decoder.ok()) << decoder.status();
  if (!decoder.ok()) return {};
  absl::StatusOr<web_viewer::JpegDecoder::Image> image =
      (*decoder)->decode(jpeg);
  EXPECT_TRUE(image.ok()) << image.status();
  return image.ok() ? *std::move(image) : web_viewer::JpegDecoder::Image{};
}

TEST(WebMujocoViewer, ServesFramesOfTheConfiguredSize) {
  const std::unique_ptr<MujocoSimInterface> sim = makeHingeSim(testOptions());
  ASSERT_NE(sim, nullptr);
  sim->initSim();
  ASSERT_THAT(sim->viewerStatus(), IsOk());
  EXPECT_EQ(sim->activeViewerName(), kWebViewerName);
  ASSERT_TRUE(sim->webViewerPort().has_value());
  const int port = sim->webViewerPort().value_or(0);
  ASSERT_GT(port, 0);
  EXPECT_GT(port, 0);

  const HttpReply page = httpGet(port, "/");
  EXPECT_EQ(page.status, 200);
  EXPECT_THAT(page.body, HasSubstr("/frame.jpg"));
  const HttpReply frame = httpGet(port, "/frame.jpg");
  ASSERT_EQ(frame.status, 200);
  EXPECT_GT(frame.frameSequence, 0);
  const web_viewer::JpegDecoder::Image image = decode(frame.body);
  EXPECT_EQ(image.width, 320);
  EXPECT_EQ(image.height, 240);
  EXPECT_GT(byteVariance(image.rgb), 100.0) << "a uniform frame";
}

TEST(WebMujocoViewer, ADragTurnsTheCameraAndTheNextFrameDiffers) {
  const std::unique_ptr<MujocoSimInterface> sim = makeHingeSim(testOptions());
  ASSERT_NE(sim, nullptr);
  sim->initSim();
  ASSERT_THAT(sim->viewerStatus(), IsOk());
  const int port = sim->webViewerPort().value_or(0);
  ASSERT_GT(port, 0);
  const HttpReply before = httpGet(port, "/frame.jpg");
  ASSERT_EQ(before.status, 200);
  const std::optional<double> azimuth =
      jsonNumberField(httpGet(port, "/state").body, "azimuth");
  ASSERT_TRUE(azimuth.has_value());

  ASSERT_EQ(httpPost(port, "/input?type=drag&button=left&shift=0&dx=0.25&dy=0")
                .status,
            204);
  const std::string state = awaitState(port, [&](const std::string& body) {
    return jsonNumberField(body, "azimuth") != azimuth;
  });
  EXPECT_NE(jsonNumberField(state, "azimuth"), azimuth) << state;
  // The physics does not run, so only the camera can change the picture.
  const HttpReply after = httpGet(
      port, absl::StrCat("/frame.jpg?after=", before.frameSequence + 1));
  ASSERT_EQ(after.status, 200);
  EXPECT_GT(
      differingPixelFraction(decode(before.body).rgb, decode(after.body).rgb),
      0.01);
}

TEST(WebMujocoViewer, KeysCameraModesAndVisualizationsFollowTheBrowser) {
  const std::unique_ptr<MujocoSimInterface> sim = makeHingeSim(testOptions());
  ASSERT_NE(sim, nullptr);
  sim->initSim();
  ASSERT_THAT(sim->viewerStatus(), IsOk());
  const int port = sim->webViewerPort().value_or(0);
  ASSERT_GT(port, 0);
  const std::string initial = awaitState(port, [](const std::string& body) {
    return jsonStringField(body, "mode").has_value();
  });
  EXPECT_EQ(jsonStringField(initial, "mode"), "tracking");

  ASSERT_EQ(httpPost(port, "/input?type=key&key=k").status, 204);
  EXPECT_EQ(jsonStringField(awaitState(port,
                                       [](const std::string& body) {
                                         return jsonStringField(body, "mode") ==
                                                "free";
                                       }),
                            "mode"),
            "free");
  ASSERT_EQ(httpPost(port, "/input?type=drag&button=left&shift=0&dx=0.3&dy=0.1")
                .status,
            204);
  ASSERT_EQ(httpPost(port, "/input?type=reset_camera").status, 204);
  const std::string reset = awaitState(port, [&](const std::string& body) {
    return jsonStringField(body, "mode") == "tracking";
  });
  EXPECT_EQ(jsonStringField(reset, "mode"), "tracking");
  EXPECT_EQ(jsonNumberField(reset, "azimuth"),
            jsonNumberField(initial, "azimuth"));
  ASSERT_EQ(httpPost(port, "/input?type=camera&mode=free").status, 204);
  EXPECT_EQ(jsonStringField(awaitState(port,
                                       [](const std::string& body) {
                                         return jsonStringField(body, "mode") ==
                                                "free";
                                       }),
                            "mode"),
            "free");

  EXPECT_EQ(visualizationEnabled(initial, "metrics"), true);
  ASSERT_EQ(httpPost(port, "/input?type=visualization&name=metrics").status,
            204);
  EXPECT_EQ(visualizationEnabled(awaitState(port,
                                            [](const std::string& body) {
                                              return visualizationEnabled(
                                                         body, "metrics") ==
                                                     false;
                                            }),
                                 "metrics"),
            false);

  // A key past the geometry groups is refused by the server and changes
  // nothing.
  const std::string beforeBadKey = httpGet(port, "/state").body;
  EXPECT_EQ(httpPost(port, "/input?type=key&key=9").status, 400);
  absl::SleepFor(absl::Milliseconds(300));
  EXPECT_EQ(httpGet(port, "/state").body, beforeBadKey);
}

TEST(WebMujocoViewer, DrawsNothingWithoutAClient) {
  const std::unique_ptr<MujocoSimInterface> sim = makeHingeSim(testOptions());
  ASSERT_NE(sim, nullptr);
  absl::StatusOr<std::unique_ptr<WebMujocoViewer>> viewer =
      WebMujocoViewer::Create(sim.get(), testOptions());
  ASSERT_TRUE(viewer.ok()) << viewer.status();
  ASSERT_THAT((*viewer)->start(), IsOk());
  EXPECT_EQ(WebMujocoViewerTestPeer::latestSequence(**viewer), 0)
      << "a frame was drawn before anybody asked";
  ASSERT_EQ(httpGet((*viewer)->port().value_or(0), "/frame.jpg").status, 200);
  // Past the demand lease of the request, the renderer goes idle.
  absl::SleepFor(web_viewer::ViewerFrameMailbox::kDemandLease +
                 absl::Milliseconds(300));
  const int64_t idle = WebMujocoViewerTestPeer::latestSequence(**viewer);
  absl::SleepFor(absl::Milliseconds(1300));
  EXPECT_EQ(WebMujocoViewerTestPeer::latestSequence(**viewer), idle)
      << "frames were drawn with no client";
}

TEST(WebMujocoViewer, AStartUpTimeoutEndsTheRenderThreadBeforeStartReturns) {
  const std::unique_ptr<MujocoSimInterface> sim = makeHingeSim(testOptions());
  ASSERT_NE(sim, nullptr);
  WebViewerOptions options = testOptions();
  options.startTimeout = absl::Milliseconds(300);
  const std::unique_ptr<WebMujocoViewer> viewer =
      makeViewer(sim.get(), options);
  ASSERT_NE(viewer, nullptr);
  WebMujocoViewerTestPeer::setInitDelay(*viewer, absl::Seconds(5));
  const absl::Time begin = absl::Now();
  EXPECT_THAT(viewer->start(),
              StatusIs(absl::StatusCode::kDeadlineExceeded,
                       HasSubstr("did not set up EGL and OpenGL")));
  EXPECT_LT(absl::Now() - begin, absl::Seconds(3))
      << "start() waited for the delayed set-up";
  EXPECT_FALSE(WebMujocoViewerTestPeer::renderThreadJoinable(*viewer))
      << "the render thread outlived start()";
  EXPECT_FALSE(viewer->ok());
}

// A set-up step that runs past the deadline cannot be cut short: start() says
// it is stopping the render thread, waits for that step to end and only then
// returns DeadlineExceeded. It used to wait in silence, up to twice the timeout
// under load, while its comment promised to wait no longer than
// options.startTimeout.
TEST(WebMujocoViewer,
     AStartUpTimeoutDuringAStepThatCannotBeInterruptedSaysItWaitsForTheStep) {
  const std::unique_ptr<MujocoSimInterface> sim = makeHingeSim(testOptions());
  ASSERT_NE(sim, nullptr);
  WebViewerOptions options = testOptions();
  options.startTimeout = absl::Milliseconds(300);
  const std::unique_ptr<WebMujocoViewer> viewer =
      makeViewer(sim.get(), options);
  ASSERT_NE(viewer, nullptr);
  constexpr absl::Duration kStepDuration = absl::Milliseconds(900);
  WebMujocoViewerTestPeer::setUninterruptibleInitDelay(*viewer, kStepDuration);
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(
      log,
      Log(absl::LogSeverity::kInfo, _,
          HasSubstr(
              "stopping the render thread after the start-up timeout (300ms)")))
      .Times(1);
  log.StartCapturingLogs();
  const absl::Time begin = absl::Now();
  EXPECT_THAT(viewer->start(),
              StatusIs(absl::StatusCode::kDeadlineExceeded,
                       HasSubstr("did not set up EGL and OpenGL")));
  EXPECT_GE(absl::Now() - begin, kStepDuration)
      << "start() returned while the render thread was still in its set-up";
  log.StopCapturingLogs();
  EXPECT_FALSE(WebMujocoViewerTestPeer::renderThreadJoinable(*viewer))
      << "the render thread outlived start()";
  EXPECT_FALSE(viewer->ok());
}

TEST(WebMujocoViewer, TheGlPreflightRefusesMoreSamplesThanTheContextOffers) {
  // GL_MAX_SAMPLES of this machine's context.
  absl::StatusOr<const EglLibrary* absl_nonnull> egl =
      EglLibrary::Get("libEGL.so.1");
  ASSERT_TRUE(egl.ok()) << egl.status();
  absl::StatusOr<std::unique_ptr<EglContext>> context =
      EglContext::Create(*egl, EglContext::Options{});
  ASSERT_TRUE(context.ok()) << context.status();
  ASSERT_THAT((*context)->makeCurrent(), IsOk());
  const absl::StatusOr<GlCapabilities> capabilities =
      (*context)->queryGlCapabilities();
  ASSERT_THAT((*context)->release(), IsOk());
  ASSERT_TRUE(capabilities.ok()) << capabilities.status();

  const std::unique_ptr<MujocoSimInterface> sim = makeHingeSim(testOptions());
  ASSERT_NE(sim, nullptr);
  WebViewerOptions options = testOptions();
  options.msaaSamples = capabilities->maxSamples + 1;
  const std::unique_ptr<WebMujocoViewer> viewer =
      makeViewer(sim.get(), options);
  ASSERT_NE(viewer, nullptr);
  // MuJoCo would have ended the process; the preflight refuses first, and the
  // process lives on.
  EXPECT_THAT(viewer->start(), StatusIs(absl::StatusCode::kFailedPrecondition,
                                        HasSubstr("GL_MAX_SAMPLES")));
  EXPECT_FALSE(viewer->ok());
}

// While the render thread sets up EGL (up to startTimeout), /state already
// names the configured markers, size and rate instead of an empty registry and
// 0x0, which made the page say "No markers are configured for this robot".
TEST(WebMujocoViewer, TheStateBeforeTheSetUpListsTheConfiguredMarkers) {
  const std::unique_ptr<MujocoSimInterface> sim = makeHingeSim(testOptions());
  ASSERT_NE(sim, nullptr);
  const std::unique_ptr<WebMujocoViewer> viewer =
      makeViewer(sim.get(), testOptions());
  ASSERT_NE(viewer, nullptr);
  const std::string state = httpGet(viewer->port().value_or(0), "/state").body;
  EXPECT_EQ(jsonNumberField(state, "width"), 320) << state;
  EXPECT_EQ(jsonNumberField(state, "targetFps"), 30) << state;
  EXPECT_EQ(jsonStringField(state, "mode"), "tracking") << state;
  EXPECT_TRUE(visualizationEnabled(state, "contact_forces").has_value())
      << state;
  EXPECT_TRUE(visualizationEnabled(state, "metrics").has_value()) << state;
}

// A frame that cannot be drawn stops the viewer, and the page learns it at
// once: frame requests answer 503 'the viewer has stopped' instead of a 204
// every 2 s forever, and /state no longer claims frames are rendered.
TEST(WebMujocoViewer, ARenderFailureStopsServingFrames) {
  const std::unique_ptr<MujocoSimInterface> sim = makeHingeSim(testOptions());
  ASSERT_NE(sim, nullptr);
  const std::unique_ptr<WebMujocoViewer> viewer =
      makeViewer(sim.get(), testOptions());
  ASSERT_NE(viewer, nullptr);
  ASSERT_THAT(viewer->start(), IsOk());
  const int port = viewer->port().value_or(0);
  const HttpReply first = httpGet(port, "/frame.jpg");
  ASSERT_EQ(first.status, 200);
  WebMujocoViewerTestPeer::failNextFrame(*viewer);
  const absl::Time start = absl::Now();
  const HttpReply next = httpGet(
      port, absl::StrCat("/frame.jpg?after=",
                         WebMujocoViewerTestPeer::latestSequence(*viewer)));
  EXPECT_EQ(next.status, 503) << next.body;
  EXPECT_THAT(next.body, HasSubstr("stopped"));
  EXPECT_LT(absl::Now() - start, absl::Seconds(1));
  EXPECT_TRUE(eventually([&viewer] { return !viewer->ok(); }));
  EXPECT_EQ(jsonNumberField(httpGet(port, "/state").body, "measuredFps"), 0.0);
}

// A page may ask for more than the offscreen buffer the GL preflight checked:
// the renderer caps it at maxWidth x maxHeight, so that no resize reaches
// mju_error.
TEST(WebMujocoViewer, SettingsBeyondTheMaximumSizeAreCapped) {
  WebViewerOptions options = testOptions();
  options.maxWidth = 640;
  options.maxHeight = 480;
  const std::unique_ptr<MujocoSimInterface> sim = makeHingeSim(options);
  ASSERT_NE(sim, nullptr);
  const std::unique_ptr<WebMujocoViewer> viewer =
      makeViewer(sim.get(), options);
  ASSERT_NE(viewer, nullptr);
  ASSERT_THAT(viewer->start(), IsOk());
  const int port = viewer->port().value_or(0);
  const HttpReply first = httpGet(port, "/frame.jpg");
  ASSERT_EQ(first.status, 200);
  ASSERT_EQ(httpPost(port, "/settings?width=3840&height=2160").status, 204);
  const std::string state = awaitState(port, [](const std::string& body) {
    return jsonNumberField(body, "width") != 320;
  });
  EXPECT_EQ(jsonNumberField(state, "width"), 640) << state;
  EXPECT_EQ(jsonNumberField(state, "height"), 480) << state;
  const HttpReply resized = httpGet(
      port, absl::StrCat("/frame.jpg?after=",
                         WebMujocoViewerTestPeer::latestSequence(*viewer)));
  ASSERT_EQ(resized.status, 200);
  const web_viewer::JpegDecoder::Image image = decode(resized.body);
  EXPECT_EQ(image.width, 640);
  EXPECT_EQ(image.height, 480);
}

// At 1 fps the render thread spends most of each second in its frame-period
// pause, which the destructor (and a stop) now cuts short; it used to sleep it
// out, up to a second, or longer after a backward step of the wall clock.
TEST(WebMujocoViewer, IsDestroyedPromptlyAtOneFramePerSecond) {
  const std::unique_ptr<MujocoSimInterface> sim = makeHingeSim(testOptions());
  ASSERT_NE(sim, nullptr);
  std::unique_ptr<WebMujocoViewer> viewer =
      makeViewer(sim.get(), testOptions());
  ASSERT_NE(viewer, nullptr);
  ASSERT_THAT(viewer->start(), IsOk());
  const int port = viewer->port().value_or(0);
  ASSERT_EQ(httpGet(port, "/frame.jpg").status, 200);
  ASSERT_EQ(httpPost(port, "/settings?fps=1").status, 204);
  ASSERT_EQ(jsonNumberField(awaitState(port,
                                       [](const std::string& body) {
                                         return jsonNumberField(
                                                    body, "targetFps") == 1;
                                       }),
                            "targetFps"),
            1);
  // A frame just published: the render thread is at the start of its pause.
  const HttpReply next = httpGet(
      port, absl::StrCat("/frame.jpg?after=",
                         WebMujocoViewerTestPeer::latestSequence(*viewer)));
  ASSERT_EQ(next.status, 200);
  const absl::Time begin = absl::Now();
  viewer.reset();
  EXPECT_LT(absl::Now() - begin, absl::Milliseconds(500));
}

TEST(WebMujocoViewer, IsDestroyedPromptlyWithAStreamOpen) {
  const std::unique_ptr<MujocoSimInterface> sim = makeHingeSim(testOptions());
  ASSERT_NE(sim, nullptr);
  std::unique_ptr<WebMujocoViewer> viewer =
      makeViewer(sim.get(), testOptions());
  ASSERT_NE(viewer, nullptr);
  ASSERT_THAT(viewer->start(), IsOk());
  MjpegStreamTestClient stream(viewer->port().value_or(0));
  const absl::Time deadline = absl::Now() + absl::Seconds(5);
  while (stream.bytesReceived() < 10'000 && absl::Now() < deadline)
    absl::SleepFor(absl::Milliseconds(10));
  ASSERT_GE(stream.bytesReceived(), 10'000) << "the stream delivered no frame";
  const absl::Time begin = absl::Now();
  viewer.reset();
  EXPECT_LT(absl::Now() - begin, absl::Seconds(3));
}

/** The CPUs the thread `tid` of this process may run on. */
std::vector<int> threadCores(int tid) {
  cpu_set_t set;
  CPU_ZERO(&set);
  std::vector<int> cores;
  if (sched_getaffinity(tid, sizeof(set), &set) != 0) return cores;
  for (int core = 0; core < CPU_SETSIZE; ++core) {
    if (CPU_ISSET(core, &set)) cores.push_back(core);
  }
  return cores;
}

/** Field `index` (1-based, as proc(5) numbers them) of
 * /proc/self/task/<tid>/stat, after the parenthesized name. */
std::string threadStatField(int tid, int index) {
  std::ifstream file(absl::StrCat("/proc/self/task/", tid, "/stat"));
  std::stringstream contents;
  contents << file.rdbuf();
  const std::string stat = contents.str();
  const size_t nameEnd = stat.rfind(')');
  if (nameEnd == std::string::npos) return "";
  const std::vector<std::string> fields =
      absl::StrSplit(stat.substr(nameEnd + 2), ' ');
  const size_t position = static_cast<size_t>(
      index - 3);  // fields 1 and 2 are the pid and the name
  return position < fields.size() ? fields[position] : "";
}

std::string threadName(int tid) {
  std::ifstream file(absl::StrCat("/proc/self/task/", tid, "/comm"));
  std::string name;
  std::getline(file, name);
  return name;
}

TEST(WebMujocoViewerPlacement,
     EveryThreadItStartsRunsOnItsCoresUnderSchedIdle) {
  const std::vector<int> allowed = threadCores(/*tid=*/0);
  if (allowed.size() < 2)
    GTEST_SKIP() << "needs two allowed cores, the process has "
                 << allowed.size();
  const std::vector<int> cores = {allowed[allowed.size() - 2],
                                  allowed[allowed.size() - 1]};

  const std::unique_ptr<MujocoSimInterface> sim = makeHingeSim(testOptions());
  ASSERT_NE(sim, nullptr);
  const std::vector<int> before = processThreadIds();
  WebViewerOptions options = testOptions();
  options.cores = cores;
  options.threadPriority = web_viewer::ViewerThreadPriority::kIdle;
  const std::unique_ptr<WebMujocoViewer> viewer =
      makeViewer(sim.get(), options);
  ASSERT_NE(viewer, nullptr);
  ASSERT_THAT(viewer->start(), IsOk());
  ASSERT_EQ(httpGet(viewer->port().value_or(0), "/frame.jpg").status, 200)
      << "a frame, so that every rendering thread exists";

  std::vector<int> started;
  for (const int tid : processThreadIds()) {
    if (!std::binary_search(before.begin(), before.end(), tid))
      started.push_back(tid);
  }
  EXPECT_GE(started.size(), 3U)
      << "the render thread, the listen thread and the request pool";
  int llvmpipeThreads = 0;
  for (const int tid : started) {
    const std::string name = threadName(tid);
    if (absl::StartsWith(name, "llvmpipe")) ++llvmpipeThreads;
    int nice = 0;
    EXPECT_TRUE(absl::SimpleAtoi(threadStatField(tid, /*index=*/19), &nice));
    int policy = -1;
    EXPECT_TRUE(absl::SimpleAtoi(threadStatField(tid, /*index=*/41), &policy));
    if (absl::StrContains(name, "disk$")) {
      // Mesa's shader disk cache queue places itself: every core, at nice 19
      // (README.md, "The web viewer").
      EXPECT_EQ(nice, 19) << "thread " << tid << " (" << name << ")";
      continue;
    }
    EXPECT_EQ(threadCores(tid), cores)
        << "thread " << tid << " (" << name << ")";
    EXPECT_EQ(policy, SCHED_IDLE) << "thread " << tid << " (" << name << ")";
  }
  int lpNumThreads = 0;
  const char* absl_nullable configured = std::getenv("LP_NUM_THREADS");
  if (configured != nullptr && !absl::SimpleAtoi(configured, &lpNumThreads))
    lpNumThreads = 0;
  EXPECT_LE(llvmpipeThreads,
            std::max(lpNumThreads, static_cast<int>(cores.size())));
}

}  // namespace
}  // namespace robot::mujoco_sim_interface
