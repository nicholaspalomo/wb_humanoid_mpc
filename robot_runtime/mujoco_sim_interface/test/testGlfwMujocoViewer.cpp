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

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "GLFW/glfw3.h"
#include "absl/base/nullability.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"
#include "mujoco/mujoco.h"

#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "mujoco_sim_interface/viewer/GlfwMujocoViewer.h"
#include "mujoco_sim_interface/viewer/ViewerNames.h"
#include "mujoco_sim_interface/viewer/WebViewerOptions.h"
#include "robot_runtime/mujoco_sim_interface/test/ViewerTestSupport.h"

/*
 * The `glfw` viewer: how it fails without a display (it logs why and the simulation runs on), the keys it hands to the
 * frame renderer, and that its window shows exactly what the web viewer's offscreen buffer shows for the same state -
 * the window on an Xvfb of the test's own, both rendered by llvmpipe.
 */

namespace robot::mujoco_sim_interface {

/** Reaches into a GlfwMujocoViewer for the parity test: one capture of its window's back buffer. */
class GlfwMujocoViewerTestPeer {
 public:
  struct Capture {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgb;  // rows bottom-up
  };

  /** Asks the render loop for one capture and waits up to `timeout` for it. */
  static absl::StatusOr<Capture> captureWindow(GlfwMujocoViewer& viewer, absl::Duration timeout) {
    {
      absl::MutexLock lock(viewer.captureMutex_);
      viewer.capture_.reset();
    }
    viewer.captureRequested_.store(true);
    absl::MutexLock lock(viewer.captureMutex_);
    if (!viewer.captureMutex_.AwaitWithTimeout(absl::Condition(&hasCapture, &viewer.capture_), timeout) || !viewer.capture_.has_value()) {
      return absl::DeadlineExceededError("the render loop served no capture");
    }
    return Capture{.width = viewer.capture_->width, .height = viewer.capture_->height, .rgb = std::move(viewer.capture_->rgb)};
  }

 private:
  static bool hasCapture(std::optional<GlfwMujocoViewer::WindowCapture>* absl_nonnull capture) { return capture->has_value(); }
};

namespace {

/** True when GLFW could find a display here, in which case the viewer would really start. */
bool hasDisplay() {
  return std::getenv("DISPLAY") != nullptr || std::getenv("WAYLAND_DISPLAY") != nullptr;
}

TEST(GlfwMujocoViewer, WithoutADisplayTheViewerStopsAndSaysSo) {
  if (hasDisplay()) GTEST_SKIP() << "a display is available, so the viewer would start";
  absl::StatusOr<std::unique_ptr<MujocoSimInterface>> sim = MujocoSimInterface::Create(hingeSimConfig({kNoViewerName}), hingeUrdfPath());
  ASSERT_TRUE(sim.ok()) << sim.status();
  {
    GlfwMujocoViewer viewer(sim->get());
    EXPECT_TRUE(viewer.ok());
    EXPECT_TRUE(viewer.start().ok()) << "the GLFW viewer does not wait for its window";
    viewer.waitForInit();
    EXPECT_FALSE(viewer.ok()) << "the viewer reported a window it could not create";
  }  // The destructor joins the render thread without touching the window that was never created.
  (*sim)->simulationStep();
}

TEST(GlfwMujocoViewer, ASimulatorWithAViewerButNoDisplayRunsOn) {
  if (hasDisplay()) GTEST_SKIP() << "a display is available, so the viewer would start";
  absl::StatusOr<std::unique_ptr<MujocoSimInterface>> sim = MujocoSimInterface::Create(hingeSimConfig({kGlfwViewerName}), hingeUrdfPath());
  ASSERT_TRUE(sim.ok()) << sim.status();
  // This used to end the process: the viewer's GLFW start-up failure was a MuJoCo fatal error.
  (*sim)->initSim();
  EXPECT_TRUE((*sim)->viewerStatus().ok()) << "the GLFW viewer cannot fail at start-up: " << (*sim)->viewerStatus();
  EXPECT_EQ((*sim)->activeViewerName(), kGlfwViewerName);
  for (int step = 0; step < 10; ++step) (*sim)->simulationStep();
  (*sim)->updateInterfaceStateFromRobot();
  EXPECT_GT((*sim)->getRobotState().getTime(), 0.0) << "the simulation clock advanced";
}

TEST(GlfwMujocoViewer, MapsTheKeysOfTheViewerToTheirCharacters) {
  for (int group = 0; group < kNumToggledGeomGroups; ++group) {
    EXPECT_EQ(viewerKeyFromGlfwKey(GLFW_KEY_0 + group), static_cast<char>('0' + group));
  }
  EXPECT_EQ(viewerKeyFromGlfwKey(GLFW_KEY_6), std::nullopt) << "there are only six geometry groups";
  EXPECT_EQ(viewerKeyFromGlfwKey(GLFW_KEY_9), std::nullopt);
  EXPECT_EQ(viewerKeyFromGlfwKey(GLFW_KEY_A), 'a');
  EXPECT_EQ(viewerKeyFromGlfwKey(GLFW_KEY_K), 'k');
  EXPECT_EQ(viewerKeyFromGlfwKey(GLFW_KEY_Z), 'z');
  EXPECT_EQ(viewerKeyFromGlfwKey(GLFW_KEY_SPACE), std::nullopt);
  EXPECT_EQ(viewerKeyFromGlfwKey(GLFW_KEY_LEFT_SHIFT), std::nullopt);
  EXPECT_EQ(viewerKeyFromGlfwKey(GLFW_KEY_F1), std::nullopt);
}

/** An Xvfb of the test's own, on the display number it picks itself (-displayfd); stopped by the destructor. */
class XvfbProcess {
 public:
  /** Starts Xvfb with a screen of `width` x `height`; NotFound when it is not installed. */
  static absl::StatusOr<std::unique_ptr<XvfbProcess>> Start(int width, int height) {
    const char* absl_nonnull binary = "/usr/bin/Xvfb";
    if (access(binary, X_OK) != 0) return absl::NotFoundError("there is no Xvfb (the xvfb package)");
    std::array<int, 2> pipe = {-1, -1};
    if (pipe2(pipe.data(), O_CLOEXEC) != 0) return absl::InternalError("pipe2 failed");
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, pipe[1], /*newfd=*/3);  // dup2 clears close-on-exec on fd 3
    const std::string screen = absl::StrCat(width, "x", height, "x24");
    std::vector<std::string> arguments = {binary, "-displayfd", "3", "-nolisten", "tcp", "-screen", "0", screen};
    std::vector<char* absl_nullable> argv;
    for (std::string& argument : arguments) argv.push_back(argument.data());
    argv.push_back(nullptr);
    pid_t pid = 0;
    const int spawned = posix_spawn(&pid, binary, &actions, /*attrp=*/nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    close(pipe[1]);
    if (spawned != 0) {
      close(pipe[0]);
      return absl::InternalError(absl::StrCat("posix_spawn of Xvfb failed: ", spawned));
    }
    std::unique_ptr<XvfbProcess> xvfb = absl::WrapUnique(new XvfbProcess(pid));
    std::string written;
    const absl::Time deadline = absl::Now() + absl::Seconds(10);
    while (!absl::StrContains(written, '\n') && absl::Now() < deadline) {
      pollfd readable = {.fd = pipe[0], .events = POLLIN, .revents = 0};
      if (poll(&readable, /*nfds=*/1, /*timeout=*/100) <= 0) continue;
      std::array<char, 16> buffer = {};
      const ssize_t count = read(pipe[0], buffer.data(), buffer.size());
      if (count <= 0) break;
      written.append(buffer.data(), static_cast<size_t>(count));
    }
    close(pipe[0]);
    if (!absl::SimpleAtoi(absl::StripAsciiWhitespace(written), &xvfb->display_)) {
      return absl::UnavailableError(absl::StrCat("Xvfb reported no display number ('", written, "')"));
    }
    return xvfb;
  }

  XvfbProcess(const XvfbProcess&) = delete;
  XvfbProcess& operator=(const XvfbProcess&) = delete;
  XvfbProcess(XvfbProcess&&) = delete;
  XvfbProcess& operator=(XvfbProcess&&) = delete;

  ~XvfbProcess() {
    kill(pid_, SIGTERM);
    int status = 0;
    waitpid(pid_, &status, /*options=*/0);
  }

  /** The DISPLAY value of the server, ":<number>". */
  std::string display() const { return absl::StrCat(":", display_); }

 private:
  explicit XvfbProcess(pid_t pid) : pid_(pid) {}

  pid_t pid_;
  int display_ = -1;
};

/** Points DISPLAY at `display` for its lifetime, then puts back what was there. */
class ScopedDisplay {
 public:
  explicit ScopedDisplay(const std::string& display) {
    const char* absl_nullable previous = std::getenv("DISPLAY");
    if (previous != nullptr) previous_ = std::string(previous);
    setenv("DISPLAY", display.c_str(), /*replace=*/1);
  }
  ScopedDisplay(const ScopedDisplay&) = delete;
  ScopedDisplay& operator=(const ScopedDisplay&) = delete;
  ScopedDisplay(ScopedDisplay&&) = delete;
  ScopedDisplay& operator=(ScopedDisplay&&) = delete;
  ~ScopedDisplay() {
    if (previous_.has_value()) {
      setenv("DISPLAY", previous_->c_str(), /*replace=*/1);
    } else {
      unsetenv("DISPLAY");
    }
  }

 private:
  std::optional<std::string> previous_;
};

TEST(GlfwMujocoViewer, TheWindowShowsWhatTheOffscreenBufferShows) {
  absl::StatusOr<std::unique_ptr<XvfbProcess>> xvfb = XvfbProcess::Start(/*width=*/1920, /*height=*/1080);
  if (absl::IsNotFound(xvfb.status())) {
    // On CI (the CI variable set) a skipped test would pass without having run.
    ASSERT_EQ(std::getenv("CI"), nullptr) << xvfb.status() << "; on CI this test must run: install xvfb (dependencies.txt)";
    GTEST_SKIP() << xvfb.status();
  }
  ASSERT_TRUE(xvfb.ok()) << xvfb.status();
  // Before any thread of this test starts: setenv is not thread-safe.
  const ScopedDisplay display((*xvfb)->display());

  absl::StatusOr<std::unique_ptr<MujocoSimInterface>> sim = MujocoSimInterface::Create(hingeSimConfig({kNoViewerName}), hingeUrdfPath());
  ASSERT_TRUE(sim.ok()) << sim.status();
  GlfwMujocoViewerTestPeer::Capture window;
  {
    GlfwMujocoViewer viewer(sim->get());
    ASSERT_TRUE(viewer.start().ok());
    viewer.waitForInit();
    ASSERT_TRUE(viewer.ok()) << "the GLFW viewer did not open its window on " << (*xvfb)->display();
    // Some 60 frames of the same state, by which the tracking camera, which eases towards the robot, has settled.
    absl::SleepFor(absl::Seconds(1));
    absl::StatusOr<GlfwMujocoViewerTestPeer::Capture> captured = GlfwMujocoViewerTestPeer::captureWindow(viewer, absl::Seconds(10));
    ASSERT_TRUE(captured.ok()) << captured.status();
    window = *std::move(captured);
  }
  ASSERT_GT(window.width, 0);
  ASSERT_GT(window.height, 0);
  ASSERT_EQ(window.rgb.size(), static_cast<size_t>(window.width) * window.height * 3);

  // The window has no multisampling, and shadows of the model's own size.
  WebViewerOptions options;
  options.maxWidth = std::max(window.width, kMinWebViewerWidth);
  options.maxHeight = std::max(window.height, kMinWebViewerHeight);
  options.msaaSamples = 0;
  options.shadowSize = (*sim)->getModel()->vis.quality.shadowsize;
  absl::StatusOr<std::unique_ptr<OffscreenTestRenderer>> offscreen = OffscreenTestRenderer::Create(sim->get(), options);
  ASSERT_TRUE(offscreen.ok()) << offscreen.status();
  absl::StatusOr<std::vector<uint8_t>> frame = absl::UnknownError("not rendered");
  for (int warmUp = 0; warmUp < 60; ++warmUp) frame = (*offscreen)->render(window.width, window.height);
  ASSERT_TRUE(frame.ok()) << frame.status();

  // Both are llvmpipe, so the pixels agree but for a few on depth-sensitive edges - the floor's far edge and the
  // shadows' - where the window's 24-bit depth buffer and MuJoCo's offscreen one round differently.
  int64_t equal = 0;
  int64_t close = 0;
  int largestDifference = 0;
  for (size_t pixel = 0; pixel + 2 < window.rgb.size(); pixel += 3) {
    int difference = 0;
    for (size_t channel = 0; channel < 3; ++channel) {
      difference =
          std::max(difference, std::abs(static_cast<int>(window.rgb[pixel + channel]) - static_cast<int>((*frame)[pixel + channel])));
    }
    largestDifference = std::max(largestDifference, difference);
    if (difference == 0) ++equal;
    if (difference <= 2) ++close;
  }
  const double pixels = static_cast<double>(window.rgb.size() / 3);
  EXPECT_GE(static_cast<double>(equal) / pixels, 0.995) << "of " << window.width << "x" << window.height << " pixels";
  EXPECT_GE(static_cast<double>(close) / pixels, 0.9995)
      << "within 2 levels of " << window.width << "x" << window.height << " pixels; the largest difference is " << largestDifference;
  EXPECT_GT(byteVariance(window.rgb), 100.0) << "the window was uniform";
  if (HasFailure()) {
    writeTestImage("parity_window", window.rgb, window.width, window.height);
    writeTestImage("parity_offscreen", *frame, window.width, window.height);
  }
}

}  // namespace
}  // namespace robot::mujoco_sim_interface
