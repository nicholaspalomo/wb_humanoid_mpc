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

#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"

#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "mujoco_sim_interface/viewer/ViewerNames.h"
#include "robot_runtime/mujoco_sim_interface/test/ViewerTestSupport.h"
#include "robot_runtime/mujoco_sim_interface/test/WebViewerTestClient.h"

/*
 * The web viewer does not slow the simulation down. The Atlas simulator runs in
 * real time with its physics thread pinned as the robot app pins it, once
 * without a viewer and once with the web viewer (its threads under SCHED_IDLE,
 * as shipped) and a browser pulling frames, with the contact timeline and the
 * target contact patches drawn (the timeline is the one lock the physics thread
 * shares with the viewer). The physics loop's per-step lateness
 * (StepLatenessHistogram) and its real-time factor must stay where they are
 * without the viewer. The bounds are calibrated in README.md ("Viewers").
 *
 * Each test is a placement of ThreadAffinity.h's default viewer cores and runs
 * in a process of its own (BUILD.bazel), since llvmpipe's rasterizer threads
 * are started once per process and keep the cores and the scheduling of the
 * first render thread.
 */

namespace robot::mujoco_sim_interface {
namespace {

// LINT.IfChange(sim_rate_bounds)
constexpr absl::Duration kP99Margin = absl::Microseconds(250);
// A step later than this counts as a late step. The machine preempts the
// physics thread for 2 to 4 ms in about one run in ten with or without the
// viewer (README.md), and each such preemption makes some ten steps late in a
// row, so the share of late steps and a generous maximum tell a viewer that
// delays the physics from that noise.
constexpr absl::Duration kLateStep = absl::Milliseconds(1);
constexpr double kLateStepShareMargin = 0.005;  // of the steps
constexpr absl::Duration kMaxMargin = absl::Milliseconds(10);
constexpr double kRealTimeFactorRatio = 0.98;
// LINT.ThenChange(//robot_runtime/mujoco_sim_interface/README.md:sim_rate_bounds)
constexpr absl::Duration kWarmUp = absl::Milliseconds(500);
constexpr absl::Duration kWindow = absl::Seconds(3);

/** What one run of the simulator measured over kWindow. */
struct Phase {
  absl::Duration p99;
  absl::Duration max;
  int64_t lateSteps = 0;  // later than kLateStep
  double realTimeFactor = 0.0;
  int64_t steps = 0;
  int64_t frames = 0;
  int64_t droppedTimelineSamples = 0;
};

std::vector<int> allowedCores() {
  cpu_set_t set;
  CPU_ZERO(&set);
  std::vector<int> cores;
  if (sched_getaffinity(/*pid=*/0, sizeof(set), &set) != 0) return cores;
  for (int core = 0; core < CPU_SETSIZE; ++core) {
    if (CPU_ISSET(core, &set)) cores.push_back(core);
  }
  return cores;
}

bool pin(pthread_t thread, const std::vector<int>& cores) {
  cpu_set_t set;
  CPU_ZERO(&set);
  for (const int core : cores) CPU_SET(core, &set);
  return pthread_setaffinity_np(thread, sizeof(set), &set) == 0;
}

/** Where one run puts the physics thread, the viewer's threads and the browser
 * pulling frames. */
struct Placement {
  std::vector<int> physicsCores;
  std::vector<int> viewerCores;
  std::vector<int> browserCores;
};

/** Runs Atlas in real time with `viewer`, its threads where `placement` says;
 * pulls frames for web. */
Phase run(absl::string_view viewer, const Placement& placement) {
  MujocoSimConfig config = atlasSimConfig(
      {std::string(viewer)},
      {"contact_timeline", "target_contact_patches", "contact_forces"});
  config.viewer.web.port = 0;
  config.viewer.web.cores = placement.viewerCores;
  absl::StatusOr<std::unique_ptr<MujocoSimInterface>> created =
      MujocoSimInterface::Create(config, kAtlasUrdfPath);
  EXPECT_TRUE(created.ok()) << created.status();
  if (!created.ok()) return {};
  MujocoSimInterface& sim = **created;
  sim.initSim();
  EXPECT_TRUE(sim.viewerStatus().ok()) << sim.viewerStatus();
  sim.setTargetContactFlags({true, false});
  sim.startSim();
  EXPECT_TRUE(
      pin(sim.getSimulationThread().native_handle(), placement.physicsCores));

  std::atomic<bool> stop = false;
  std::atomic<int64_t> frames = 0;
  std::thread browser;
  if (const std::optional<int> port = sim.webViewerPort(); port.has_value()) {
    browser = std::thread([&stop, &frames, &placement, port = *port] {
      // A browser runs elsewhere; on the physics' cores, this thread would
      // measure itself instead of the viewer.
      EXPECT_TRUE(pin(pthread_self(), placement.browserCores));
      int64_t after = 0;
      while (!stop.load()) {
        const HttpReply frame =
            httpGet(port, absl::StrCat("/frame.jpg?after=", after));
        if (frame.status != 200) {
          absl::SleepFor(absl::Milliseconds(10));
          continue;
        }
        after = frame.frameSequence;
        frames.fetch_add(1);
      }
    });
  }
  absl::SleepFor(kWarmUp);
  sim.resetStepLateness();
  frames.store(0);
  const absl::Time begin = absl::Now();
  absl::SleepFor(kWindow);
  Phase phase;
  phase.steps = sim.stepLateness().count();
  phase.realTimeFactor = static_cast<double>(phase.steps) * config.dt /
                         absl::ToDoubleSeconds(absl::Now() - begin);
  phase.p99 = sim.stepLateness().percentile(/*fraction=*/0.99);
  phase.max = sim.stepLateness().max();
  phase.lateSteps = sim.stepLateness().countLaterThan(kLateStep);
  phase.frames = frames.load();
  phase.droppedTimelineSamples = sim.droppedContactTimelineSamples();
  stop.store(true);
  if (browser.joinable()) browser.join();
  return phase;
}

std::string describe(const Phase& phase) {
  return absl::StrCat(
      "p99 ", absl::FormatDuration(phase.p99), ", max ",
      absl::FormatDuration(phase.max), ", ", phase.lateSteps,
      " steps later than ", absl::FormatDuration(kLateStep), ", RTF ",
      phase.realTimeFactor, ", ", phase.steps, " steps, ", phase.frames,
      " frames, ", phase.droppedTimelineSamples, " timeline samples dropped");
}

double lateStepShare(const Phase& phase) {
  return static_cast<double>(phase.lateSteps) /
         static_cast<double>(std::max<int64_t>(phase.steps, 1));
}

/** Runs Atlas without a viewer and with the web viewer, both placed as
 * `placement` says, and compares the two. */
void expectTheWebViewerDoesNotDelayThePhysics(const Placement& placement) {
  const Phase without = run(kNoViewerName, placement);
  const Phase with = run(kWebViewerName, placement);
  ::testing::Test::RecordProperty("without_viewer", describe(without));
  ::testing::Test::RecordProperty("with_web_viewer", describe(with));
  absl::PrintF(
      "physics on %s, viewer on %s\nwithout a viewer: %s\nwith the web viewer: "
      "%s\n",
      absl::StrJoin(placement.physicsCores, ","),
      absl::StrJoin(placement.viewerCores, ","), describe(without),
      describe(with));

  EXPECT_LE(with.p99, without.p99 + kP99Margin)
      << describe(with) << " against " << describe(without);
  EXPECT_LE(with.max, without.max + kMaxMargin)
      << describe(with) << " against " << describe(without);
  EXPECT_LE(lateStepShare(with), lateStepShare(without) + kLateStepShareMargin)
      << describe(with) << " against " << describe(without);
  EXPECT_GE(with.realTimeFactor, kRealTimeFactorRatio * without.realTimeFactor)
      << describe(with) << " against " << describe(without);
  EXPECT_GE(with.frames, 10) << "the browser hardly pulled any frames, so the "
                                "viewer was not exercised";
  // The physics thread never waits for the viewer; a sample it cannot append is
  // kept for the next, and dropped only if the viewer holds the timeline
  // through two sample periods.
  EXPECT_LE(with.droppedTimelineSamples, with.steps / 100) << describe(with);
}

/** The process's allowed cores, or empty (and the test skipped) under 4. */
std::vector<int> fourOrMoreAllowedCores() {
  std::vector<int> allowed = allowedCores();
  if (allowed.size() < 4) allowed.clear();
  return allowed;
}

// LINT.IfChange(default_placements)
// Above 16 cores: the viewer on cores of its own (ThreadAffinity.h's cores past
// the 16 the partitions name), the browser on the viewer's.
TEST(WebViewerSimRate, TheWebViewerOnCoresOfItsOwnDoesNotDelayThePhysics) {
  const std::vector<int> allowed = fourOrMoreAllowedCores();
  if (allowed.empty())
    GTEST_SKIP()
        << "needs 4 allowed cores (physics and viewer on disjoint pairs)";
  const std::vector<int> viewerCores = {allowed[allowed.size() - 2],
                                        allowed[allowed.size() - 1]};
  expectTheWebViewerDoesNotDelayThePhysics(
      {.physicsCores = {allowed[0], allowed[1]},
       .viewerCores = viewerCores,
       .browserCores = viewerCores});
}

// 8 to 16 cores: the viewer on the physics' two (or four) simulation cores.
TEST(WebViewerSimRate, TheWebViewerOnThePhysicsCoresDoesNotDelayThePhysics) {
  const std::vector<int> allowed = fourOrMoreAllowedCores();
  if (allowed.empty())
    GTEST_SKIP() << "needs 4 allowed cores (physics and viewer on a pair, the "
                    "browser on another)";
  const std::vector<int> physicsCores = {allowed[0], allowed[1]};
  expectTheWebViewerDoesNotDelayThePhysics(
      {.physicsCores = physicsCores,
       .viewerCores = physicsCores,
       .browserCores = {allowed[allowed.size() - 2],
                        allowed[allowed.size() - 1]}});
}

// 4 to 7 cores: the viewer on the physics' one simulation core.
TEST(WebViewerSimRate, TheWebViewerOnThePhysicsCoreDoesNotDelayThePhysics) {
  const std::vector<int> allowed = fourOrMoreAllowedCores();
  if (allowed.empty())
    GTEST_SKIP() << "needs 4 allowed cores (physics and viewer on one, the "
                    "browser on two others)";
  expectTheWebViewerDoesNotDelayThePhysics(
      {.physicsCores = {allowed[0]},
       .viewerCores = {allowed[0]},
       .browserCores = {allowed[allowed.size() - 2],
                        allowed[allowed.size() - 1]}});
}
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/common/ThreadAffinity.h:viewer_cores, //robot_runtime/mujoco_sim_interface/BUILD.bazel:sim_rate_targets)
// clang-format on

}  // namespace
}  // namespace robot::mujoco_sim_interface
