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

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ocs2_mpc/SystemObservation.h"

#include "humanoid_common_mpc/common/ThreadAffinity.h"
#include "humanoid_common_mpc_app/robot/RemoteMpcLinkAdapter.h"
#include "humanoid_common_mpc_app/robot/RobotAppOptions.h"
#include "humanoid_nmpc/humanoid_common_mpc_app/robot/test/RobotTestSupport.h"
#include "mujoco_sim_interface/viewer/ViewerNames.h"
#include "mujoco_sim_interface/viewer/ViewerSettings.h"

/*
 * The command line of the robot binaries (the core lists, the viewer by name -
 * from --viewer, the plot configuration's mujoco_viewer or the default - and
 * the web viewer's flags) and the MpcLink over the remote link: created on a
 * bus that is not running yet, handed to the controller's factory exactly once,
 * its first observation sent.
 */

namespace ocs2::humanoid {
namespace {

TEST(RobotAppOptions, CoreLists) {
  const std::vector<int> defaults{4, 5};
  EXPECT_EQ(*parseCoreList("default", defaults), defaults);
  EXPECT_TRUE(parseCoreList("none", defaults)->empty());
  EXPECT_TRUE(parseCoreList(/*value=*/"", defaults)->empty());
  EXPECT_EQ(*parseCoreList("2, 3,7", defaults), (std::vector<int>{2, 3, 7}));
  EXPECT_FALSE(parseCoreList("2,x", defaults).ok());
  EXPECT_FALSE(parseCoreList("-1", defaults).ok());
}

/** The viewer flags at their defaults (RobotAppFlags.cpp): no --viewer and no
 * --plot_config. */
ViewerFlagValues defaultViewerFlags() {
  ViewerFlagValues flags;
  flags.viewer = "";
  flags.plotConfig = "";
  flags.webViewerAddress = "127.0.0.1";
  flags.webViewerPort = 6090;
  flags.viewerCores = "default";
  return flags;
}

/** The viewer settings of `flags` with the viewer cores of a 20-core machine; a
 * test failure and defaults when refused. */
robot::mujoco_sim_interface::ViewerSettings settingsOf(
    const ViewerFlagValues& flags) {
  const absl::StatusOr<robot::mujoco_sim_interface::ViewerSettings> settings =
      viewerSettingsFromFlags(flags, /*defaultViewerCores=*/{16, 17, 18, 19});
  if (!settings.ok()) {
    ADD_FAILURE() << settings.status();
    return robot::mujoco_sim_interface::ViewerSettings();
  }
  return *settings;
}

/** The refusal of `flags`: InvalidArgument, or a test failure. */
absl::Status refusalOf(const ViewerFlagValues& flags) {
  const absl::StatusOr<robot::mujoco_sim_interface::ViewerSettings> settings =
      viewerSettingsFromFlags(flags, /*defaultViewerCores=*/{16, 17, 18, 19});
  EXPECT_EQ(settings.status().code(), absl::StatusCode::kInvalidArgument)
      << settings.status();
  return settings.status();
}

TEST(RobotAppOptions, TheViewerIsChosenByName) {
  const robot::mujoco_sim_interface::ViewerSettings defaults =
      settingsOf(defaultViewerFlags());
  EXPECT_EQ(defaults.names, std::vector<std::string>{"web"})
      << "the default: the browser tab";
  EXPECT_EQ(defaults.web.address, "127.0.0.1") << "loopback only";
  EXPECT_EQ(defaults.web.port, 6090);
  EXPECT_TRUE(defaults.web.allowedHosts.empty());
  EXPECT_FALSE(defaults.web.eglDeviceId.has_value());

  ViewerFlagValues flags = defaultViewerFlags();
  flags.viewer = "web, none";
  EXPECT_EQ(settingsOf(flags).names, (std::vector<std::string>{"web", "none"}))
      << "falls back to none";
  flags.viewer = "web";
  EXPECT_EQ(settingsOf(flags).names, std::vector<std::string>{"web"});

  flags.viewer = "vnc";
  const absl::Status unknown = refusalOf(flags);
  EXPECT_THAT(unknown.message(), ::testing::HasSubstr("--viewer=vnc"));
  EXPECT_THAT(unknown.message(), ::testing::HasSubstr("glfw, web, none"))
      << "the valid names";
  // glfw cannot fail at start-up, so nothing after it could ever be used;
  // nothing follows none either.
  for (const char* absl_nonnull list :
       {"glfw,none", "glfw,web", "none,web", "web,web", "web,"}) {
    flags.viewer = list;
    EXPECT_FALSE(viewerSettingsFromFlags(flags, /*defaultViewerCores=*/{}).ok())
        << list;
  }
  flags.viewer = "glfw,none";
  EXPECT_THAT(refusalOf(flags).message(),
              ::testing::HasSubstr("cannot fail at start-up"));
  // Blank is not given: the default, as the launch files' empty `viewer` passes
  // it.
  flags.viewer = "  ";
  EXPECT_EQ(settingsOf(flags).names, std::vector<std::string>{"web"});
}

/** The header of a plot configuration
 * (humanoid_nmpc/humanoid_rerun_viewer/proto/plot_config.proto), two lines. */
constexpr char kPlotConfigHeader[] =
    "# proto-file: "
    "humanoid_nmpc/humanoid_rerun_viewer/proto/plot_config.proto\n"
    "# proto-message: humanoid_rerun_viewer_proto.PlotConfig\n";

/** Writes `text` to the file `name` of the test's scratch directory, and
 * returns the file's path. */
std::string writeScratchFile(absl::string_view name, absl::string_view text) {
  const std::string path =
      (std::filesystem::path(std::getenv("TEST_TMPDIR")) / std::string(name))
          .string();
  std::ofstream(path, std::ios::trunc) << text;
  return path;
}

// The precedence: an explicit --viewer (the Makefile's VIEWER=,
// WB_ROBOT_VIEWER), else the plot configuration's mujoco_viewer, else the web
// viewer.
TEST(RobotAppOptions,
     WithoutViewerThePlotConfigurationChoosesAndAnExplicitViewerWins) {
  ViewerFlagValues flags = defaultViewerFlags();
  flags.plotConfig = writeScratchFile(
      "glfw.textproto",
      absl::StrCat(kPlotConfigHeader,
                   "signals: \"base_pose/*\"\nmujoco_viewer: \"glfw\"\n"));
  EXPECT_EQ(settingsOf(flags).names, std::vector<std::string>{"glfw"})
      << "the field, without --viewer";
  flags.viewer = "none";
  EXPECT_EQ(settingsOf(flags).names, std::vector<std::string>{"none"})
      << "an explicit --viewer wins";
  // Nor is the file read then: one the robot would refuse changes nothing.
  flags.plotConfig = writeScratchFile(
      "refused.textproto", "mujoco_viewer: \"vnc\"\nno_such_field: 1\n");
  EXPECT_EQ(settingsOf(flags).names, std::vector<std::string>{"none"});

  flags = defaultViewerFlags();
  flags.plotConfig = writeScratchFile(
      "web_none.textproto",
      absl::StrCat(kPlotConfigHeader, "mujoco_viewer: [\"web\", \"none\"]\n"));
  EXPECT_EQ(settingsOf(flags).names, (std::vector<std::string>{"web", "none"}))
      << "a preference list";
  flags.plotConfig = writeScratchFile(
      "no_viewer.textproto",
      absl::StrCat(kPlotConfigHeader, "signals: \"base_pose/*\"\n"));
  EXPECT_EQ(settingsOf(flags).names, std::vector<std::string>{"web"})
      << "a file without the field: the default";
  flags.plotConfig = "";
  EXPECT_EQ(settingsOf(flags).names, std::vector<std::string>{"web"})
      << "no plot configuration: the default";
  EXPECT_EQ(kDefaultViewerName, robot::mujoco_sim_interface::kWebViewerName);
}

TEST(RobotAppOptions, ABadPlotConfigurationIsRefusedSayingWhereAndWhy) {
  ViewerFlagValues flags = defaultViewerFlags();
  flags.plotConfig = writeScratchFile(
      "vnc.textproto",
      absl::StrCat(kPlotConfigHeader, "mujoco_viewer: \"vnc\"\n"));
  const absl::Status unknown = refusalOf(flags);
  EXPECT_THAT(unknown.message(),
              ::testing::HasSubstr(absl::StrCat(
                  "--plot_config=", flags.plotConfig, ": mujoco_viewer: ")));
  EXPECT_THAT(unknown.message(), ::testing::HasSubstr("'vnc'"));
  EXPECT_THAT(unknown.message(), ::testing::HasSubstr("glfw, web, none"))
      << "the valid names";
  flags.plotConfig = writeScratchFile(
      "none_web.textproto",
      absl::StrCat(kPlotConfigHeader, "mujoco_viewer: [\"none\", \"web\"]\n"));
  EXPECT_THAT(refusalOf(flags).message(),
              ::testing::HasSubstr("nothing may follow it"));
  flags.plotConfig = writeScratchFile(
      "empty_name.textproto",
      absl::StrCat(kPlotConfigHeader, "mujoco_viewer: \"\"\n"));
  EXPECT_THAT(refusalOf(flags).message(),
              ::testing::HasSubstr("unknown viewer ''"));

  // Read strictly: a misspelled field is refused at its line and column, and so
  // is another message's file.
  const std::string typo = writeScratchFile(
      "typo.textproto",
      absl::StrCat(kPlotConfigHeader, "mujoco_viwer: \"web\"\n"));
  flags.plotConfig = typo;
  EXPECT_THAT(refusalOf(flags).message(),
              ::testing::HasSubstr(absl::StrCat(typo, ":3:1: ")));
  flags.plotConfig = writeScratchFile(
      "task.textproto",
      "# proto-message: humanoid_mpc_config.TaskFile\n\nmujoco_viewer: "
      "\"web\"\n");
  EXPECT_THAT(refusalOf(flags).message(),
              ::testing::HasSubstr("humanoid_rerun_viewer_proto.PlotConfig"));

  flags.plotConfig = absl::StrCat(typo, ".missing");
  const absl::StatusOr<robot::mujoco_sim_interface::ViewerSettings> missing =
      viewerSettingsFromFlags(flags, /*defaultViewerCores=*/{});
  EXPECT_EQ(missing.status().code(), absl::StatusCode::kNotFound)
      << missing.status();
  EXPECT_THAT(
      missing.status().message(),
      ::testing::HasSubstr(absl::StrCat("--plot_config=", flags.plotConfig)));
}

// make launch-<robot>-sim opens the web viewer from the shipped plot
// configuration alone.
TEST(RobotAppOptions, TheShippedPlotConfigurationStartsTheWebViewer) {
  ViewerFlagValues flags = defaultViewerFlags();
  flags.plotConfig =
      "humanoid_nmpc/humanoid_rerun_viewer/config/plot_config.textproto";
  EXPECT_EQ(settingsOf(flags).names, std::vector<std::string>{"web"});
}

TEST(RobotAppOptions, TheWebViewerFlagsAreChecked) {
  ViewerFlagValues flags = defaultViewerFlags();
  for (const int port : {-1, 70000}) {
    flags.webViewerPort = port;
    EXPECT_THAT(refusalOf(flags).message(),
                ::testing::HasSubstr("--web_viewer_port"))
        << port;
  }
  flags.webViewerPort = 0;
  EXPECT_EQ(settingsOf(flags).web.port, 0) << "any free port";

  flags = defaultViewerFlags();
  flags.webViewerAddress = "  ";
  EXPECT_THAT(refusalOf(flags).message(),
              ::testing::HasSubstr("--web_viewer_address"));
  flags.webViewerAddress = "0.0.0.0";
  EXPECT_EQ(settingsOf(flags).web.address, "0.0.0.0");

  flags = defaultViewerFlags();
  flags.webViewerAllowedHosts = "a, b,,robot.lan ";
  EXPECT_EQ(settingsOf(flags).web.allowedHosts,
            (std::vector<std::string>{"a", "b", "robot.lan"}));
}

TEST(RobotAppOptions, TheViewerCoresDefaultToTheAllocationsViewerCores) {
  const SystemCoreAllocation twenty = coreAllocationFor(
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19});
  ViewerFlagValues flags = defaultViewerFlags();
  absl::StatusOr<robot::mujoco_sim_interface::ViewerSettings> settings =
      viewerSettingsFromFlags(flags, twenty.viewerCores);
  ASSERT_TRUE(settings.ok()) << settings.status();
  EXPECT_EQ(settings->web.cores, (std::vector<int>{16, 17, 18, 19}))
      << "the cores no other partition takes";
  flags.viewerCores = "none";
  EXPECT_TRUE(settingsOf(flags).web.cores.empty()) << "unpinned";
  flags.viewerCores = "2,3";
  EXPECT_EQ(settingsOf(flags).web.cores, (std::vector<int>{2, 3}));
  flags.viewerCores = "fast";
  EXPECT_THAT(refusalOf(flags).message(),
              ::testing::HasSubstr("--viewer_cores=fast"));
}

// LP_NUM_THREADS: one llvmpipe rasterizer thread per web viewer core (at least
// 2) when the environment sets none, and nothing for a list without the web
// viewer, so that the GLFW window keeps Mesa's default.
TEST(RobotAppOptions, TheRasterizerThreadsFitTheWebViewersCores) {
  robot::mujoco_sim_interface::ViewerSettings web;
  web.names = {"web", "none"};
  web.web.cores = {16, 17, 18, 19};
  EXPECT_EQ(softwareRasterizerThreadsToSet(web, /*current=*/nullptr), "4");
  EXPECT_EQ(softwareRasterizerThreadsToSet(web, /*current=*/""), "4")
      << "an empty value is Mesa's default too";
  EXPECT_EQ(softwareRasterizerThreadsToSet(web, /*current=*/"8"), std::nullopt)
      << "the environment's value stays";
  web.web.cores = {3};
  EXPECT_EQ(softwareRasterizerThreadsToSet(web, /*current=*/nullptr), "2");
  web.web.cores = {};
  EXPECT_EQ(softwareRasterizerThreadsToSet(web, /*current=*/nullptr), "2")
      << "unpinned";

  robot::mujoco_sim_interface::ViewerSettings glfw;
  glfw.names = {"glfw"};
  EXPECT_EQ(softwareRasterizerThreadsToSet(glfw, /*current=*/nullptr),
            std::nullopt);
  glfw.names = {"none"};
  EXPECT_EQ(softwareRasterizerThreadsToSet(glfw, /*current=*/nullptr),
            std::nullopt);
}

// The start-up log line of LP_NUM_THREADS counts the viewer's cores, or says
// that it has none.
TEST(RobotAppOptions, TheRasterizerThreadsLogLineNamesTheViewerCores) {
  robot::mujoco_sim_interface::ViewerSettings web;
  web.names = {"web"};
  web.web.cores = {16, 17, 18, 19};
  EXPECT_EQ(
      describeSoftwareRasterizerThreads(web, /*threads=*/"4"),
      "LP_NUM_THREADS=4: llvmpipe's rasterizer threads, for 4 viewer core(s).");
  web.web.cores = {3};
  EXPECT_EQ(
      describeSoftwareRasterizerThreads(web, /*threads=*/"2"),
      "LP_NUM_THREADS=2: llvmpipe's rasterizer threads, for 1 viewer core(s).");
  web.web.cores = {};
  EXPECT_EQ(describeSoftwareRasterizerThreads(web, /*threads=*/"2"),
            "LP_NUM_THREADS=2: llvmpipe's rasterizer threads (viewer cores "
            "unrestricted).");
}

TEST(RobotAppOptions, TheEglDeviceComesFromTheEnvironment) {
  ViewerFlagValues flags = defaultViewerFlags();
  flags.eglDeviceId = "1";
  EXPECT_EQ(settingsOf(flags).web.eglDeviceId, 1);
  for (const char* absl_nonnull value : {"gpu", "-1", ""}) {
    flags.eglDeviceId = value;
    EXPECT_THAT(refusalOf(flags).message(),
                ::testing::HasSubstr("MUJOCO_EGL_DEVICE_ID"))
        << value;
  }
}

TEST(RemoteMpcLinkAdapter, IsCreatedBeforeTheBusRunsAndHandedOverOnce) {
  std::unique_ptr<robot::ipc::Bus> bus = robot_test::createLoopbackBus("robot");
  ipc::RemoteMpcLink::Config config;
  config.dimensions =
      ipc::ModelDimensions{.stateDim = 3, .inputDim = 2, .numModes = 4};
  absl::StatusOr<std::unique_ptr<RemoteMpcLinkAdapter>> adapter =
      RemoteMpcLinkAdapter::Create(bus.get(), config);
  ASSERT_TRUE(adapter.ok()) << adapter.status();
  RemoteMpcLinkAdapter* absl_nonnull link = adapter->get();
  EXPECT_TRUE(link->isHealthy());
  EXPECT_FALSE(link->initialPolicyReceived());

  const MpcLinkFactory factory = handOverMpcLink(*std::move(adapter));
  std::unique_ptr<MpcLink> taken = factory(/*resetTarget=*/nullptr);
  EXPECT_EQ(taken.get(), link);
  EXPECT_EQ(factory(/*resetTarget=*/nullptr), nullptr)
      << "the link is handed over once";

  // The first observation goes into the link's triple buffer; the bus's IO
  // thread publishes it once the bus runs.
  SystemObservation observation;
  observation.time = 1.0;
  observation.state = vector_t::Zero(3);
  observation.input = vector_t::Zero(2);
  taken->start(observation);
  ASSERT_TRUE(bus->start().ok());
  EXPECT_TRUE(robot_test::waitFor(
      [&]() { return link->remote().statistics().observationsPublished > 0; }));
  bus->stop();

  // A running bus takes no more subscriptions.
  std::unique_ptr<robot::ipc::Bus> running =
      robot_test::createLoopbackBus("robot");
  ASSERT_TRUE(running->start().ok());
  EXPECT_FALSE(RemoteMpcLinkAdapter::Create(running.get(), config).ok());
  running->stop();
}

}  // namespace
}  // namespace ocs2::humanoid
