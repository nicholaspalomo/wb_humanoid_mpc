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

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "mujoco_sim_interface/viewer/MujocoViewer.h"
#include "mujoco_sim_interface/viewer/MujocoViewerRegistry.h"
#include "mujoco_sim_interface/viewer/ViewerNames.h"
#include "mujoco_sim_interface/viewer/ViewerSettings.h"
#include "mujoco_sim_interface/viewer/WebViewerOptions.h"
#include "robot_runtime/mujoco_sim_interface/test/ViewerTestSupport.h"
#include "web_viewer/ViewerThreadPriority.h"

/*
 * The viewer registry and the selection by name: the names, the lists
 * parseViewerNames() accepts and refuses, the ranges of the web viewer's
 * options, and what initSim() starts for a list - including the fallback of
 * `web,none` when the web viewer cannot start, and no thread at all for `none`.
 */

namespace robot::mujoco_sim_interface {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

TEST(ViewerNames, AreGlfwWebAndNone) {
  std::vector<std::string> names;
  for (const ViewerInfo& viewer : availableViewers()) {
    names.push_back(viewer.name);
    EXPECT_FALSE(viewer.description.empty()) << viewer.name;
  }
  EXPECT_THAT(names, ElementsAre("glfw", "web", "none"));
  EXPECT_EQ(availableViewerNames(), "glfw, web, none");
}

TEST(ParseViewerNames, SplitsAPreferenceList) {
  EXPECT_THAT(parseViewerNames("web,none"),
              IsOkAndHolds(ElementsAre("web", "none")));
  EXPECT_THAT(parseViewerNames(" web , none "),
              IsOkAndHolds(ElementsAre("web", "none")));
  EXPECT_THAT(parseViewerNames("glfw"), IsOkAndHolds(ElementsAre("glfw")));
  EXPECT_THAT(parseViewerNames("none"), IsOkAndHolds(ElementsAre("none")));
  EXPECT_THAT(parseViewerNames("web"), IsOkAndHolds(ElementsAre("web")));
}

TEST(ParseViewerNames, AnUnknownNameIsRefusedWithTheValidOnes) {
  EXPECT_THAT(parseViewerNames("vnc"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("unknown viewer 'vnc'"),
                             HasSubstr("glfw, web, none"))));
  EXPECT_THAT(
      parseViewerNames("web,Glfw"),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("'Glfw'")));
}

TEST(ParseViewerNames, EmptyListsAndEntriesAreRefused) {
  EXPECT_THAT(parseViewerNames(""), StatusIs(absl::StatusCode::kInvalidArgument,
                                             HasSubstr("glfw, web, none")));
  EXPECT_THAT(parseViewerNames("  "),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      parseViewerNames("web,,none"),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("empty entry")));
  EXPECT_THAT(
      parseViewerNames("web,"),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("empty entry")));
}

TEST(ParseViewerNames, ARepeatedNameIsRefused) {
  EXPECT_THAT(
      parseViewerNames("web,web"),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("listed twice")));
}

TEST(ParseViewerNames, NothingMayFollowAViewerThatAlwaysStarts) {
  EXPECT_THAT(parseViewerNames("none,web"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("'none' always starts")));
  // A GLFW window that cannot open is logged and the simulation runs on, so a
  // later entry could never be used.
  EXPECT_THAT(parseViewerNames("glfw,none"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("cannot fail at start-up")));
  EXPECT_THAT(parseViewerNames("glfw,web"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("cannot fail at start-up")));
  EXPECT_THAT(parseViewerNames("web,glfw"), IsOk());
  EXPECT_THAT(parseViewerNames("web,glfw,none"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("cannot fail at start-up")));
}

TEST(ValidateViewerSettings, TheDefaultsAreValid) {
  EXPECT_THAT(validateViewerSettings(ViewerSettings{}), IsOk());
  EXPECT_THAT(ViewerSettings{}.names, ElementsAre("glfw"))
      << "the default is the viewer of old";
}

TEST(ValidateViewerSettings, ChecksTheNames) {
  ViewerSettings settings;
  settings.names = {};
  EXPECT_THAT(validateViewerSettings(settings),
              StatusIs(absl::StatusCode::kInvalidArgument));
  settings.names = {"none", "glfw"};
  EXPECT_THAT(validateViewerSettings(settings),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("'none' always starts")));
}

TEST(ValidateViewerSettings, ChecksEveryWebViewerOptionAgainstItsRange) {
  struct Case {
    std::string field;
    WebViewerOptions options;
  };
  std::vector<Case> cases;
  const std::vector<std::string> fields = {
      "port",          "maxClients", "width",       "height",
      "maxWidth",      "maxHeight",  "fps",         "quality",
      "msaaSamples",   "shadowSize", "cores",       "threadPriority",
      "EGL device id", "address",    "EGL library", "start timeout",
      "allowed hosts"};
  for (const std::string& field : fields)
    cases.push_back({.field = field, .options = WebViewerOptions{}});
  cases[0].options.port = 70000;
  cases[1].options.maxClients = 0;
  cases[2].options.width = 1921;  // past maxWidth
  cases[3].options.height = 100;
  cases[4].options.maxWidth = 4000;
  cases[5].options.maxHeight = 2200;
  cases[6].options.fps = 0;
  cases[7].options.quality = 100;
  cases[8].options.msaaSamples = 17;
  cases[9].options.shadowSize = -1;
  cases[10].options.cores = {2, -1};
  cases[11].options.threadPriority =
      static_cast<web_viewer::ViewerThreadPriority>(7);
  cases[12].options.eglDeviceId = -1;
  cases[13].options.address = "";
  cases[14].options.eglLibrary = "";
  cases[15].options.startTimeout = absl::ZeroDuration();
  cases[16].options.allowedHosts = {"robot.local", ""};
  for (const Case& bad : cases) {
    ViewerSettings settings;
    settings.names = {kWebViewerName, kNoViewerName};
    settings.web = bad.options;
    EXPECT_THAT(validateViewerSettings(settings),
                StatusIs(absl::StatusCode::kInvalidArgument))
        << bad.field;
  }
  ViewerSettings ephemeral;
  ephemeral.web.port = 0;
  EXPECT_THAT(validateViewerSettings(ephemeral), IsOk())
      << "port 0 binds an ephemeral port";
}

TEST(MujocoViewerRegistry, AnUnknownNameIsAnInvalidArgument) {
  absl::StatusOr<std::unique_ptr<MujocoSimInterface>> sim =
      MujocoSimInterface::Create(hingeSimConfig({kNoViewerName}),
                                 hingeUrdfPath());
  ASSERT_TRUE(sim.ok()) << sim.status();
  EXPECT_THAT(createViewer("vnc", sim->get()).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("glfw, web, none")));
  absl::StatusOr<std::unique_ptr<MujocoViewer>> none =
      createViewer(kNoViewerName, sim->get());
  ASSERT_TRUE(none.ok()) << none.status();
  EXPECT_EQ((*none)->name(), kNoViewerName);
  EXPECT_THAT((*none)->start(), IsOk());
  EXPECT_TRUE((*none)->ok());
  EXPECT_EQ((*none)->port(), std::nullopt);
}

TEST(MujocoSimInterfaceViewer, CreateRefusesInvalidViewerSettings) {
  MujocoSimConfig config = hingeSimConfig({"vnc"});
  EXPECT_THAT(MujocoSimInterface::Create(config, hingeUrdfPath()).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("unknown viewer 'vnc'")));
  config = hingeSimConfig({kWebViewerName});
  config.viewer.web.fps = 500;
  EXPECT_THAT(MujocoSimInterface::Create(config, hingeUrdfPath()).status(),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("fps")));
}

TEST(MujocoSimInterfaceViewer, NoneStartsNoThread) {
  absl::StatusOr<std::unique_ptr<MujocoSimInterface>> sim =
      MujocoSimInterface::Create(hingeSimConfig({kNoViewerName}),
                                 hingeUrdfPath());
  ASSERT_TRUE(sim.ok()) << sim.status();
  EXPECT_EQ((*sim)->activeViewerName(), "")
      << "nothing started before initSim()";
  EXPECT_THAT((*sim)->viewerStatus(), IsOk());
  const std::vector<int> before = processThreadIds();
  (*sim)->initSim();
  EXPECT_EQ(processThreadIds(), before)
      << "initSim() started a thread for the 'none' viewer";
  EXPECT_THAT((*sim)->viewerStatus(), IsOk());
  EXPECT_EQ((*sim)->activeViewerName(), kNoViewerName);
  EXPECT_EQ((*sim)->webViewerPort(), std::nullopt);
}

/** A TCP port of 127.0.0.1 that nothing listens on now: bound once by the
 * kernel's choice, then closed. */
int freeLoopbackPort() {
  const int socket = ::socket(AF_INET, SOCK_STREAM, /*protocol=*/0);
  sockaddr_in address = {};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  socklen_t length = sizeof(address);
  if (bind(socket, reinterpret_cast<sockaddr*>(&address), sizeof(address)) !=
          0 ||
      getsockname(socket, reinterpret_cast<sockaddr*>(&address), &length) !=
          0) {
    close(socket);
    return -1;
  }
  close(socket);
  return ntohs(address.sin_port);
}

/** True when a listening socket can be bound to 127.0.0.1:`port` (with
 * SO_REUSEADDR, as the viewer's server binds). */
bool canBind(int port) {
  const int socket = ::socket(AF_INET, SOCK_STREAM, /*protocol=*/0);
  const int reuse = 1;
  setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
  sockaddr_in address = {};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = htons(static_cast<uint16_t>(port));
  const bool bound = bind(socket, reinterpret_cast<sockaddr*>(&address),
                          sizeof(address)) == 0 &&
                     listen(socket, /*n=*/1) == 0;
  close(socket);
  return bound;
}

TEST(MujocoSimInterfaceViewer, WebWithoutEglRefusesAndStartsNoThread) {
  MujocoSimConfig config = hingeSimConfig({kWebViewerName});
  config.viewer.web.port = 0;
  config.viewer.web.eglLibrary = "libEGL_missing.so.1";
  absl::StatusOr<std::unique_ptr<MujocoSimInterface>> sim =
      MujocoSimInterface::Create(config, hingeUrdfPath());
  ASSERT_TRUE(sim.ok()) << sim.status();
  const std::vector<int> before = processThreadIds();
  (*sim)->initSim();
  EXPECT_THAT(
      (*sim)->viewerStatus(),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               AllOf(HasSubstr("libEGL_missing.so.1"), HasSubstr("libegl1"),
                     HasSubstr("list none after it"))));
  EXPECT_EQ((*sim)->activeViewerName(), "");
  EXPECT_EQ(processThreadIds(), before)
      << "a viewer that did not start left a thread behind";
  // The simulation runs whether or not a viewer started.
  (*sim)->simulationStep();
}

TEST(MujocoSimInterfaceViewer, WebThenNoneFallsBackAndFreesThePort) {
  const int port = freeLoopbackPort();
  ASSERT_GT(port, 0);
  MujocoSimConfig config = hingeSimConfig({kWebViewerName, kNoViewerName});
  config.viewer.web.port = port;
  // The server binds in createViewer(), and only then does start() fail: its
  // render thread cannot set up in 1 ns.
  config.viewer.web.startTimeout = absl::Nanoseconds(1);
  absl::StatusOr<std::unique_ptr<MujocoSimInterface>> sim =
      MujocoSimInterface::Create(config, hingeUrdfPath());
  ASSERT_TRUE(sim.ok()) << sim.status();
  (*sim)->initSim();
  EXPECT_THAT((*sim)->viewerStatus(), IsOk());
  EXPECT_EQ((*sim)->activeViewerName(), kNoViewerName);
  EXPECT_EQ((*sim)->webViewerPort(), std::nullopt);
  EXPECT_TRUE(canBind(port))
      << "the web viewer that did not start still holds port " << port;
}

}  // namespace
}  // namespace robot::mujoco_sim_interface
