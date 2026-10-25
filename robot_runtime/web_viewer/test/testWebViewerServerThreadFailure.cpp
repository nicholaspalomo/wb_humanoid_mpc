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

#include <atomic>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "gtest/gtest.h"

#include "robot_runtime/web_viewer/test/PthreadCreateInterposer.h"
#include "robot_runtime/web_viewer/test/WebViewerTestSupport.h"
#include "web_viewer/ViewerFrameMailbox.h"
#include "web_viewer/ViewerInputQueue.h"
#include "web_viewer/WebViewerServer.h"

/*
 * WebViewerServer::Create() when a request thread cannot start (EAGAIN from
 * pthread_create, as under a pids cgroup or RLIMIT_NPROC near its limit): it
 * must fail with Internal rather than return a server that never accepts. A
 * binary of its own, since it interposes pthread_create for the whole process
 * (PthreadCreateInterposer.cpp).
 */

namespace robot::web_viewer {
namespace {

// How many more pthread_create calls succeed before one fails with EAGAIN;
// negative: none fails.
std::atomic<int> callsBeforeFailure = -1;

}  // namespace

bool takePthreadCreateFailure() {
  return callsBeforeFailure.load() >= 0 && callsBeforeFailure.fetch_sub(1) == 0;
}

namespace {

TEST(WebViewerServerThreadFailureTest,
     CreateFailsWhenARequestThreadCannotStart) {
  ViewerFrameMailbox mailbox(/*maxClients=*/4);
  ViewerInputQueue inputs;
  // The listen thread and the first request thread start; the second request
  // thread does not.
  callsBeforeFailure.store(2);
  absl::StatusOr<std::unique_ptr<WebViewerServer>> server =
      WebViewerServer::Create(test_support::testServerOptions(), &mailbox,
                              &inputs);
  callsBeforeFailure.store(-1);
  EXPECT_EQ(server.status().code(), absl::StatusCode::kInternal)
      << server.status();
  EXPECT_TRUE(absl::StrContains(server.status().message(),
                                "could not start its request threads"))
      << server.status();

  // With its threads, a server starts.
  absl::StatusOr<std::unique_ptr<WebViewerServer>> again =
      WebViewerServer::Create(test_support::testServerOptions(), &mailbox,
                              &inputs);
  EXPECT_TRUE(again.ok()) << again.status();
}

}  // namespace
}  // namespace robot::web_viewer
