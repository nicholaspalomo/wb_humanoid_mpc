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

#include "humanoid_common_mpc_app/node/ShutdownSignal.h"

#include <signal.h>

#include <atomic>
#include <csignal>

#include "absl/time/clock.h"

namespace ocs2::humanoid::node {
namespace {

// Lock-free, so that the handler may write it (a lock-free atomic is async-signal-safe).
std::atomic<bool> shutdownRequestedFlag{false};
static_assert(std::atomic<bool>::is_always_lock_free, "the signal handler needs a lock-free flag");

void onShutdownSignal(int signalNumber) {
  if (shutdownRequestedFlag.exchange(true)) {
    // The second signal: the default action (termination) at once. signal() and raise() are async-signal-safe.
    std::signal(signalNumber, SIG_DFL);
    std::raise(signalNumber);
  }
}

void installHandler(int signalNumber) {
  struct sigaction action {};
  action.sa_handler = &onShutdownSignal;
  sigemptyset(&action.sa_mask);
  // No SA_RESTART: a blocking read in the main thread returns EINTR, so that it can look at the flag.
  action.sa_flags = 0;
  sigaction(signalNumber, &action, /*oact=*/nullptr);
}

}  // namespace

void installShutdownSignalHandlers() {
  installHandler(SIGINT);
  installHandler(SIGTERM);
}

bool shutdownRequested() {
  return shutdownRequestedFlag.load();
}

void waitForShutdown(absl::Duration pollPeriod) {
  while (!shutdownRequested()) {
    absl::SleepFor(pollPeriod);
  }
}

}  // namespace ocs2::humanoid::node
