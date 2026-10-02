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

#pragma once

#include "absl/time/time.h"

namespace ocs2::humanoid::node {

/**
 * How the laptop-side binaries (the MPC nodes, the dummy simulators, the keyboard teleoperation) end: SIGINT (Ctrl-C)
 * and SIGTERM (the launcher's stop, `docker stop`) ask them to shut down cleanly, and a second one ends the process at
 * once with the signal's default action, so that a shutdown that hangs can still be cut short.
 */

/** Installs the handlers of SIGINT and SIGTERM. Idempotent; call it before starting any thread. */
void installShutdownSignalHandlers();

/** True once SIGINT or SIGTERM has arrived since installShutdownSignalHandlers() (or requestShutdown()). */
bool shutdownRequested();

/** Asks for the shutdown as a signal would, e.g. from a thread that has failed. */
void requestShutdown();

/** Returns once shutdownRequested(), checking every `pollPeriod`. */
void waitForShutdown(absl::Duration pollPeriod = absl::Milliseconds(50));

}  // namespace ocs2::humanoid::node
