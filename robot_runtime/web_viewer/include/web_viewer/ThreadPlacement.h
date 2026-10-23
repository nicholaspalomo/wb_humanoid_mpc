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

#include "absl/status/status.h"
#include "absl/types/span.h"

#include "web_viewer/ViewerThreadPriority.h"

namespace robot::web_viewer {

/**
 * Places the calling thread for viewer work: pins it to `cores` (CPU indices; empty leaves the affinity as it is) and,
 * for ViewerThreadPriority::kIdle, switches it to SCHED_IDLE, so that a physics or MPC thread it shares a core with
 * takes the core whenever it wakes. Threads it creates afterwards inherit both, which is why the viewer's threads call
 * it first, before they start libraries that spawn workers (llvmpipe, the HTTP server's pool).
 *
 * InvalidArgument, with nothing applied, for a core index out of [0, CPU_SETSIZE) or a `priority` that is none of
 * ViewerThreadPriority's. Otherwise it applies what it can: when one of the two system calls fails (a core outside the
 * process's allowed set, for one), the other is still applied, and the Status carries the error of
 * pthread_setaffinity_np or pthread_setschedparam. The viewer logs it as a warning and runs on.
 */
absl::Status placeCurrentThread(absl::Span<const int> cores, ViewerThreadPriority priority);

}  // namespace robot::web_viewer
