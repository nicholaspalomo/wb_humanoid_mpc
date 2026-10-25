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

#include "web_viewer/ThreadPlacement.h"

#include <pthread.h>
#include <sched.h>

#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"

#include "web_viewer/ViewerThreadPriority.h"

namespace robot::web_viewer {
namespace {

absl::Status checkPlacement(absl::Span<const int> cores,
                            ViewerThreadPriority priority) {
  for (const int core : cores) {
    if (core < 0 || core >= CPU_SETSIZE) {
      return absl::InvalidArgumentError(absl::StrCat(
          "core ", core, " is not a CPU index (0 to ", CPU_SETSIZE - 1, ")"));
    }
  }
  switch (priority) {
    case ViewerThreadPriority::kIdle:
    case ViewerThreadPriority::kInherited:
      return absl::OkStatus();
  }
  return absl::InvalidArgumentError(
      absl::StrCat("viewer thread priority ", static_cast<int>(priority),
                   " is none of ViewerThreadPriority's (kIdle, kInherited)"));
}

// Pins the calling thread to `cores`.
absl::Status pinToCores(absl::Span<const int> cores) {
  cpu_set_t set;
  CPU_ZERO(&set);
  for (const int core : cores) CPU_SET(core, &set);
  const int error = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  if (error == 0) return absl::OkStatus();
  return absl::ErrnoToStatus(error,
                             absl::StrCat("pthread_setaffinity_np(cores ",
                                          absl::StrJoin(cores, ","), ")"));
}

// Switches the calling thread to SCHED_IDLE, whose only priority is 0. Lowering
// a thread's policy needs no privilege.
absl::Status idleScheduling() {
  const sched_param parameters = {.sched_priority = 0};
  const int error =
      pthread_setschedparam(pthread_self(), SCHED_IDLE, &parameters);
  if (error == 0) return absl::OkStatus();
  return absl::ErrnoToStatus(error,
                             "pthread_setschedparam(this thread, SCHED_IDLE)");
}

}  // namespace

absl::Status placeCurrentThread(absl::Span<const int> cores,
                                ViewerThreadPriority priority) {
  absl::Status valid = checkPlacement(cores, priority);
  if (!valid.ok()) return valid;
  absl::Status pinned = cores.empty() ? absl::OkStatus() : pinToCores(cores);
  absl::Status idled = priority == ViewerThreadPriority::kIdle
                           ? idleScheduling()
                           : absl::OkStatus();
  if (!pinned.ok() && !idled.ok())
    return absl::Status(pinned.code(),
                        absl::StrCat(pinned.message(), "; ", idled.message()));
  if (!pinned.ok()) return pinned;
  return idled;
}

}  // namespace robot::web_viewer
