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

// Setting up the calling thread to run a realtime loop: SCHED_FIFO scheduling, locked memory, a prefaulted stack, CPU
// pinning and a name. Linux only. Each call is made once, before the loop starts; none of them belongs in the loop.
//
// The privileged calls return an error instead of failing silently or aborting, so that the robot process can tell the
// operator what is missing and decide whether to carry on without it (in simulation, say):
//   - SCHED_FIFO needs CAP_SYS_NICE or an RLIMIT_RTPRIO at least as high as the priority (`ulimit -r`,
//     /etc/security/limits.conf, `docker run --cap-add=SYS_NICE --ulimit rtprio=99`);
//   - mlockall needs CAP_IPC_LOCK or an RLIMIT_MEMLOCK larger than the process (`ulimit -l unlimited`,
//     `docker run --ulimit memlock=-1`).

#include <cstddef>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace robot::realtime {

/// The longest thread name Linux keeps: TASK_COMM_LEN is 16 bytes, including the terminating NUL.
inline constexpr std::size_t kMaxThreadNameLength = 15;

/// How much stack configureCurrentThread() prefaults for a realtime thread by default.
inline constexpr std::size_t kDefaultStackPrefaultBytes = 256 * 1024;

/**
 * Runs the calling thread under SCHED_FIFO at `priority`.
 *
 * @return InvalidArgument when `priority` is outside SCHED_FIFO's range (1 to 99 on Linux), PermissionDenied when the
 *         process may not use it (the message says how to grant it); the thread's scheduling is then unchanged.
 */
absl::Status setCurrentThreadRealtime(int priority);

/// Returns the calling thread to the default time-sharing scheduler (SCHED_OTHER), e.g. once a realtime loop ends.
absl::Status setCurrentThreadNonRealtime();

/**
 * Locks every page the process has mapped, and every page it maps from now on, into RAM (mlockall with MCL_CURRENT and
 * MCL_FUTURE), so that the realtime thread never waits for a page to be read back from swap or a file.
 *
 * Process-wide. Call it before prefaultStack(), so that the pages the prefault touches stay locked.
 *
 * @return PermissionDenied without CAP_IPC_LOCK and with an RLIMIT_MEMLOCK of 0, ResourceExhausted when RLIMIT_MEMLOCK
 *         is smaller than what the process maps; nothing is locked then.
 */
absl::Status lockProcessMemory();

/**
 * Touches `bytes` of the calling thread's stack below the current frame, so that the realtime loop does not take a page
 * fault the first time it reaches a deeper call. Once memory is locked the pages stay resident.
 *
 * @return OutOfRange when the thread's stack has less than `bytes` (plus a safety margin) left below the caller;
 *         nothing is touched then.
 */
absl::Status prefaultStack(std::size_t bytes);

/**
 * Pins the calling thread to `cores` (0-based CPU numbers).
 *
 * @return InvalidArgument when `cores` is empty or names a CPU that does not exist or that the process may not use (a
 *         container's cpuset), FailedPrecondition when the kernel left out some of the cores; the message names them.
 */
absl::Status setCurrentThreadAffinity(absl::Span<const int> cores);

/**
 * Names the calling thread, as shown by `top -H`, `ps -L`, gdb and perf.
 *
 * @return InvalidArgument when `name` is empty, longer than kMaxThreadNameLength or contains a NUL.
 */
absl::Status setCurrentThreadName(absl::string_view name);

/// What configureCurrentThread() does to the memory of the process when it makes the thread realtime.
enum class MemoryLock {
  /// lockProcessMemory(), then prefaultStack(): no page of the process is ever read back from swap or a file, which a
  /// realtime loop that must not miss a deadline needs. Process-wide: the memory of every other thread is locked too,
  /// and MCL_FUTURE makes every later mmap() or brk() of the process fail once its locked memory would exceed a finite
  /// RLIMIT_MEMLOCK (CAP_IPC_LOCK lifts the limit).
  kLockProcess,
  /// Neither: the thread only gets its scheduling. For a thread of a process that allocates freely and is not hard
  /// realtime, such as the MPC solver, where a failed allocation would be worse than a page fault.
  kNone,
};

/// Everything configureCurrentThread() sets up. A default-constructed configuration changes nothing.
struct RealtimeThreadConfig {
  /// Thread name, at most kMaxThreadNameLength characters; empty keeps the current name.
  std::string name;
  /// SCHED_FIFO priority, 1 to 99; 0 keeps the thread on the time-sharing scheduler and its memory unlocked.
  int priority = 0;
  /// CPU cores to pin the thread to; empty leaves its affinity alone.
  std::vector<int> cores;
  /// Stack to prefault when `priority` > 0 and `memoryLock` is kLockProcess.
  std::size_t stackPrefaultBytes = kDefaultStackPrefaultBytes;
  /// What happens to the memory of the process when `priority` > 0.
  MemoryLock memoryLock = MemoryLock::kLockProcess;
};

/**
 * Sets up the calling thread as `config` asks, in this order: name, CPU affinity and, when `config.priority` > 0,
 * lockProcessMemory() and prefaultStack() (unless `config.memoryLock` is kNone), then setCurrentThreadRealtime().
 *
 * A step that fails does not stop the ones after it: a thread whose name is refused, that the kernel pinned to only
 * some of its cores, or whose process may not lock its memory, still runs SCHED_FIFO when it may. A negative priority
 * is rejected before anything changes.
 *
 * @return OK when every step took effect. Otherwise the error of the first step that failed, with a message that names
 *         every step that failed and why ("pinning the thread: ...; locking memory: ..."); the others took effect.
 */
absl::Status configureCurrentThread(const RealtimeThreadConfig& config);

}  // namespace robot::realtime
