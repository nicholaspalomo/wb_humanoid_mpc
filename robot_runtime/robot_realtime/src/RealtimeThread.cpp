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

#include "robot_realtime/RealtimeThread.h"

#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace robot::realtime {
namespace {

// The prefault touches the stack in frames of this size, one recursive call per frame: a fixed-size array per frame
// instead of alloca() or a variable-length array, which the Google style guide does not allow.
constexpr size_t kStackPrefaultChunkBytes = 16 * 1024;
// What a frame of touchStackChunks() adds to its array (return address, saved registers, alignment), generously.
constexpr size_t kStackPrefaultFrameOverheadBytes = 256;
// Stack the prefault leaves untouched at the bottom: room for the calls the thread makes from its deepest frame, for a
// signal handler, and for the inaccuracy of the stack bounds glibc reports for the main thread.
constexpr size_t kStackSafetyMarginBytes = 64 * 1024;

// The soft limit of `resource`, as text for an error message.
std::string softLimitText(int resource) {
  rlimit limit{};
  if (getrlimit(resource, &limit) != 0) {
    return "unknown";
  }
  if (limit.rlim_cur == RLIM_INFINITY) {
    return "unlimited";
  }
  return absl::StrCat(limit.rlim_cur);
}

// The failed steps of configureCurrentThread(): the code of the first, and the step and message of every one, so that
// its error says everything it could not do.
class StepFailures {
 public:
  void record(absl::string_view step, const absl::Status& status) {
    if (status.ok()) {
      return;
    }
    if (messages_.empty()) {
      code_ = status.code();
    }
    messages_.push_back(absl::StrCat(step, ": ", status.message()));
  }

  absl::Status status() const {
    if (messages_.empty()) {
      return absl::OkStatus();
    }
    return absl::Status(code_, absl::StrJoin(messages_, "; "));
  }

 private:
  absl::StatusCode code_ = absl::StatusCode::kOk;
  std::vector<std::string> messages_;
};

// Touches `chunks` frames of kStackPrefaultChunkBytes, one page at a time, by calling itself.
[[gnu::noinline]] void touchStackChunks(size_t chunks, size_t pageSize) {
  volatile unsigned char chunk[kStackPrefaultChunkBytes];
  for (size_t offset = 0; offset < kStackPrefaultChunkBytes; offset += pageSize) {
    chunk[offset] = 0;
  }
  chunk[kStackPrefaultChunkBytes - 1] = 0;
  if (chunks > 1) {
    touchStackChunks(chunks - 1, pageSize);
  }
  // Reading the chunk after the call keeps the compiler from turning the call into a jump that reuses this frame.
  static_cast<void>(chunk[0]);
}

}  // namespace

absl::Status setCurrentThreadRealtime(int priority) {
  const int minPriority = sched_get_priority_min(SCHED_FIFO);
  const int maxPriority = sched_get_priority_max(SCHED_FIFO);
  if (priority < minPriority || priority > maxPriority) {
    return absl::InvalidArgumentError(
        absl::StrCat("SCHED_FIFO priority ", priority, " is outside the valid range [", minPriority, ", ", maxPriority, "]"));
  }
  sched_param parameters{};
  parameters.sched_priority = priority;
  const int error = pthread_setschedparam(pthread_self(), SCHED_FIFO, &parameters);
  if (error == EPERM) {
    return absl::PermissionDeniedError(absl::StrCat(
        "not permitted to run this thread under SCHED_FIFO at priority ", priority,
        ": the process needs CAP_SYS_NICE or an RLIMIT_RTPRIO of at least ", priority, " (it is ", softLimitText(RLIMIT_RTPRIO),
        "). Raise it with `ulimit -r`, /etc/security/limits.conf or `docker run --cap-add=SYS_NICE --ulimit rtprio=99`; "
        "in a cgroup with realtime group scheduling, cpu.rt_runtime_us must be nonzero as well"));
  }
  if (error != 0) {
    return absl::ErrnoToStatus(error, absl::StrCat("pthread_setschedparam(SCHED_FIFO, ", priority, ") failed"));
  }
  return absl::OkStatus();
}

absl::Status setCurrentThreadNonRealtime() {
  sched_param parameters{};
  parameters.sched_priority = 0;
  const int error = pthread_setschedparam(pthread_self(), SCHED_OTHER, &parameters);
  if (error != 0) {
    return absl::ErrnoToStatus(error, "pthread_setschedparam(SCHED_OTHER) failed");
  }
  return absl::OkStatus();
}

absl::Status lockProcessMemory() {
  if (mlockall(MCL_CURRENT | MCL_FUTURE) == 0) {
    return absl::OkStatus();
  }
  const int error = errno;
  if (error == EPERM) {
    return absl::PermissionDeniedError(
        absl::StrCat("not permitted to lock the process's memory: the process needs CAP_IPC_LOCK or a nonzero RLIMIT_MEMLOCK (it is ",
                     softLimitText(RLIMIT_MEMLOCK), " bytes). Raise it with `ulimit -l unlimited`, /etc/security/limits.conf or ",
                     "`docker run --ulimit memlock=-1`"));
  }
  if (error == ENOMEM) {
    return absl::ResourceExhaustedError(absl::StrCat("cannot lock the process's memory: it maps more than its RLIMIT_MEMLOCK of ",
                                                     softLimitText(RLIMIT_MEMLOCK),
                                                     " bytes allows. Raise it with `ulimit -l unlimited`, "
                                                     "/etc/security/limits.conf or `docker run --ulimit memlock=-1`"));
  }
  return absl::ErrnoToStatus(error, "mlockall(MCL_CURRENT | MCL_FUTURE) failed");
}

absl::Status prefaultStack(size_t bytes) {
  if (bytes == 0) {
    return absl::OkStatus();
  }
  pthread_attr_t attributes;
  int error = pthread_getattr_np(pthread_self(), &attributes);
  if (error != 0) {
    return absl::ErrnoToStatus(error, "pthread_getattr_np failed");
  }
  void* absl_nullable stackLowest = nullptr;
  size_t stackSize = 0;
  error = pthread_attr_getstack(&attributes, &stackLowest, &stackSize);
  pthread_attr_destroy(&attributes);
  if (error != 0) {
    return absl::ErrnoToStatus(error, "pthread_attr_getstack failed");
  }

  // The stack grows down, from just above this frame towards stackLowest.
  const unsigned char marker = 0;
  const uintptr_t current = reinterpret_cast<uintptr_t>(&marker);
  const uintptr_t lowest = reinterpret_cast<uintptr_t>(stackLowest);
  const size_t available = current > lowest ? static_cast<size_t>(current - lowest) : 0;
  const size_t chunks = (bytes + kStackPrefaultChunkBytes - 1) / kStackPrefaultChunkBytes;
  const size_t needed = chunks * (kStackPrefaultChunkBytes + kStackPrefaultFrameOverheadBytes) + kStackSafetyMarginBytes;
  if (needed > available) {
    return absl::OutOfRangeError(absl::StrCat("cannot prefault ", bytes, " bytes of stack: the thread's stack of ", stackSize,
                                              " bytes has ", available, " bytes left below the caller, and ", kStackSafetyMarginBytes,
                                              " of them stay untouched as a margin"));
  }

  const int64_t pageSize = sysconf(_SC_PAGESIZE);
  touchStackChunks(chunks, pageSize > 0 ? static_cast<size_t>(pageSize) : 4096);
  return absl::OkStatus();
}

absl::Status setCurrentThreadAffinity(absl::Span<const int> cores) {
  if (cores.empty()) {
    return absl::InvalidArgumentError("no CPU cores given to pin the thread to");
  }
  const int64_t configuredCpus = sysconf(_SC_NPROCESSORS_CONF);
  const int cpuCount = configuredCpus > 0 ? static_cast<int>(std::min<int64_t>(configuredCpus, CPU_SETSIZE)) : CPU_SETSIZE;

  cpu_set_t requested;
  CPU_ZERO(&requested);
  for (const int core : cores) {
    if (core < 0 || core >= cpuCount) {
      return absl::InvalidArgumentError(absl::StrCat("CPU core ", core, " does not exist: this machine has cores 0 to ", cpuCount - 1));
    }
    CPU_SET(core, &requested);
  }

  int error = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &requested);
  if (error == EINVAL) {
    return absl::InvalidArgumentError(
        absl::StrCat("none of the CPU cores [", absl::StrJoin(cores, ", "), "] is available to this process (its cpuset)"));
  }
  if (error != 0) {
    return absl::ErrnoToStatus(error, "pthread_setaffinity_np failed");
  }

  // The kernel intersects the request with the cores the process may use and only fails when nothing is left.
  cpu_set_t applied;
  CPU_ZERO(&applied);
  error = pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &applied);
  if (error != 0) {
    return absl::ErrnoToStatus(error, "pthread_getaffinity_np failed");
  }
  if (!CPU_EQUAL(&requested, &applied)) {
    std::vector<int> leftOut;
    for (const int core : cores) {
      if (!CPU_ISSET(core, &applied)) {
        leftOut.push_back(core);
      }
    }
    return absl::FailedPreconditionError(absl::StrCat("the kernel left out CPU cores [", absl::StrJoin(leftOut, ", "),
                                                      "], which this process may not use (its cpuset); the thread runs on the others"));
  }
  return absl::OkStatus();
}

absl::Status setCurrentThreadName(absl::string_view name) {
  if (name.empty() || name.size() > kMaxThreadNameLength) {
    return absl::InvalidArgumentError(
        absl::StrCat("thread name \"", name, "\" has ", name.size(), " characters; Linux keeps 1 to ", kMaxThreadNameLength));
  }
  if (absl::StrContains(name, '\0')) {
    return absl::InvalidArgumentError("a thread name cannot contain a NUL character");
  }
  char terminated[kMaxThreadNameLength + 1] = {};
  name.copy(terminated, name.size());
  const int error = pthread_setname_np(pthread_self(), terminated);
  if (error != 0) {
    return absl::ErrnoToStatus(error, "pthread_setname_np failed");
  }
  return absl::OkStatus();
}

absl::Status configureCurrentThread(const RealtimeThreadConfig& config) {
  if (config.priority < 0) {
    return absl::InvalidArgumentError(absl::StrCat("realtime priority ", config.priority, " is negative; 0 means not realtime"));
  }
  StepFailures failures;
  if (!config.name.empty()) {
    failures.record("naming the thread", setCurrentThreadName(config.name));
  }
  if (!config.cores.empty()) {
    failures.record("pinning the thread", setCurrentThreadAffinity(config.cores));
  }
  if (config.priority > 0) {
    if (config.memoryLock == MemoryLock::kLockProcess) {
      failures.record("locking memory", lockProcessMemory());
      failures.record("prefaulting the stack", prefaultStack(config.stackPrefaultBytes));
    }
    failures.record("setting the realtime priority", setCurrentThreadRealtime(config.priority));
  }
  return failures.status();
}

}  // namespace robot::realtime
