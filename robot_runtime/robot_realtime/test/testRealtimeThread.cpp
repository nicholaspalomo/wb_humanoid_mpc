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

#include <gtest/gtest.h>

#include <linux/capability.h>
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/match.h"

#include "robot_realtime/RealtimeThread.h"

namespace robot::realtime {
namespace {

// The calling thread's scheduling policy (SCHED_OTHER, SCHED_FIFO, ...).
int currentPolicy() {
  int policy = -1;
  sched_param parameters{};
  pthread_getschedparam(pthread_self(), &policy, &parameters);
  return policy;
}

std::string currentThreadName() {
  char name[kMaxThreadNameLength + 1] = {};
  pthread_getname_np(pthread_self(), name, sizeof(name));
  return std::string(name);
}

// The cores the calling thread may run on.
std::vector<int> allowedCores() {
  cpu_set_t set;
  CPU_ZERO(&set);
  std::vector<int> cores;
  if (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &set) != 0) {
    return cores;
  }
  for (int core = 0; core < CPU_SETSIZE; ++core) {
    if (CPU_ISSET(core, &set)) {
      cores.push_back(core);
    }
  }
  return cores;
}

long minorPageFaultsOfThisThread() {
  rusage usage{};
  getrusage(RUSAGE_THREAD, &usage);
  return usage.ru_minflt;
}

constexpr std::size_t kTestChunkBytes = 16 * 1024;

// Uses `chunks` frames of kTestChunkBytes of stack, as a deep call chain of the realtime loop would.
[[gnu::noinline]] void useStack(std::size_t chunks) {
  volatile unsigned char chunk[kTestChunkBytes];
  for (std::size_t offset = 0; offset < kTestChunkBytes; offset += 1024) {
    chunk[offset] = 1;
  }
  if (chunks > 1) {
    useStack(chunks - 1);
  }
  static_cast<void>(chunk[0]);
}

// ---------------------------------------------------------------------------------------------------------------------
// Name and affinity
// ---------------------------------------------------------------------------------------------------------------------

TEST(RealtimeThreadTest, namesTheThread) {
  absl::Status shortName;
  std::string shortNameSeen;
  absl::Status longestName;
  std::string longestNameSeen;
  std::thread thread([&]() {
    shortName = setCurrentThreadName("rt_loop");
    shortNameSeen = currentThreadName();
    longestName = setCurrentThreadName("fifteen_chars__");
    longestNameSeen = currentThreadName();
  });
  thread.join();
  EXPECT_TRUE(shortName.ok()) << shortName;
  EXPECT_EQ(shortNameSeen, "rt_loop");
  EXPECT_TRUE(longestName.ok()) << longestName;
  EXPECT_EQ(longestNameSeen, "fifteen_chars__");
}

TEST(RealtimeThreadTest, rejectsNamesLinuxCannotKeep) {
  EXPECT_EQ(setCurrentThreadName("sixteen_chars___").code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(setCurrentThreadName("").code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(setCurrentThreadName(std::string("rt\0loop", 7)).code(), absl::StatusCode::kInvalidArgument);
}

TEST(RealtimeThreadTest, pinsTheThreadToAnAllowedCore) {
  absl::Status status;
  std::vector<int> before;
  std::vector<int> after;
  std::thread thread([&]() {
    before = allowedCores();
    if (before.empty()) {
      return;
    }
    status = setCurrentThreadAffinity(std::vector<int>{before.back()});
    after = allowedCores();
  });
  thread.join();
  ASSERT_FALSE(before.empty());
  EXPECT_TRUE(status.ok()) << status;
  EXPECT_EQ(after, std::vector<int>{before.back()});
}

TEST(RealtimeThreadTest, rejectsCoresThatDoNotExist) {
  std::vector<absl::Status> statuses;
  std::vector<int> unchanged;
  std::vector<int> original;
  std::thread thread([&]() {
    original = allowedCores();
    statuses.push_back(setCurrentThreadAffinity(std::vector<int>{}));
    statuses.push_back(setCurrentThreadAffinity(std::vector<int>{-1}));
    statuses.push_back(setCurrentThreadAffinity(std::vector<int>{CPU_SETSIZE}));
    statuses.push_back(setCurrentThreadAffinity(std::vector<int>{0, static_cast<int>(sysconf(_SC_NPROCESSORS_CONF))}));
    unchanged = allowedCores();
  });
  thread.join();
  for (const absl::Status& status : statuses) {
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
  }
  EXPECT_EQ(unchanged, original) << "a rejected request leaves the affinity alone";
}

// ---------------------------------------------------------------------------------------------------------------------
// Scheduling
// ---------------------------------------------------------------------------------------------------------------------

TEST(RealtimeThreadTest, rejectsPrioritiesOutsideTheSchedFifoRange) {
  for (const int priority : {-1, 0, 100}) {
    const absl::Status status = setCurrentThreadRealtime(priority);
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
  }
  EXPECT_EQ(currentPolicy(), SCHED_OTHER);
}

// Holds on any machine: with the privilege the thread runs SCHED_FIFO (and is put back), without it the call reports
// the missing permission and the thread keeps its scheduling.
TEST(RealtimeThreadTest, schedFifoEitherTakesEffectOrReportsTheMissingPermission) {
  absl::Status status;
  int policyAfterwards = -1;
  int priorityAfterwards = -1;
  absl::Status restored;
  int policyRestored = -1;
  std::thread thread([&]() {
    status = setCurrentThreadRealtime(/*priority=*/1);
    sched_param parameters{};
    pthread_getschedparam(pthread_self(), &policyAfterwards, &parameters);
    priorityAfterwards = parameters.sched_priority;
    restored = setCurrentThreadNonRealtime();
    policyRestored = currentPolicy();
  });
  thread.join();
  if (status.ok()) {
    EXPECT_EQ(policyAfterwards, SCHED_FIFO);
    EXPECT_EQ(priorityAfterwards, 1);
  } else {
    EXPECT_EQ(status.code(), absl::StatusCode::kPermissionDenied) << status;
    EXPECT_EQ(policyAfterwards, SCHED_OTHER);
  }
  EXPECT_TRUE(restored.ok()) << restored;
  EXPECT_EQ(policyRestored, SCHED_OTHER);
}

// Removes the capabilities that let a process ignore RLIMIT_RTPRIO and RLIMIT_MEMLOCK from the calling thread, so
// that the unprivileged path runs even where the tests themselves run with privileges (as root in CI, say).
bool dropRealtimeCapabilities() {
  __user_cap_header_struct header{};
  header.version = _LINUX_CAPABILITY_VERSION_3;
  header.pid = 0;
  __user_cap_data_struct data[_LINUX_CAPABILITY_U32S_3] = {};
  if (syscall(SYS_capget, &header, data) != 0) {
    return false;
  }
  for (const int capability : {CAP_SYS_NICE, CAP_IPC_LOCK}) {
    data[CAP_TO_INDEX(capability)].effective &= ~CAP_TO_MASK(capability);
    data[CAP_TO_INDEX(capability)].permitted &= ~CAP_TO_MASK(capability);
    data[CAP_TO_INDEX(capability)].inheritable &= ~CAP_TO_MASK(capability);
  }
  return syscall(SYS_capset, &header, data) == 0;
}

// Reports `status` and counts it as a failure unless it has `code` and mentions `hint`.
int expectFailure(const char* call, const absl::Status& status, absl::StatusCode code, const char* hint) {
  if (status.code() == code && absl::StrContains(status.message(), hint)) {
    return 0;
  }
  std::fprintf(stderr, "%s returned %s; expected %s mentioning \"%s\"\n", call, status.ToString().c_str(),
               absl::StatusCodeToString(code).c_str(), hint);
  return 1;
}

// Runs in a child process (the death-test machinery forks or re-executes the test binary), because it lowers the
// process's resource limits for good. Exits 0 when every call failed cleanly with the expected error.
[[noreturn]] void runSetupWithoutPrivileges(rlim_t memlockLimit) {
  if (!dropRealtimeCapabilities()) {
    std::fprintf(stderr, "capset failed: %s\n", std::strerror(errno));
    _exit(2);
  }
  const rlimit noRealtimePriority{.rlim_cur = 0, .rlim_max = 0};
  const rlimit memlock{.rlim_cur = memlockLimit, .rlim_max = memlockLimit};
  if (setrlimit(RLIMIT_RTPRIO, &noRealtimePriority) != 0 || setrlimit(RLIMIT_MEMLOCK, &memlock) != 0) {
    std::fprintf(stderr, "setrlimit failed: %s\n", std::strerror(errno));
    _exit(3);
  }

  int failures = 0;
  failures += expectFailure("setCurrentThreadRealtime", setCurrentThreadRealtime(/*priority=*/10), absl::StatusCode::kPermissionDenied,
                            "CAP_SYS_NICE");
  if (currentPolicy() != SCHED_OTHER) {
    std::fprintf(stderr, "the thread's policy changed although the call failed\n");
    ++failures;
  }
  if (memlockLimit == 0) {
    failures += expectFailure("lockProcessMemory", lockProcessMemory(), absl::StatusCode::kPermissionDenied, "CAP_IPC_LOCK");
    RealtimeThreadConfig config;
    config.name = "rt_test";
    config.priority = 10;
    const absl::Status locked = configureCurrentThread(config);
    failures += expectFailure("configureCurrentThread", locked, absl::StatusCode::kPermissionDenied, "locking memory");
    // The memory lock failing does not keep the thread from being made realtime: that step is tried and reported too.
    failures += expectFailure("configureCurrentThread", locked, absl::StatusCode::kPermissionDenied, "setting the realtime priority");
    if (currentThreadName() != "rt_test") {
      std::fprintf(stderr, "configureCurrentThread did not name the thread\n");
      ++failures;
    }

    // Without the memory lock only the scheduling is tried, so only the scheduling can fail.
    config.memoryLock = MemoryLock::kNone;
    const absl::Status unlocked = configureCurrentThread(config);
    failures +=
        expectFailure("configureCurrentThread(kNone)", unlocked, absl::StatusCode::kPermissionDenied, "setting the realtime priority");
    if (absl::StrContains(unlocked.message(), "locking memory") || absl::StrContains(unlocked.message(), "prefaulting")) {
      std::fprintf(stderr, "configureCurrentThread(kNone) touched the memory: %s\n", unlocked.ToString().c_str());
      ++failures;
    }
  } else {
    failures += expectFailure("lockProcessMemory", lockProcessMemory(), absl::StatusCode::kResourceExhausted, "RLIMIT_MEMLOCK");
  }
  _exit(failures == 0 ? 0 : 1);
}

TEST(RealtimeThreadDeathTest, withoutPrivilegesTheSetupReportsWhatIsMissing) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(runSetupWithoutPrivileges(/*memlockLimit=*/0), ::testing::ExitedWithCode(0), "");
}

TEST(RealtimeThreadDeathTest, aMemoryLockLimitTooSmallForTheProcessIsReported) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(runSetupWithoutPrivileges(/*memlockLimit=*/4096), ::testing::ExitedWithCode(0), "");
}

// ---------------------------------------------------------------------------------------------------------------------
// Stack
// ---------------------------------------------------------------------------------------------------------------------

TEST(RealtimeThreadTest, aPrefaultedStackTakesNoPageFaultsWhenTheLoopReachesIt) {
  absl::Status status;
  long faults = -1;
  std::thread thread([&]() {
    status = prefaultStack(/*bytes=*/1024 * 1024);
    useStack(/*chunks=*/1);  // pages in the code of useStack itself
    const long before = minorPageFaultsOfThisThread();
    useStack(/*chunks=*/32);  // 512 KiB, all within what was prefaulted
    faults = minorPageFaultsOfThisThread() - before;
  });
  thread.join();
  EXPECT_TRUE(status.ok()) << status;
  EXPECT_LE(faults, 4) << "touching half a megabyte of fresh stack would take over a hundred faults";
}

TEST(RealtimeThreadTest, prefaultingMoreStackThanTheThreadHasIsRefused) {
  absl::Status tooMuch;
  absl::Status nothing;
  std::thread thread([&]() {
    tooMuch = prefaultStack(/*bytes=*/std::size_t{1} << 30);
    nothing = prefaultStack(/*bytes=*/0);
  });
  thread.join();
  EXPECT_EQ(tooMuch.code(), absl::StatusCode::kOutOfRange) << tooMuch;
  EXPECT_TRUE(nothing.ok()) << nothing;
}

// ---------------------------------------------------------------------------------------------------------------------
// configureCurrentThread
// ---------------------------------------------------------------------------------------------------------------------

TEST(RealtimeThreadTest, aNonRealtimeConfigurationNamesAndPinsWithoutTouchingTheScheduler) {
  absl::Status status;
  std::string name;
  std::vector<int> original;
  std::vector<int> pinned;
  int policy = -1;
  std::thread thread([&]() {
    original = allowedCores();
    if (original.empty()) {
      return;
    }
    status = configureCurrentThread({.name = "rt_comm", .priority = 0, .cores = {original.front()}});
    name = currentThreadName();
    pinned = allowedCores();
    policy = currentPolicy();
  });
  thread.join();
  ASSERT_FALSE(original.empty());
  EXPECT_TRUE(status.ok()) << status;
  EXPECT_EQ(name, "rt_comm");
  EXPECT_EQ(pinned, std::vector<int>{original.front()});
  EXPECT_EQ(policy, SCHED_OTHER);
}

TEST(RealtimeThreadTest, aDefaultConfigurationChangesNothing) {
  absl::Status status;
  std::vector<int> before;
  std::vector<int> after;
  std::thread thread([&]() {
    before = allowedCores();
    status = configureCurrentThread(RealtimeThreadConfig{});
    after = allowedCores();
  });
  thread.join();
  EXPECT_TRUE(status.ok()) << status;
  EXPECT_EQ(after, before);
}

TEST(RealtimeThreadTest, configurationErrorsNameTheStepThatFailed) {
  RealtimeThreadConfig negativePriority;
  negativePriority.priority = -1;
  const absl::Status negative = configureCurrentThread(negativePriority);
  EXPECT_EQ(negative.code(), absl::StatusCode::kInvalidArgument) << negative;

  RealtimeThreadConfig tooLong;
  tooLong.name = "a_name_that_is_far_too_long";
  const absl::Status longName = configureCurrentThread(tooLong);
  EXPECT_EQ(longName.code(), absl::StatusCode::kInvalidArgument) << longName;
  EXPECT_TRUE(absl::StartsWith(longName.message(), "naming the thread")) << longName;
}

TEST(RealtimeThreadTest, aStepThatFailsDoesNotStopTheStepsAfterIt) {
  absl::Status status;
  std::vector<int> original;
  std::vector<int> pinned;
  std::thread thread([&]() {
    original = allowedCores();
    if (original.empty()) {
      return;
    }
    status = configureCurrentThread({.name = "a_name_that_is_far_too_long", .priority = 0, .cores = {original.back()}});
    pinned = allowedCores();
  });
  thread.join();
  ASSERT_FALSE(original.empty());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
  EXPECT_TRUE(absl::StartsWith(status.message(), "naming the thread")) << status;
  EXPECT_FALSE(absl::StrContains(status.message(), "pinning")) << status;
  EXPECT_EQ(pinned, std::vector<int>{original.back()}) << "the refused name kept the thread from being pinned";
}

TEST(RealtimeThreadTest, everyStepThatFailsIsNamedAndTheFirstGivesTheCode) {
  absl::Status status;
  std::thread thread([&]() { status = configureCurrentThread({.name = "a_name_that_is_far_too_long", .priority = 0, .cores = {-1}}); });
  thread.join();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
  EXPECT_TRUE(absl::StartsWith(status.message(), "naming the thread")) << status;
  EXPECT_TRUE(absl::StrContains(status.message(), "; pinning the thread: CPU core -1 does not exist")) << status;
}

}  // namespace
}  // namespace robot::realtime
