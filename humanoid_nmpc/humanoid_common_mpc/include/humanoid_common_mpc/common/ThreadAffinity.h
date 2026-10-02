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

#include <pthread.h>
#include <sched.h>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "absl/log/log.h"

namespace ocs2::humanoid {

/**
 * Sets the CPU core affinity for a specific pthread.
 *
 * @param cpuCores: List of 0-indexed CPU core IDs to pin the thread to.
 * @param thread: Target pthread (defaults to current calling thread).
 * @param threadName: Optional human-readable name for logging.
 * @return True if affinity was successfully set.
 */
inline bool setThreadCpuAffinity(const std::vector<int>& cpuCores, pthread_t thread = pthread_self(), const std::string& threadName = "") {
  if (cpuCores.empty()) {
    return true;
  }

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);

  unsigned int numHardwareThreads = std::thread::hardware_concurrency();
  std::vector<int> validCores;

  for (int core : cpuCores) {
    if (core >= 0 && (numHardwareThreads == 0 || static_cast<unsigned int>(core) < numHardwareThreads)) {
      CPU_SET(core, &cpuset);
      validCores.push_back(core);
    }
  }

  if (validCores.empty()) {
    LOG(WARNING) << "No valid CPU cores specified for thread affinity" << (threadName.empty() ? "" : " on " + threadName) << ".";
    return false;
  }

  int rc = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
  if (rc != 0) {
    LOG(WARNING) << "Failed to set thread CPU affinity" << (threadName.empty() ? "" : " on " + threadName) << " (error code: " << rc
                 << ").";
    return false;
  }

  std::ostringstream oss;
  for (size_t i = 0; i < validCores.size(); ++i) {
    oss << validCores[i] << (i + 1 < validCores.size() ? "," : "");
  }
  LOG(INFO) << "[ThreadAffinity] Successfully pinned " << (threadName.empty() ? "thread" : threadName) << " to CPU core(s): [" << oss.str()
            << "]";

  return true;
}

/**
 * System CPU core allocation partitions.
 */
struct SystemCoreAllocation {
  std::vector<int> simCores;  ///< Cores for MuJoCo physics and rendering
  std::vector<int> mrtCores;  ///< Cores for 500 Hz MRT joint control loop
  std::vector<int> mpcCores;  ///< Cores for MPC solver worker and SQP threads
};

/**
 * The CPUs this process may run on (sched_getaffinity: a container's cpuset, a taskset), ascending. Falls back to
 * 0 .. hardware_concurrency() - 1 when the affinity cannot be read.
 */
inline std::vector<int> availableCores() {
  std::vector<int> cores;
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(/*pid=*/0, sizeof(cpu_set_t), &allowed) == 0) {
    for (int core = 0; core < CPU_SETSIZE; ++core) {
      if (CPU_ISSET(core, &allowed)) cores.push_back(core);
    }
  }
  if (cores.empty()) {
    for (int core = 0; core < static_cast<int>(std::thread::hardware_concurrency()); ++core) cores.push_back(core);
  }
  return cores;
}

/** Appends cores[first, last) to `partition` (as far as `cores` reaches). */
inline void appendCores(const std::vector<int>& cores, size_t first, size_t last, std::vector<int>& partition) {
  for (size_t index = first; index < last && index < cores.size(); ++index) partition.push_back(cores[index]);
}

/**
 * Recommended core partitions over `cores` (ascending CPU numbers): the n-th entry of a partition is the n-th CPU of
 * `cores`, so that every core it names is one the process may use. With n cores:
 *   16 or more: the first 4 for the simulation and rendering, the next 2 for the 500 Hz MRT joint control loop, the
 *               next (up to 10) for the MPC solver and its SQP threads;
 *   8 to 15:    2 simulation, 1 MRT, the rest MPC;
 *   4 to 7:     1 simulation, 1 MRT, the rest MPC;
 *   fewer:      no restriction, every core in every partition.
 */
inline SystemCoreAllocation coreAllocationFor(const std::vector<int>& cores) {
  const size_t numCores = cores.size();
  SystemCoreAllocation alloc;
  if (numCores >= 16) {
    appendCores(cores, /*first=*/0, /*last=*/4, alloc.simCores);
    appendCores(cores, /*first=*/4, /*last=*/6, alloc.mrtCores);
    appendCores(cores, /*first=*/6, /*last=*/16, alloc.mpcCores);
  } else if (numCores >= 8) {
    appendCores(cores, /*first=*/0, /*last=*/2, alloc.simCores);
    appendCores(cores, /*first=*/2, /*last=*/3, alloc.mrtCores);
    appendCores(cores, /*first=*/3, numCores, alloc.mpcCores);
  } else if (numCores >= 4) {
    appendCores(cores, /*first=*/0, /*last=*/1, alloc.simCores);
    appendCores(cores, /*first=*/1, /*last=*/2, alloc.mrtCores);
    appendCores(cores, /*first=*/2, numCores, alloc.mpcCores);
  } else {
    // Under 4 cores: do not restrict
    alloc.simCores = cores;
    alloc.mrtCores = cores;
    alloc.mpcCores = cores;
  }
  return alloc;
}

/**
 * Recommended core partitions over the cores this process may use (availableCores()), not the machine's: in a
 * container started with a cpuset, `default` then names cores of that cpuset instead of cores the realtime thread
 * could not be pinned to. On a machine without a restriction it is what it always was, cores 0, 1, 2, ...
 */
inline SystemCoreAllocation getDefaultCoreAllocation() {
  return coreAllocationFor(availableCores());
}

}  // namespace ocs2::humanoid
