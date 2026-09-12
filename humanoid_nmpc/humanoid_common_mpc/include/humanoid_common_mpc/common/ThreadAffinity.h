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
    std::cerr << "WARNING: No valid CPU cores specified for thread affinity" << (threadName.empty() ? "" : " on " + threadName) << "."
              << std::endl;
    return false;
  }

  int rc = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
  if (rc != 0) {
    std::cerr << "WARNING: Failed to set thread CPU affinity" << (threadName.empty() ? "" : " on " + threadName) << " (error code: " << rc
              << ")." << std::endl;
    return false;
  }

  std::ostringstream oss;
  for (size_t i = 0; i < validCores.size(); ++i) {
    oss << validCores[i] << (i + 1 < validCores.size() ? "," : "");
  }
  std::cout << "[ThreadAffinity] Successfully pinned " << (threadName.empty() ? "thread" : threadName) << " to CPU core(s): [" << oss.str()
            << "]" << std::endl;

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
 * Computes recommended core partitions based on available hardware cores.
 */
inline SystemCoreAllocation getDefaultCoreAllocation() {
  unsigned int numCores = std::thread::hardware_concurrency();
  SystemCoreAllocation alloc;

  if (numCores >= 16) {
    // 16+ cores (e.g. 20-core systems):
    // Cores 0-3: Simulation and Rendering (4 cores)
    // Cores 4-5: MRT 500 Hz Joint Controller Loop (2 cores)
    // Cores 6-15: MPC Solver and SQP threads (10 dedicated cores)
    alloc.simCores = {0, 1, 2, 3};
    alloc.mrtCores = {4, 5};
    for (int i = 6; i < static_cast<int>(std::min(numCores, 16u)); ++i) {
      alloc.mpcCores.push_back(i);
    }
  } else if (numCores >= 8) {
    // 8-15 cores:
    // Cores 0-1: Sim & Rendering
    // Core 2: MRT Joint Controller Loop
    // Cores 3-7: MPC Solver
    alloc.simCores = {0, 1};
    alloc.mrtCores = {2};
    for (int i = 3; i < static_cast<int>(numCores); ++i) {
      alloc.mpcCores.push_back(i);
    }
  } else if (numCores >= 4) {
    // 4-7 cores:
    alloc.simCores = {0};
    alloc.mrtCores = {1};
    for (int i = 2; i < static_cast<int>(numCores); ++i) {
      alloc.mpcCores.push_back(i);
    }
  } else {
    // Under 4 cores: do not restrict
    for (int i = 0; i < static_cast<int>(numCores); ++i) {
      alloc.simCores.push_back(i);
      alloc.mrtCores.push_back(i);
      alloc.mpcCores.push_back(i);
    }
  }

  return alloc;
}

}  // namespace ocs2::humanoid
