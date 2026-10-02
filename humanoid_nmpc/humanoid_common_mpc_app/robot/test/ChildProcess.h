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

#include <sys/types.h>

#include <string>
#include <vector>

#include "absl/time/time.h"

namespace ocs2::humanoid::test_support {

/**
 * A binary of the distributed runtime started as its own process, as a launch file starts it: posix_spawn with the
 * test's environment, working directory and output, stopped with SIGTERM like the launcher stops it. A process still
 * running when the handle is destroyed is killed (SIGKILL), so a failed test leaves nothing behind.
 */
class ChildProcess {
 public:
  /** Starts `arguments` (the program first; a path, not looked up in PATH). CHECK-fails when it cannot be spawned. */
  explicit ChildProcess(const std::vector<std::string>& arguments);

  /** SIGKILL and reap, unless it has exited. */
  ~ChildProcess();
  ChildProcess(const ChildProcess&) = delete;
  ChildProcess& operator=(const ChildProcess&) = delete;

  /** Whether it has not exited yet (reaps it when it has). */
  bool running();

  /**
   * SIGTERM, then its exit code: the code it exited with, 128 + N when a signal N ended it, or -1 when it did not exit
   * within `timeout` (it is then killed).
   */
  int terminate(absl::Duration timeout);

  /** The name the process was started as, for messages. */
  const std::string& name() const { return name_; }

 private:
  std::string name_;
  pid_t pid_ = -1;
  bool exited_ = false;
  int status_ = 0;
};

}  // namespace ocs2::humanoid::test_support
