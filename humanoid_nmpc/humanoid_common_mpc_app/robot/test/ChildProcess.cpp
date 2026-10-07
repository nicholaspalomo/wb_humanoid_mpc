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

#include "humanoid_nmpc/humanoid_common_mpc_app/robot/test/ChildProcess.h"

#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/check.h"

#include "humanoid_nmpc/humanoid_common_mpc_app/robot/test/LoopbackNetwork.h"

namespace ocs2::humanoid::test_support {

ChildProcess::ChildProcess(const std::vector<std::string>& arguments) {
  CHECK(!arguments.empty()) << "a process needs a program";
  name_ = arguments.front();
  std::vector<char* absl_nullable> argv;  // null-terminated, as posix_spawn() takes it
  for (const std::string& argument : arguments) argv.push_back(const_cast<char*>(argument.c_str()));
  argv.push_back(nullptr);
  CHECK_EQ(::posix_spawn(&pid_, argv[0], /*file_actions=*/nullptr, /*attrp=*/nullptr, argv.data(), environ), 0) << "cannot start " << name_;
}

ChildProcess::~ChildProcess() {
  if (pid_ > 0 && !exited_) {
    ::kill(pid_, SIGKILL);
    ::waitpid(pid_, /*wstatus=*/nullptr, /*options=*/0);
  }
}

bool ChildProcess::running() {
  if (exited_) return false;
  int status = 0;
  if (::waitpid(pid_, &status, WNOHANG) == pid_) {
    exited_ = true;
    status_ = status;
  }
  return !exited_;
}

int ChildProcess::terminate(absl::Duration timeout) {
  if (running()) ::kill(pid_, SIGTERM);
  const bool exited = waitFor([this]() { return !running(); }, timeout);
  if (!exited) return -1;
  return WIFEXITED(status_) ? WEXITSTATUS(status_) : 128 + WTERMSIG(status_);
}

}  // namespace ocs2::humanoid::test_support
