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

// Prints every value the start-up of a robot configuration's stack builds from its files (tools/config_dump/README.md),
// one configuration per run:
//
//   bazel run //tools/config_dump -- --robot=drc_atlas > before.txt
//   # edit the configuration, or the code that reads it
//   bazel run //tools/config_dump -- --robot=drc_atlas | diff before.txt -
//
// It runs in the checkout ($BUILD_WORKSPACE_DIRECTORY), so that the files are the checkout's and the MPC reuses the
// CppAD libraries of its cppad_code_gen/. The dump goes to the standard output and nothing else does.

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>

#include "absl/base/nullability.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"

#include "humanoid_mpc_validation/closed_loop/RobotConfiguration.h"
#include "tools/config_dump/StackDump.h"

ABSL_FLAG(std::string,
          robot,
          "",
          "The configuration (RobotConfiguration::name): drc_atlas, engineai_sa01, unitree_g1, unitree_r1, unitree_g1_wb.");

int main(int argc, char* absl_nonnull* absl_nonnull argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kWarning);
  const char* absl_nullable workspace = std::getenv("BUILD_WORKSPACE_DIRECTORY");
  if (workspace == nullptr) {
    LOG(ERROR) << "run it with `bazel run`, which sets BUILD_WORKSPACE_DIRECTORY: the dump reads the checkout's files.";
    return 2;
  }
  std::error_code error;
  std::filesystem::current_path(workspace, error);
  if (error) {
    LOG(ERROR) << "cannot change into " << workspace << ": " << error.message();
    return 2;
  }
  const absl::StatusOr<ocs2::humanoid::validation::RobotConfiguration> configuration =
      ocs2::humanoid::validation::findRobotConfiguration(absl::GetFlag(FLAGS_robot));
  if (!configuration.ok()) {
    LOG(ERROR) << configuration.status();
    return 2;
  }
  const absl::StatusOr<std::string> dump = ocs2::humanoid::config_dump::dumpConfiguration(*configuration);
  if (!dump.ok()) {
    LOG(ERROR) << configuration->name << ": " << dump.status();
    return 1;
  }
  std::cout << *dump;
  return std::cout.good() ? 0 : 1;
}
