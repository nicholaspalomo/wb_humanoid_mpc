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

// Writes humanoid_nmpc/humanoid_mpc_config/config_registries.textproto in the checkout from the linked registries
// (tools/config_registries/README.md):
//
//   bazel run //tools/config_registries:print_config_registries
//
// Run it after a registry gains, loses or renames a name; //tools/config_registries:config_registries_test fails until
// the file is written again.

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "absl/base/nullability.h"
#include "absl/flags/parse.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

#include "tools/config_registries/ConfigRegistries.h"

int main(int argc, char* absl_nonnull* absl_nonnull argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  const char* absl_nullable workspace = std::getenv("BUILD_WORKSPACE_DIRECTORY");
  if (workspace == nullptr) {
    LOG(ERROR) << "run it with `bazel run`, which sets BUILD_WORKSPACE_DIRECTORY: it writes a file of the checkout.";
    return 2;
  }
  const std::string path = (std::filesystem::path(workspace) / ocs2::humanoid::config_registries::kConfigRegistriesFile).string();
  const std::string text =
      ocs2::humanoid::config_registries::configRegistriesText(ocs2::humanoid::config_registries::collectConfigRegistries());
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file << text;
  file.close();
  if (!file) {
    LOG(ERROR) << "cannot write " << path;
    return 1;
  }
  std::cout << "wrote " << path << "\n";
  return 0;
}
