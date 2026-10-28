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

#include "humanoid_multi_contact/app/CommandLinePaths.h"

#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>

#include "absl/base/nullability.h"

namespace ocs2::humanoid::multi_contact {

std::string resolveInputPath(const std::string& path) {
  if (path.empty() || std::filesystem::path(path).is_absolute()) return path;
  for (const char* absl_nullable variable :
       {std::getenv("BUILD_WORKING_DIRECTORY"),
        std::getenv("BUILD_WORKSPACE_DIRECTORY")}) {
    if (variable == nullptr) continue;
    const std::filesystem::path candidate =
        std::filesystem::path(variable) / path;
    std::error_code error;
    if (std::filesystem::exists(candidate, error)) return candidate.string();
  }
  return path;
}

std::string resolveOutputPath(const std::string& path) {
  const char* absl_nullable directory = std::getenv("BUILD_WORKING_DIRECTORY");
  if (std::filesystem::path(path).is_absolute() || directory == nullptr) {
    return path;
  }
  return (std::filesystem::path(directory) / path).string();
}

}  // namespace ocs2::humanoid::multi_contact
