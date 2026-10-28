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

#include "support/MultiContactTestFiles.h"

#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "robot_core/ResourcePaths.h"

namespace ocs2::humanoid {
namespace {

/** The runfiles path of `relativePath`; a missing file fails the calling test
 * and is empty. */
std::string runfile(absl::string_view relativePath) {
  absl::StatusOr<std::string> path = robot::resolveResourcePath(relativePath);
  EXPECT_TRUE(path.ok()) << path.status();
  return path.ok() ? *std::move(path) : std::string();
}

}  // namespace

CentroidalRobotFiles atlasMultiContactFiles() {
  return CentroidalRobotFiles{
      .taskFile = runfile(kAtlasMultiContactTaskFile),
      .referenceFile = runfile(kAtlasMultiContactReferenceFile),
      .urdfFile = runfile(kAtlasUrdfFile)};
}

absl::StatusOr<multi_contact::ContactSet> loadAtlasContactSet() {
  ASSIGN_OR_RETURN(const std::string path,
                   robot::resolveResourcePath(kAtlasContactSetFile));
  ASSIGN_OR_RETURN(const multi_contact::config::ContactSet config,
                   multi_contact::loadContactSetFile(path));
  return multi_contact::ContactSet::Create(config);
}

}  // namespace ocs2::humanoid
