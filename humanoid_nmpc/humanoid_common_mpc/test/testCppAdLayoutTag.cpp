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

#include <array>
#include <filesystem>
#include <string>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/StateLayout.h"
#include "robot_core/ResourcePaths.h"

/*
 * The CppAD library folder of the MPCs carries the state layout's tag
 * (common/StateLayout.h, design decision D14 of
 * humanoid_nmpc/docs/quaternion_base_orientation/README.md). Every shipped task
 * file sets model_settings.recompile_libraries_cpp_ad: false, and a library is
 * reused whenever its .so file exists, so a library taped for another state
 * layout - a stale one of an untagged layout sits one folder up, in the folder
 * without the tag - must never be loaded for the quaternion layout: the folder
 * of every MPC is cppad_code_gen/cppad_<mpc><robot>/<tag>, the tag its last
 * path component, and the tag is pinned here with its spelling, so that
 * changing it (and so regenerating every library) is a deliberate edit of this
 * test too. The pattern of humanoid_wb_mpc's testJointTorqueCostLibraryName.
 */

namespace ocs2::humanoid {
namespace {

/** A shipped MPC's task file and URDF, from the repository root, and the name
 * prefix its interface creates it with. */
struct ShippedMpc {
  const char* absl_nonnull taskFile;
  const char* absl_nonnull urdfFile;
  const char* absl_nonnull mpcName;
};

constexpr std::array<ShippedMpc, 2> kMpcs = {{
    {.taskFile = "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/"
                 "task.textproto",
     .urdfFile = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf",
     .mpcName = "centroidal_mpc_"},
    {.taskFile = "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto",
     .urdfFile = "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf",
     .mpcName = "wb_mpc_"},
}};

TEST(CppAdLayoutTag, TheTagKeepsItsSpelling) {
  EXPECT_EQ(kStateLayoutTag, "translation_spherical_quaternion_v1");
}

TEST(CppAdLayoutTag, EveryMpcsLibraryFolderEndsInTheTag) {
  for (const ShippedMpc& mpc : kMpcs) {
    SCOPED_TRACE(mpc.taskFile);
    const absl::StatusOr<std::string> taskFile =
        robot::resolveResourcePath(mpc.taskFile);
    const absl::StatusOr<std::string> urdfFile =
        robot::resolveResourcePath(mpc.urdfFile);
    ASSERT_TRUE(taskFile.ok() && urdfFile.ok())
        << "the shipped files are not in the runfiles";
    const absl::StatusOr<ModelSettings> settings = ModelSettings::Create(
        *taskFile, *urdfFile, mpc.mpcName, /*verbose=*/false);
    ASSERT_TRUE(settings.ok()) << settings.status();
    const std::string folderWithoutTag =
        absl::StrCat("cppad_code_gen/cppad_", mpc.mpcName, settings->robotName);
    EXPECT_EQ(settings->modelFolderCppAd,
              absl::StrCat(folderWithoutTag, "/", kStateLayoutTag));
    const std::filesystem::path folder(settings->modelFolderCppAd);
    EXPECT_EQ(folder.filename().string(), std::string(kStateLayoutTag))
        << "the tag is the folder's last component";
    EXPECT_EQ(folder.parent_path().string(), folderWithoutTag)
        << "the libraries of an untagged layout, one folder up, are never read "
           "for the quaternion layout";
    EXPECT_TRUE(absl::StrContains(settings->modelFolderCppAd, kStateLayoutTag));
  }
}

}  // namespace
}  // namespace ocs2::humanoid
