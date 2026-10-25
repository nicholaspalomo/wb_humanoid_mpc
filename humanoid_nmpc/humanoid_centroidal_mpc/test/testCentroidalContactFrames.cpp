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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "humanoid_centroidal_mpc/CentroidalMpcConfig.h"
#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"

/*
 * The contact frames of the centroidal MPC's model are model_settings' names:
 * loadCustomPinocchioInterface() adds a frame of each name to the parent joint
 * model_settings gives it, and CentroidalMpcInterface checks once that every
 * contact frame exists (checkContactFramesExist()) before the centroidal model
 * info and the terms read them by index. A contact whose parent joint the model
 * does not have is refused with NotFound naming it
 * (loadCustomPinocchioInterface()), not built on an out-of-range parent index.
 */

namespace ocs2::humanoid {
namespace {

constexpr char kTaskFile[] =
    "robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto";
constexpr char kReferenceFile[] =
    "robot_models/unitree_g1/g1_centroidal_mpc/config/command/"
    "reference.textproto";
constexpr char kUrdfFile[] =
    "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf";

/** The absolute path of a data file of this test. */
std::string runfilePath(absl::string_view relativePath) {
  const char* absl_nullable srcDir = std::getenv("TEST_SRCDIR");
  const std::filesystem::path root =
      srcDir != nullptr ? std::filesystem::path(srcDir) / "_main"
                        : std::filesystem::current_path();
  return (root / std::string(relativePath)).string();
}

CentroidalMpcConfig shippedConfig() {
  absl::StatusOr<CentroidalMpcConfig> config = loadCentroidalMpcConfig(
      runfilePath(kTaskFile), runfilePath(kReferenceFile));
  EXPECT_TRUE(config.ok()) << config.status();
  return config.ok() ? *std::move(config) : CentroidalMpcConfig();
}

TEST(CentroidalContactFrames, EveryContactFrameOfTheShippedModelExists) {
  absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> interface =
      CentroidalMpcInterface::CreateControllerModels(shippedConfig(),
                                                     runfilePath(kUrdfFile));
  ASSERT_TRUE(interface.ok()) << interface.status();
  const std::vector<std::string>& contactNames =
      (*interface)->modelSettings().contactNames;
  EXPECT_TRUE(
      checkContactFramesExist((*interface)->getPinocchioInterface().getModel(),
                              contactNames)
          .ok());
}

TEST(CentroidalContactFrames, ARenamedContactFrameIsAddedUnderItsNewName) {
  CentroidalMpcConfig config = shippedConfig();
  ASSERT_EQ(config.task.model_settings.contact_names_6dof.size(), 2U);
  config.task.model_settings.contact_names_6dof[1] = "renamed_right_sole";
  absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> interface =
      CentroidalMpcInterface::CreateControllerModels(std::move(config),
                                                     runfilePath(kUrdfFile));
  ASSERT_TRUE(interface.ok()) << interface.status();
  EXPECT_TRUE((*interface)
                  ->getPinocchioInterface()
                  .getModel()
                  .existFrame("renamed_right_sole"));
}

TEST(CentroidalContactFrames, AContactOnAJointTheModelLacksIsRefused) {
  CentroidalMpcConfig config = shippedConfig();
  ASSERT_EQ(config.task.model_settings.contact_parent_joint_names.size(), 2U);
  config.task.model_settings.contact_parent_joint_names[1] =
      "no_such_ankle_joint";
  const absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> interface =
      CentroidalMpcInterface::CreateControllerModels(std::move(config),
                                                     runfilePath(kUrdfFile));
  ASSERT_FALSE(interface.ok());
  EXPECT_EQ(interface.status().code(), absl::StatusCode::kNotFound)
      << interface.status();
  EXPECT_NE(interface.status().message().find("no_such_ankle_joint"),
            absl::string_view::npos)
      << interface.status();
}

}  // namespace
}  // namespace ocs2::humanoid
