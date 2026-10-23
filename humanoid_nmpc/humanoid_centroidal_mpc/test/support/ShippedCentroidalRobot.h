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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "humanoid_centroidal_mpc/CentroidalMpcConfig.h"
#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "support/TypedConfigFiles.h"

namespace ocs2::humanoid {

/** A shipped configuration of the centroidal MPC; the files of each are the data of :shipped_centroidal_robot. */
// LINT.IfChange(shipped_centroidal_robots)
enum class ShippedCentroidalRobot { kDrcAtlas, kEngineAiSa01, kUnitreeG1, kUnitreeR1 };
// LINT.ThenChange(//humanoid_nmpc/humanoid_centroidal_mpc/BUILD.bazel:shipped_centroidal_robot_data)

/** The configuration's name for a test case: "drc_atlas", "engineai_sa01", "unitree_g1", "unitree_r1". */
std::string shippedCentroidalRobotName(ShippedCentroidalRobot robot);

/** The test-case name of a parameterized test over the shipped configurations. */
std::string shippedCentroidalRobotTestName(const ::testing::TestParamInfo<ShippedCentroidalRobot>& info);

/** The configuration's files in the test's runfiles (TypedConfigFiles.h). */
CentroidalRobotFiles shippedCentroidalRobotFiles(ShippedCentroidalRobot robot);

/**
 * The robot model of a shipped configuration, with what it refers to, as CentroidalMpcInterface builds it but without
 * its solver or any CppAD library: the typed files, the model settings, the reduced Pinocchio model, the centroidal model
 * info and the wrench-input robot model on them. The members are owned by pointer, so that a move keeps the model's
 * references valid. Not thread-safe (the robot model updates its Pinocchio data).
 */
struct LoadedCentroidalRobot {
  std::string name;
  CentroidalRobotFiles files;
  CentroidalMpcConfig config;
  std::unique_ptr<ModelSettings> modelSettings;
  std::unique_ptr<PinocchioInterface> pinocchioInterface;
  std::unique_ptr<CentroidalModelInfo> info;
  std::unique_ptr<CentroidalMpcRobotModel<scalar_t>> model;
};

/**
 * Loads `robot`, with the task file's centroidal_model replaced by `centroidalModel` (e.g. "single_rigid_body_dynamics")
 * when it is not empty. Fails like the loaders it calls (a missing or invalid file).
 */
absl::StatusOr<LoadedCentroidalRobot> loadShippedCentroidalRobot(ShippedCentroidalRobot robot, absl::string_view centroidalModel);

}  // namespace ocs2::humanoid
