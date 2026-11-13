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

#include "support/ShippedCentroidalRobot.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"
#include "ocs2_core/Types.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "support/TypedConfigFiles.h"

namespace ocs2::humanoid {

std::string shippedCentroidalRobotName(ShippedCentroidalRobot robot) {
  switch (robot) {
    case ShippedCentroidalRobot::kDrcAtlas:
      return "drc_atlas";
    case ShippedCentroidalRobot::kEngineAiSa01:
      return "engineai_sa01";
    case ShippedCentroidalRobot::kUnitreeG1:
      return "unitree_g1";
    case ShippedCentroidalRobot::kUnitreeR1:
      return "unitree_r1";
    case ShippedCentroidalRobot::kAnymalB:
      return "anymal_b";
    case ShippedCentroidalRobot::kTron1:
      return "tron1";
    case ShippedCentroidalRobot::kAnymalCWheels:
      return "anymal_c_wheels";
  }
  return "unknown";
}

std::string shippedCentroidalRobotTestName(
    const ::testing::TestParamInfo<ShippedCentroidalRobot>& info) {
  return shippedCentroidalRobotName(info.param);
}

CentroidalRobotFiles shippedCentroidalRobotFiles(ShippedCentroidalRobot robot) {
  switch (robot) {
    case ShippedCentroidalRobot::kDrcAtlas:
      return atlasFiles();
    case ShippedCentroidalRobot::kEngineAiSa01:
      return sa01Files();
    case ShippedCentroidalRobot::kUnitreeG1:
      return g1Files();
    case ShippedCentroidalRobot::kUnitreeR1:
      return r1Files();
    case ShippedCentroidalRobot::kAnymalB:
      return anymalBFiles();
    case ShippedCentroidalRobot::kTron1:
      return tron1Files();
    case ShippedCentroidalRobot::kAnymalCWheels:
      return anymalCWheelsFiles();
  }
  return CentroidalRobotFiles{};
}

std::string shippedCentroidalRobotGaitFile(ShippedCentroidalRobot robot) {
  switch (robot) {
    case ShippedCentroidalRobot::kDrcAtlas:
    case ShippedCentroidalRobot::kEngineAiSa01:
    case ShippedCentroidalRobot::kUnitreeG1:
    case ShippedCentroidalRobot::kUnitreeR1:
      return "humanoid_nmpc/humanoid_common_mpc/config/command/gait.textproto";
    // LINT.IfChange(anymal_b_gait_file)
    case ShippedCentroidalRobot::kAnymalB:
      return "humanoid_nmpc/humanoid_common_mpc/config/command/"
             "quadruped_gait.textproto";
      // LINT.ThenChange(//robot_models/anybotics_anymal_b/anymal_b_centroidal_mpc/launch/mpc.textproto:gait_file, //robot_models/anybotics_anymal_b/anymal_b_centroidal_mpc/launch/dummy_sim.textproto:gait_file, //humanoid_nmpc/humanoid_centroidal_mpc/BUILD.bazel:centroidal_mpc_loop_quadruped_gait)
    // LINT.IfChange(wheeled_gait_file)
    case ShippedCentroidalRobot::kTron1:
      return "humanoid_nmpc/humanoid_common_mpc/config/command/"
             "wheeled_gait.textproto";
      // LINT.ThenChange(//robot_models/limx_tron1/tron1_centroidal_mpc/launch/mpc.textproto:gait_file, //robot_models/limx_tron1/tron1_centroidal_mpc/launch/dummy_sim.textproto:gait_file, //humanoid_nmpc/humanoid_centroidal_mpc/BUILD.bazel:centroidal_mpc_loop_wheeled_gait)
    // LINT.IfChange(wheeled_quadruped_gait_file)
    case ShippedCentroidalRobot::kAnymalCWheels:
      return "humanoid_nmpc/humanoid_common_mpc/config/command/"
             "wheeled_quadruped_gait.textproto";
      // LINT.ThenChange(//robot_models/anybotics_anymal_c_wheels/anymal_c_wheels_centroidal_mpc/launch/mpc.textproto:gait_file, //robot_models/anybotics_anymal_c_wheels/anymal_c_wheels_centroidal_mpc/launch/dummy_sim.textproto:gait_file, //humanoid_nmpc/humanoid_centroidal_mpc/BUILD.bazel:centroidal_mpc_loop_wheeled_quadruped_gait)
  }
  return "";
}

absl::StatusOr<LoadedCentroidalRobot> loadShippedCentroidalRobot(
    ShippedCentroidalRobot robot, absl::string_view centroidalModel) {
  LoadedCentroidalRobot loaded;
  loaded.name = shippedCentroidalRobotName(robot);
  loaded.files = shippedCentroidalRobotFiles(robot);
  ASSIGN_OR_RETURN(loaded.config, loadConfigOf(loaded.files));
  if (!centroidalModel.empty()) {
    loaded.config.task.centroidal_model = std::string(centroidalModel);
  }
  ASSIGN_OR_RETURN(
      ModelSettings modelSettings,
      ModelSettings::Create(loaded.config.task, loaded.files.urdfFile,
                            "shipped_robot_test", /*verbose=*/false));
  loaded.modelSettings =
      std::make_unique<ModelSettings>(std::move(modelSettings));
  ASSIGN_OR_RETURN(
      PinocchioInterface pinocchioInterface,
      loadCustomPinocchioInterface(loaded.config.task, loaded.files.urdfFile,
                                   *loaded.modelSettings));
  loaded.pinocchioInterface =
      std::make_unique<PinocchioInterface>(std::move(pinocchioInterface));
  ASSIGN_OR_RETURN(
      CentroidalModelInfo info,
      centroidalModelInfoOf(loaded.config, *loaded.pinocchioInterface,
                            *loaded.modelSettings));
  loaded.info = std::make_unique<CentroidalModelInfo>(std::move(info));
  loaded.model = std::make_unique<CentroidalMpcRobotModel<scalar_t>>(
      *loaded.modelSettings, *loaded.pinocchioInterface, *loaded.info);
  return loaded;
}

}  // namespace ocs2::humanoid
