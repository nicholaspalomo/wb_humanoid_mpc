/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

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

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <ocs2_centroidal_model/AccessHelperFunctions.h>
#include <ocs2_centroidal_model/CentroidalModelInfo.h>
#include <ocs2_centroidal_model/CentroidalModelPinocchioMapping.h>
#include <ocs2_centroidal_model/FactoryFunctions.h>
#include <ocs2_centroidal_model/ModelHelperFunctions.h>

#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"

#include <ament_index_cpp/get_package_share_directory.hpp>

namespace ocs2::humanoid {

/**
 * The centroidal MPC model of a shipped robot - task file, URDF, reference file, Pinocchio interface and MPC robot
 * models - for the tests of this package. The Unitree G1 unless another robot is asked for.
 *
 * The files are read from the test's runfiles, which are symlinks into the checkout and therefore current; the ament
 * index, which the dev container populates from a COPY of the source tree that can be stale, is only the fallback.
 */
struct CentroidalTestingModelInterface {
 public:
  enum class Robot { kUnitreeG1, kDrcAtlas };

  std::string taskFile;
  std::string urdfFile;
  std::string referenceFile;

  std::unique_ptr<PinocchioInterface> pinocchioInterfacePtr;
  std::unique_ptr<ModelSettings> modelSettingsPtr;
  std::unique_ptr<CentroidalMpcRobotModel<scalar_t>> mpcRobotModelPtr_;
  std::unique_ptr<CentroidalMpcRobotModel<ad_scalar_t>> mpcRobotModelADPtr_;

  explicit CentroidalTestingModelInterface(Robot robot = Robot::kUnitreeG1) {
    // LINT.IfChange(testing_model_files)
    if (robot == Robot::kDrcAtlas) {
      taskFile = locate("drc_atlas_centroidal_mpc", "robot_models/drc_atlas/drc_atlas_centroidal_mpc", "config/mpc/task.yaml");
      urdfFile = locate("drc_atlas_description", "robot_models/drc_atlas/drc_atlas_description", "urdf/atlas.urdf");
      referenceFile =
          locate("drc_atlas_centroidal_mpc", "robot_models/drc_atlas/drc_atlas_centroidal_mpc", "config/command/reference.yaml");
    } else {
      taskFile = locate("g1_centroidal_mpc", "robot_models/unitree_g1/g1_centroidal_mpc", "config/mpc/task.yaml");
      urdfFile = locate("g1_description", "robot_models/unitree_g1/g1_description", "urdf/g1_29dof.urdf");
      referenceFile = locate("g1_centroidal_mpc", "robot_models/unitree_g1/g1_centroidal_mpc", "config/command/reference.yaml");
    }
    // LINT.ThenChange(//humanoid_nmpc/humanoid_centroidal_mpc_test/BUILD.bazel:test_data)

    modelSettingsPtr = std::make_unique<ModelSettings>(taskFile, urdfFile, "centroidal_testing_interfce", /*verbose=*/false);

    pinocchioInterfacePtr = std::make_unique<PinocchioInterface>(createCustomPinocchioInterface(taskFile, urdfFile, *modelSettingsPtr));
    mpcRobotModelPtr_ =
        std::make_unique<CentroidalMpcRobotModel<scalar_t>>(*modelSettingsPtr, *pinocchioInterfacePtr, getCentroidalModelInfo());
    mpcRobotModelADPtr_ = std::make_unique<CentroidalMpcRobotModel<ad_scalar_t>>(*modelSettingsPtr, (*pinocchioInterfacePtr).toCppAd(),
                                                                                 getCentroidalModelInfo().toCppAd());
  }

  PinocchioInterface& getPinocchioInterface() const { return *pinocchioInterfacePtr; }

  CentroidalMpcRobotModel<scalar_t>& getMpcRobotModel() { return *mpcRobotModelPtr_; }
  CentroidalMpcRobotModel<ad_scalar_t>& getMpcRobotModelAD() { return *mpcRobotModelADPtr_; }

  const ModelSettings& getModelSettings() const { return *modelSettingsPtr; }

  CentroidalModelInfo getCentroidalModelInfo() const {
    return centroidal_model::createCentroidalModelInfo(
        *pinocchioInterfacePtr, centroidal_model::loadCentroidalType(taskFile),
        centroidal_model::loadDefaultJointState(pinocchioInterfacePtr->getModel().nq - 6, referenceFile),
        modelSettingsPtr->contactNames3DoF, modelSettingsPtr->contactNames6DoF);
  }

 private:
  /** `packageDirectory`/`relativePath` in the runfiles, else `relativePath` in the ament share directory of `package`. */
  static std::string locate(const std::string& package, const std::string& packageDirectory, const std::string& relativePath) {
    std::vector<std::filesystem::path> roots;
    if (const char* srcDir = std::getenv("TEST_SRCDIR")) roots.emplace_back(std::filesystem::path(srcDir) / "_main");
    roots.emplace_back(std::filesystem::current_path());
    for (const std::filesystem::path& root : roots) {
      const std::filesystem::path candidate = root / packageDirectory / relativePath;
      if (std::filesystem::exists(candidate)) return candidate.string();
    }
    return (std::filesystem::path(ament_index_cpp::get_package_share_directory(package)) / relativePath).string();
  }
};

}  // namespace ocs2::humanoid
