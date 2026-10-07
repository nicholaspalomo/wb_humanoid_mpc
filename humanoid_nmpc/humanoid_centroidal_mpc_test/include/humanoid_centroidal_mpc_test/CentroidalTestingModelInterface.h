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

#include <memory>
#include <string>

#include "ocs2_centroidal_model/AccessHelperFunctions.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"
#include "ocs2_centroidal_model/CentroidalModelPinocchioMapping.h"
#include "ocs2_centroidal_model/ModelHelperFunctions.h"

#include "humanoid_centroidal_mpc/CentroidalMpcConfig.h"
#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "support/TypedConfigFiles.h"

namespace ocs2::humanoid {

/**
 * The centroidal MPC model of a shipped robot - task file, URDF, reference file, their typed configuration, Pinocchio
 * interface and MPC robot models - for the tests of this package. The Unitree G1 unless another robot is asked for.
 *
 * The files are read from the test's runfiles (BUILD `_TEST_DATA`), which are symlinks into the checkout and therefore
 * current. A file missing from them fails the calling test, and a configuration that does not convert aborts it with
 * the conversion's error.
 */
struct CentroidalTestingModelInterface {
 public:
  enum class Robot { kUnitreeG1, kDrcAtlas };

  std::string taskFile;
  std::string urdfFile;
  std::string referenceFile;
  // The typed task and reference files (loadCentroidalMpcConfig()).
  CentroidalMpcConfig config;

  std::unique_ptr<PinocchioInterface> pinocchioInterfacePtr;
  std::unique_ptr<ModelSettings> modelSettingsPtr;
  std::unique_ptr<CentroidalMpcRobotModel<scalar_t>> mpcRobotModelPtr_;
  std::unique_ptr<CentroidalMpcRobotModel<ad_scalar_t>> mpcRobotModelADPtr_;

  explicit CentroidalTestingModelInterface(Robot robot = Robot::kUnitreeG1) {
    // LINT.IfChange(testing_model_files)
    const CentroidalRobotFiles files = robot == Robot::kDrcAtlas ? atlasFiles() : g1Files();
    // LINT.ThenChange(//humanoid_nmpc/humanoid_centroidal_mpc_test/BUILD.bazel:test_data)
    taskFile = files.taskFile;
    urdfFile = files.urdfFile;
    referenceFile = files.referenceFile;
    config = loadConfigOf(files).value();

    modelSettingsPtr = std::make_unique<ModelSettings>(
        ModelSettings::Create(config.task, urdfFile, "centroidal_testing_interfce", /*verbose=*/false).value());

    pinocchioInterfacePtr = std::make_unique<PinocchioInterface>(
        loadCustomPinocchioInterface(config.task, urdfFile, *modelSettingsPtr, /*scaleTotalMass=*/false).value());
    mpcRobotModelPtr_ =
        std::make_unique<CentroidalMpcRobotModel<scalar_t>>(*modelSettingsPtr, *pinocchioInterfacePtr, getCentroidalModelInfo());
    mpcRobotModelADPtr_ = std::make_unique<CentroidalMpcRobotModel<ad_scalar_t>>(*modelSettingsPtr, (*pinocchioInterfacePtr).toCppAd(),
                                                                                 getCentroidalModelInfo().toCppAd());
  }

  PinocchioInterface& getPinocchioInterface() const { return *pinocchioInterfacePtr; }

  CentroidalMpcRobotModel<scalar_t>& getMpcRobotModel() const { return *mpcRobotModelPtr_; }
  CentroidalMpcRobotModel<ad_scalar_t>& getMpcRobotModelAD() const { return *mpcRobotModelADPtr_; }

  const ModelSettings& getModelSettings() const { return *modelSettingsPtr; }

  /** The centroidal model info of the configuration, as CentroidalMpcInterface builds it (centroidalModelInfoOf()). */
  CentroidalModelInfo getCentroidalModelInfo() const {
    return centroidalModelInfoOf(config, *pinocchioInterfacePtr, *modelSettingsPtr).value();
  }
};

}  // namespace ocs2::humanoid
