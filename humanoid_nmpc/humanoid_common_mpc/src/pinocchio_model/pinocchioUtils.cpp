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

#include <humanoid_common_mpc/pinocchio_model/pinocchioUtils.h>

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/multibody/model.hpp>
#include "pinocchio/parsers/urdf.hpp"

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"

namespace ocs2::humanoid {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

absl::Status checkPinocchioJointNaming(const PinocchioInterface& pinocchioInterface, const ModelSettings& modelSettings, bool verbose) {
  const pinocchio::Model& model = pinocchioInterface.getModel();
  const std::vector<std::string>& mpcJointNames = modelSettings.mpcModelJointNames;
  // The first two joints of the model are the universe and the floating base.
  constexpr size_t kFirstActuatedJoint = 2;
  const size_t numModelJoints = model.names.size() > kFirstActuatedJoint ? model.names.size() - kFirstActuatedJoint : 0;
  const size_t numCompared = std::min(numModelJoints, mpcJointNames.size());
  for (size_t i = 0; i < numCompared; ++i) {
    const std::string& modelJointName = model.names[i + kFirstActuatedJoint];
    if (verbose) {
      LOG(INFO) << "[checkPinocchioJointNaming] MPC joint " << i << ": model '" << modelJointName << "', ModelSettings '"
                << mpcJointNames[i] << "'";
    }
    if (mpcJointNames[i] != modelJointName) {
      return absl::InvalidArgumentError(absl::StrCat(
          "[checkPinocchioJointNaming] MPC joint ", i, " is '", mpcJointNames[i],
          "' in ModelSettings::mpcModelJointNames (the URDF's joints without model_settings.fixedJointNames) but '", modelJointName,
          "' in the Pinocchio model. The two were built from different URDFs or from different model_settings.fixedJointNames; "
          "build both from the same task file and URDF."));
    }
  }
  if (numModelJoints != mpcJointNames.size()) {
    const std::string firstUnmatched =
        numModelJoints > mpcJointNames.size() ? model.names[numCompared + kFirstActuatedJoint] : mpcJointNames[numCompared];
    return absl::InvalidArgumentError(absl::StrCat(
        "[checkPinocchioJointNaming] the Pinocchio model has ", numModelJoints, " actuated joints but ModelSettings::mpcModelJointNames ",
        mpcJointNames.size(), "; the first without a counterpart is '", firstUnmatched,
        "'. The two were built from different URDFs or from different model_settings.fixedJointNames; build both from the same task "
        "file and URDF."));
  }
  if (verbose) {
    LOG(INFO) << "[checkPinocchioJointNaming] the joint naming of the Pinocchio model matches ModelSettings.";
  }
  return absl::OkStatus();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::pair<vector_t, vector_t> readPinocchioJointLimits(const PinocchioInterface& pinocchioInterface,
                                                       const ModelSettings& modelSettings,
                                                       bool verbose) {
  // The limits are read by position, so the model's joint order has to be ModelSettings'. Every interface the MPC
  // builds has been checked by loadCustomPinocchioInterface() already; a caller that hands in any other breaks this
  // function's precondition, which is checked in every build (the assert() it replaces was compiled out of them).
  const absl::Status naming = checkPinocchioJointNaming(pinocchioInterface, modelSettings);
  CHECK(naming.ok()) << naming.message();
  const pinocchio::Model& model = pinocchioInterface.getModel();
  // Take the tail to avoid limits of universe and root joints
  vector_t upper_limits = model.upperPositionLimit.tail(modelSettings.mpcModelJointNames.size());
  vector_t lower_limits = model.lowerPositionLimit.tail(modelSettings.mpcModelJointNames.size());
  if (verbose) {
    LOG(INFO) << "Joint Name , min, max";
    for (size_t i = 0; i < modelSettings.mpcModelJointNames.size(); i++) {
      LOG(INFO) << modelSettings.mpcModelJointNames[i] << ": " << lower_limits[i] << ", " << upper_limits[i];
    }
  }
  return {lower_limits, upper_limits};
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void scalePinocchioModelInertia(pinocchio::ModelTpl<scalar_t>& model, scalar_t targetRobotMass, bool verbose) {
  scalar_t robotMass = pinocchio::computeTotalMass(model);
  scalar_t inertiaScaleFactor = targetRobotMass / robotMass;
  if (verbose) {
    LOG(INFO) << "Current robot mass: " << robotMass;
    LOG(INFO) << "Target robot mass: " << targetRobotMass;
    LOG(INFO) << "Adapting robot mass by a factor of " << inertiaScaleFactor << ".";
  }
  for (size_t i = 0; i < model.inertias.size(); i++) {
    const pinocchio::ModelTpl<scalar_t>::Inertia inertia = model.inertias[i];
    const scalar_t scaledMass = inertia.mass() * inertiaScaleFactor;
    matrix3_t inertiaMatrix = inertia.inertia().matrix();
    inertiaMatrix = inertiaMatrix * inertiaScaleFactor;
    const pinocchio::Symmetric3 scaledInertia(inertiaMatrix);
    model.inertias[i] = pinocchio::ModelTpl<scalar_t>::Inertia(scaledMass, inertia.lever(), scaledInertia);
  }
  if (verbose) {
    LOG(INFO) << "Robot mass scaled to " << pinocchio::computeTotalMass(model) << ".";
  }
}

}  // namespace ocs2::humanoid
