/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

#include <pinocchio/fwd.hpp>

#include "humanoid_common_mpc/contact/FootprintCornerHeights.h"

#include <cmath>
#include <utility>

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include "absl/log/check.h"

#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"

namespace ocs2::humanoid {

FootprintCornerHeights::FootprintCornerHeights(const PinocchioInterface& pinocchioInterface,
                                               const MpcRobotModelBase<ad_scalar_t>& mpcRobotModelAD,
                                               std::vector<std::string> frameNames,
                                               const std::string& modelName,
                                               const ModelSettings& modelSettings)
    : frameNames_(std::move(frameNames)),
      mpcRobotModelAdPtr_(mpcRobotModelAD.clone()),
      pinocchioInterfaceCppAd_(pinocchioInterface.toCppAd()),
      modelName_(modelName),
      modelFolder_(modelSettings.modelFolderCppAd),
      verbose_(modelSettings.verboseCppAd) {
  CHECK(!frameNames_.empty()) << "[FootprintCornerHeights] " << modelName_ << " needs at least one frame";
  frameIds_.reserve(frameNames_.size());
  for (const std::string& frameName : frameNames_) {
    CHECK(pinocchioInterfaceCppAd_.getModel().existFrame(frameName))
        << "[FootprintCornerHeights] " << modelName_ << ": the robot model has no frame '" << frameName
        << "'. The footprint corner frames are added by createPinocchioModel(); check that the contact polygon of this foot is "
           "configured.";
    frameIds_.push_back(pinocchioInterfaceCppAd_.getModel().getFrameId(frameName));
  }
  createAdInterface(modelSettings.recompileLibrariesCppAd);
}

FootprintCornerHeights::FootprintCornerHeights(const FootprintCornerHeights& rhs)
    : frameNames_(rhs.frameNames_),
      frameIds_(rhs.frameIds_),
      mpcRobotModelAdPtr_(rhs.mpcRobotModelAdPtr_->clone()),
      pinocchioInterfaceCppAd_(rhs.pinocchioInterfaceCppAd_),
      modelName_(rhs.modelName_),
      modelFolder_(rhs.modelFolder_),
      verbose_(rhs.verbose_) {
  createAdInterface(false);
}

void FootprintCornerHeights::createAdInterface(bool regenerate) {
  const size_t stateDim = mpcRobotModelAdPtr_->getStateDim();
  CppAdInterface::ad_function_t heightsAd = [this, stateDim](const ad_vector_t& x, ad_vector_t& y) {
    CHECK_EQ(static_cast<size_t>(x.rows()), stateDim);
    y = this->heightsFunction(x);
  };
  adInterfacePtr_ = std::make_unique<CppAdInterface>(std::move(heightsAd), stateDim, modelName_, modelFolder_);

  // First order only. Both callers are ConstraintOrder::Linear, so the second-order tape would be generated, compiled
  // and then never evaluated - which is precisely the waste this class exists to avoid.
  if (regenerate) {
    adInterfacePtr_->createModels(CppAdInterface::ApproximationOrder::First, verbose_);
  } else {
    adInterfacePtr_->loadModelsIfAvailable(CppAdInterface::ApproximationOrder::First, verbose_);
  }
}

ad_vector_t FootprintCornerHeights::heightsFunction(const ad_vector_t& state) const {
  const pinocchio::ModelTpl<ad_scalar_t>& model = pinocchioInterfaceCppAd_.getModel();
  pinocchio::DataTpl<ad_scalar_t> data = pinocchioInterfaceCppAd_.getData();  // a copy, since this method is const
  updateFramePlacements(mpcRobotModelAdPtr_->getGeneralizedCoordinates(state), model, data);

  ad_vector_t heights(frameIds_.size());
  for (size_t corner = 0; corner < frameIds_.size(); ++corner) {
    heights(static_cast<long>(corner)) = data.oMf[frameIds_[corner]].translation()(2);
  }
  return heights;
}

vector_t FootprintCornerHeights::getHeights(const vector_t& state) const {
  return adInterfacePtr_->getFunctionValue(state);
}

matrix_t FootprintCornerHeights::getHeightsJacobian(const vector_t& state) const {
  return adInterfacePtr_->getJacobian(state);
}

SmoothMinimumHeight smoothMinimumHeight(const vector_t& heights, scalar_t smoothing) {
  CHECK_GT(heights.size(), 0) << "[smoothMinimumHeight] needs at least one height";
  CHECK_GT(smoothing, 0.0) << "[smoothMinimumHeight] contact_implicit.gapSmoothing must be positive";

  // Shifting by the true minimum before exponentiating is what keeps this finite: every exponent is then non-positive,
  // so the terms lie in (0, 1] and the sum is at least one - no overflow, and no underflow that could empty the sum.
  const scalar_t minimum = heights.minCoeff();
  const vector_t exponentials = ((minimum - heights.array()) / smoothing).exp();
  const scalar_t sum = exponentials.sum();

  SmoothMinimumHeight result;
  result.value = minimum - smoothing * std::log(sum / static_cast<scalar_t>(heights.size()));
  result.weights = exponentials / sum;
  return result;
}

}  // namespace ocs2::humanoid
