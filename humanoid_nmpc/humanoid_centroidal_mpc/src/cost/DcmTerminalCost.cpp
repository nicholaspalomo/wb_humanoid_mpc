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

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include "humanoid_centroidal_mpc/cost/DcmTerminalCost.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <boost/property_tree/ptree.hpp>

#include <ocs2_core/misc/LoadData.h>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/frames.hpp>

#include <humanoid_common_mpc/gait/MotionPhaseDefinition.h>
#include <humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h>

namespace ocs2::humanoid {

namespace {
constexpr size_t kNumParameters = 8;
constexpr size_t kResidualDim = 2;
}  // namespace

scalar_t DcmTerminalCost::Config::omega() const {
  return std::sqrt(gravity / comHeight);
}

void DcmTerminalCost::Config::validate() const {
  if (comHeight <= 0.0 || gravity <= 0.0) {
    throw std::invalid_argument("[DcmTerminalCost] comHeight and gravity must be positive");
  }
  if (weights.minCoeff() < 0.0) {
    throw std::invalid_argument("[DcmTerminalCost] weights must be non-negative");
  }
  if (supportBlendTime < 0.0) {
    throw std::invalid_argument("[DcmTerminalCost] supportBlendTime must be non-negative");
  }
}

DcmTerminalCost::DcmTerminalCost(const SwitchedModelReferenceManager& referenceManager,
                                 Config config,
                                 const PinocchioInterface& pinocchioInterface,
                                 const MpcRobotModelBase<ad_scalar_t>& mpcRobotModelAD,
                                 std::string costName,
                                 const ModelSettings& modelSettings)
    : referenceManagerPtr_(&referenceManager),
      config_(std::move(config)),
      pinocchioInterfaceCppAd_(pinocchioInterface.toCppAd()),
      mpcRobotModelAdPtr_(mpcRobotModelAD.clone()) {
  config_.validate();
  auto residualAd = [this](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) { y = this->residual(x, p); };
  adInterfacePtr_.reset(
      new CppAdInterface(residualAd, mpcRobotModelAD.getStateDim(), kNumParameters, std::move(costName), modelSettings.modelFolderCppAd));
  if (modelSettings.recompileLibrariesCppAd) {
    adInterfacePtr_->createModels(CppAdInterface::ApproximationOrder::First, modelSettings.verboseCppAd);
  } else {
    adInterfacePtr_->loadModelsIfAvailable(CppAdInterface::ApproximationOrder::First, modelSettings.verboseCppAd);
  }
  std::cout << "Initialized DcmTerminalCost with weights " << config_.weights.transpose() << ", comHeight " << config_.comHeight
            << ", velocityOffsetFactor " << config_.velocityOffsetFactor << std::endl;
}

DcmTerminalCost::DcmTerminalCost(const DcmTerminalCost& other)
    : StateCost(other),
      referenceManagerPtr_(other.referenceManagerPtr_),
      config_(other.config_),
      isActive_(other.isActive_),
      pinocchioInterfaceCppAd_(other.pinocchioInterfaceCppAd_),
      mpcRobotModelAdPtr_(other.mpcRobotModelAdPtr_->clone()),
      adInterfacePtr_(new CppAdInterface(*other.adInterfacePtr_)) {}

void DcmTerminalCost::setConfig(const Config& config) {
  config.validate();
  config_ = config;
}

ad_vector_t DcmTerminalCost::residual(const ad_vector_t& state, const ad_vector_t& parameters) {
  const ad_scalar_t weightLeft = parameters(0);
  const ad_scalar_t weightRight = parameters(1);
  const ad_scalar_t omega = parameters(2);
  const ad_vector2_t velocityCommand = parameters.segment<2>(3);
  const ad_scalar_t offsetFactor = parameters(5);
  const ad_vector2_t sqrtWeights = parameters.segment<2>(6);

  const auto& model = pinocchioInterfaceCppAd_.getModel();
  auto& data = pinocchioInterfaceCppAd_.getData();
  const ad_vector_t q = mpcRobotModelAdPtr_->getGeneralizedCoordinates(state);
  pinocchio::centerOfMass(model, data, q, false);
  pinocchio::updateFramePlacements(model, data);
  const ad_vector2_t com = data.com[0].head<2>();
  const ad_vector2_t comVelocity = mpcRobotModelAdPtr_->getBaseComLinearVelocity(state).head<2>();
  const std::vector<ad_vector3_t> contactPositions = getContactPositions<ad_scalar_t>(pinocchioInterfaceCppAd_, *mpcRobotModelAdPtr_);
  const ad_vector2_t support =
      (weightLeft * contactPositions[0].head<2>() + weightRight * contactPositions[1].head<2>()) / (weightLeft + weightRight);

  const ad_vector2_t dcm = com + comVelocity / omega;
  const ad_vector2_t dcmReference = support + offsetFactor * velocityCommand / omega;
  ad_vector_t r(kResidualDim);
  r = (dcm - dcmReference).cwiseProduct(sqrtWeights);
  return r;
}

vector2_t DcmTerminalCost::computeSupportWeights(scalar_t time) const {
  const contact_flag_t contacts = referenceManagerPtr_->getContactFlags(time);
  vector2_t weights(contacts[0] ? 1.0 : 0.0, contacts[1] ? 1.0 : 0.0);
  const ModeSchedule& schedule = referenceManagerPtr_->getModeSchedule();
  const auto& eventTimes = schedule.eventTimes;
  const auto& modeSequence = schedule.modeSequence;
  if (config_.supportBlendTime > 0.0 && !modeSequence.empty()) {
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      if (!contacts[foot]) continue;
      // Contact phase [start, end] of this foot around `time`.
      size_t index = static_cast<size_t>(std::upper_bound(eventTimes.begin(), eventTimes.end(), time) - eventTimes.begin());
      if (index >= modeSequence.size()) index = modeSequence.size() - 1;
      const auto inContact = [&](size_t i) { return modeNumber2StanceLeg(modeSequence[i])[foot]; };
      size_t first = index;
      while (first > 0 && inContact(first - 1)) --first;
      size_t last = index;
      while (last + 1 < modeSequence.size() && inContact(last + 1)) ++last;
      const scalar_t start = (first == 0) ? -std::numeric_limits<scalar_t>::infinity() : eventTimes[first - 1];
      const scalar_t end = (last + 1 >= modeSequence.size()) ? std::numeric_limits<scalar_t>::infinity() : eventTimes[last];
      const scalar_t ramp = std::min((time - start) / config_.supportBlendTime, (end - time) / config_.supportBlendTime);
      weights(foot) = std::clamp(ramp, 0.0, 1.0);
    }
  }
  if (weights.sum() < 1e-6) {
    weights.setOnes();  // flight, or both feet at a transition: fall back to the centre of both feet
  }
  return weights;
}

vector_t DcmTerminalCost::getParameters(scalar_t time, const TargetTrajectories& targetTrajectories) const {
  const vector2_t supportWeights = computeSupportWeights(time);
  vector2_t velocityCommand = vector2_t::Zero();
  if (!targetTrajectories.empty()) {
    const vector_t desiredState = targetTrajectories.getDesiredState(time);
    if (desiredState.size() >= 2) {
      velocityCommand = desiredState.head<2>();
    }
  }
  vector_t parameters(kNumParameters);
  parameters(0) = supportWeights(0);
  parameters(1) = supportWeights(1);
  parameters(2) = config_.omega();
  parameters.segment<2>(3) = velocityCommand;
  parameters(5) = config_.velocityOffsetFactor;
  parameters.segment<2>(6) = config_.weights.cwiseSqrt();
  return parameters;
}

vector2_t DcmTerminalCost::computeDcmError(const vector_t& state, const vector_t& parameters) const {
  vector_t unweighted = parameters;
  unweighted.segment<2>(6).setOnes();
  return adInterfacePtr_->getFunctionValue(state, unweighted).head<2>();
}

scalar_t DcmTerminalCost::getValue(scalar_t time,
                                   const vector_t& state,
                                   const TargetTrajectories& targetTrajectories,
                                   const PreComputation& /*preComputation*/) const {
  const vector_t parameters = getParameters(time, targetTrajectories);
  const vector_t r = adInterfacePtr_->getFunctionValue(state, parameters);
  return 0.5 * r.squaredNorm();
}

ScalarFunctionQuadraticApproximation DcmTerminalCost::getQuadraticApproximation(scalar_t time,
                                                                                const vector_t& state,
                                                                                const TargetTrajectories& targetTrajectories,
                                                                                const PreComputation& /*preComputation*/) const {
  const vector_t parameters = getParameters(time, targetTrajectories);
  return adInterfacePtr_->getGaussNewtonApproximation(state, parameters);
}

DcmTerminalCost::Config DcmTerminalCost::loadConfig(const std::string& taskFile, const std::string& prefix, bool verbose) {
  boost::property_tree::ptree pt;
  loadData::readPropertyTree(taskFile, pt);
  Config config;
  if (verbose) {
    std::cerr << "\n #### DCM Terminal Cost Config:";
    std::cerr << "\n #### =============================================================================\n";
  }
  // LINT.IfChange(dcm_terminal_cost_keys)
  loadData::loadPtreeValue(pt, config.comHeight, prefix + "comHeight", verbose);
  loadData::loadPtreeValue(pt, config.gravity, prefix + "gravity", verbose);
  loadData::loadPtreeValue(pt, config.weights(0), prefix + "weight_x", verbose);
  loadData::loadPtreeValue(pt, config.weights(1), prefix + "weight_y", verbose);
  loadData::loadPtreeValue(pt, config.velocityOffsetFactor, prefix + "velocityOffsetFactor", verbose);
  loadData::loadPtreeValue(pt, config.supportBlendTime, prefix + "supportBlendTime", verbose);
  // LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml:dcm_terminal_cost_config)
  if (verbose) {
    std::cerr << " #### =============================================================================" << std::endl;
  }
  config.validate();
  return config;
}

}  // namespace ocs2::humanoid
