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

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc_app/robot/RobotController.h"
#include "humanoid_mpc_config/joint_pd_gains_file.nproto.h"

namespace ocs2::humanoid {

/**
 * The order in which the ROS sims handed a cycle's mode and posture to their controller: the centroidal sim set the
 * mode first, the whole-body sim the posture first. Kept per formulation, so that each behaves exactly as before.
 */
enum class CycleInputOrder {
  kModeThenPosture,
  kPostureThenMode,
};

/**
 * RobotController over an MRT joint controller of either formulation; both have the same interface. Owns the
 * controller.
 */
template <typename Controller>
class MrtRobotController final : public RobotController {
 public:
  MrtRobotController(std::unique_ptr<Controller> controller, CycleInputOrder inputOrder)
      : controller_(std::move(controller)), inputOrder_(inputOrder) {}

  Controller& controller() { return *controller_; }
  const Controller& controller() const { return *controller_; }

  void prepareCycle(absl::string_view controlMode, const std::vector<scalar_t>& nominalJointPositions) override {
    if (inputOrder_ == CycleInputOrder::kModeThenPosture) {
      controller_->setControlMode(controlMode);
      controller_->setNominalJointPositions(nominalJointPositions);
    } else {
      controller_->setNominalJointPositions(nominalJointPositions);
      controller_->setControlMode(controlMode);
    }
  }

  void computeJointControlAction(const robot::model::RobotState& robotState, robot::model::RobotJointAction& jointAction) override {
    controller_->computeJointControlAction(/*time=*/0.0, robotState, jointAction);
  }

  std::optional<contact_flag_t> plannedContactFlags() const override {
    return controller_->getPlannedContactFlags(controller_->getCurrentObservation().time);
  }

  const contact_flag_t& measuredContactFlags() const override { return controller_->getMeasuredContactFlags(); }
  void requestMpcReset() override { controller_->requestMpcReset(); }
  void requestMpcResetAndHold() override { controller_->requestMpcResetAndHold(); }
  bool isMpcHealthy() const override { return controller_->isMpcHealthy(); }

  void setContactEstimator(std::shared_ptr<robot::model::ContactEstimator> contactEstimator) override {
    controller_->setContactEstimator(std::move(contactEstimator));
  }
  const ContactWrenchGate::Config& contactWrenchGateConfig() const override { return controller_->getContactWrenchGate().getConfig(); }
  void setContactWrenchGateConfig(const ContactWrenchGate::Config& config) override { controller_->setContactWrenchGateConfig(config); }

  absl::Status setPdGains(const mpc_config::JointPdGainsFile& gains) override { return controller_->setPdGains(gains); }
  void pollPdGainsFile() override { controller_->pollPdGainsFile(); }

  void setEventSink(ControllerEventSink* absl_nullable eventSink) override { controller_->setEventSink(eventSink); }
  void startMpc(const robot::model::RobotState& initialState) override { controller_->startMpcThread(initialState); }
  bool policyReady() override { return controller_->ready(); }
  const std::vector<std::string>& robotJointNames() const override { return controller_->getRobotJointNames(); }

 private:
  std::unique_ptr<Controller> controller_;
  const CycleInputOrder inputOrder_;
};

}  // namespace ocs2::humanoid
