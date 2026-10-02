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

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <memory>

#include "absl/status/statusor.h"

#include "humanoid_common_mpc_app/robot/MrtRobotController.h"
#include "humanoid_mpc_validation/closed_loop/ClosedLoopDriver.h"
#include "humanoid_wb_mpc/WBMpcInterface.h"
#include "humanoid_wb_mpc/mrt/WBMpcMrtJointController.h"

namespace ocs2::humanoid::validation {

/**
 * The whole-body MPC as its MPC node builds it (WBMpcNode::Create()): the interface and the SQP MPC with the motion
 * manager as its synchronized module. And WBMpcMrtJointController as its robot binary builds it (WBMpcRobotMain.cpp),
 * with the task file's controller settings, behind MrtRobotController with the binary's CycleInputOrder.
 */
class WholeBodyClosedLoopDriver final : public ClosedLoopDriver {
 public:
  static absl::StatusOr<std::unique_ptr<WholeBodyClosedLoopDriver>> create(const RobotConfiguration& configuration,
                                                                           const ClosedLoopDriverOptions& options);

  ~WholeBodyClosedLoopDriver() override;

  bool isEnteringMpc() const override { return controller_->controller().isHolding(); }
  const SystemObservation& currentObservation() const override { return controller_->controller().getCurrentObservation(); }
  const vector_t& latestPolicyInput() const override { return controller_->controller().getLatestPolicyInput(); }

 private:
  WholeBodyClosedLoopDriver() = default;

  absl::Status initialize(const RobotConfiguration& configuration, const ClosedLoopDriverOptions& options);

  std::unique_ptr<WBMpcInterface> interface_;
  MrtRobotController<WBMpcMrtJointController>* controller_ = nullptr;  ///< robotController_, typed
};

}  // namespace ocs2::humanoid::validation
