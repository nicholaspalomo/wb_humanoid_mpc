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
#include "ocs2_mpc/MPC_MRT_Interface.h"
#include "ocs2_mpc/SystemObservation.h"
#include "ocs2_oc/oc_data/PerformanceIndex.h"
#include "ocs2_sqp/SqpMpc.h"

#include "humanoid_centroidal_mpc/CentroidalMpcConfig.h"
#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_centroidal_mpc/multi_contact/PlannedMotionReferenceManager.h"
#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/** What one step of MultiContactPlanLoop solved. Passive data. */
struct MultiContactPlanLoopStep {
  /** The solve's performance index. */
  PerformanceIndex performance;
  /** [s] The plan time the solve tracked, and whether its clock held. */
  scalar_t planTime = 0.0;
  bool holding = false;
};

/**
 * The centroidal MPC tracking a multi-contact plan in a lockstep loop that
 * stands in for the dummy simulator, faster than real time: the interface of
 * `config` (contact_schedule_source: "multi_contact_plan") solved by its SQP
 * MPC with its planned-motion reference manager, an MRT on the state manifold,
 * and a plant that is the projected rollout of the policy over one MPC period
 * (the dummy simulator's plant, DummySimLoop). Every step solves once from the
 * plant's state and then moves the plant. The robot starts at the plan's
 * initial state. Builds the CppAD libraries of the configuration on a cold
 * cache. Not thread-safe.
 */
class MultiContactPlanLoop {
 public:
  /** The loop of `config` on the URDF at `urdfFile`. */
  static absl::StatusOr<std::unique_ptr<MultiContactPlanLoop>> Create(
      CentroidalMpcConfig config, const std::string& urdfFile);

  MultiContactPlanLoop(const MultiContactPlanLoop&) = delete;
  MultiContactPlanLoop& operator=(const MultiContactPlanLoop&) = delete;
  MultiContactPlanLoop(MultiContactPlanLoop&&) = delete;
  MultiContactPlanLoop& operator=(MultiContactPlanLoop&&) = delete;
  ~MultiContactPlanLoop() = default;

  /** One solve from the plant's state, then one MPC period of the plant; the
   * solve's status. */
  absl::StatusOr<MultiContactPlanLoopStep> step();

  CentroidalMpcInterface& interface() { return *interface_; }
  const PlannedMotionReferenceManager& referenceManager() const {
    return *interface_->getPlannedMotionReferenceManagerPtr();
  }
  const SystemObservation& observation() const { return observation_; }
  /** [s] The plant's step: the MPC's period. */
  scalar_t period() const { return period_; }

 private:
  MultiContactPlanLoop() = default;

  std::unique_ptr<CentroidalMpcInterface> interface_;
  std::unique_ptr<SqpMpc> mpc_;
  std::unique_ptr<MPC_MRT_Interface> mrt_;
  SystemObservation observation_;
  scalar_t period_ = 0.0;
};

}  // namespace ocs2::humanoid
