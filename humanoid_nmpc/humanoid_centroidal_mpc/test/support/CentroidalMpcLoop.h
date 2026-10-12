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
#include <optional>

#include "absl/status/statusor.h"
#include "ocs2_mpc/MPC_MRT_Interface.h"
#include "ocs2_mpc/SystemObservation.h"
#include "ocs2_oc/oc_data/PerformanceIndex.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"
#include "ocs2_sqp/SqpMpc.h"

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_centroidal_mpc/command/CentroidalMpcTargetTrajectoriesCalculator.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h"
#include "support/ShippedCentroidalRobot.h"

namespace ocs2::humanoid {

/** What CentroidalMpcLoop::Create() changes in the shipped task file. Passive data. */
struct CentroidalMpcLoopOptions {
  /**
   * The yaw weight of the base orientation in state_weights and final_state_weights, when set. Every shipped
   * configuration weighs no yaw and turns its base only through its feet, which does not follow a turn in place (design
   * README section 6, risk 4).
   */
  std::optional<scalar_t> baseYawWeight;
};

/** What one step of CentroidalMpcLoop solved and how the plant moved. Passive data. */
struct CentroidalMpcLoopStep {
  /** The solve's performance index (SolverBase::getPerformanceIndeces()). */
  PerformanceIndex performance;
  /**
   * [rad] The rotation part of the initial state gap the solve closed: the largest rotation of the difference, on the
   * state manifold, from the last policy's planned state at the solve's time to the observation; 0 for the first solve.
   */
  scalar_t initialGapRotation = 0.0;
  /**
   * The feedback correction of the policy where the period starts, the observation after the push: |u(t, x) - u(t,
   * x_bar(t))|_inf, x_bar the planned state; 0 for a feedforward policy and without a push (the solve starts its plan at
   * the observation).
   */
  scalar_t feedbackCorrection = 0.0;
  /** |u(t, x_bar(t))|_inf at the same times: the scale of the correction. */
  scalar_t feedforwardMagnitude = 0.0;
  /** max | |xi| - 1 | over the policy's nodes and the plant's state after the step. */
  scalar_t quaternionNormError = 0.0;
};

/**
 * The centroidal MPC of a shipped configuration in a lockstep loop, without a simulator: the interface's problem solved
 * by its SQP MPC with the reference manager, the target calculator and the procedural motion manager registered as the
 * MPC node registers them, every solve's observation handed to the motion manager before it (the command neither
 * filtered nor ramped, its limits 1, so a command is applied as given), an
 * MRT on the state manifold, and a plant that is the projected rollout of the policy over one MPC period
 * (MRT_BASE::rolloutPolicy() on the interface's rollout, which projects the quaternion back onto the unit sphere). Every
 * step solves once from the plant's state and then moves the plant. Builds the CppAD libraries of the configuration on a
 * cold cache. Not thread-safe.
 */
class CentroidalMpcLoop {
 public:
  /**
   * The loop of `robot`'s shipped files with `options`' changes, the robot standing at the task file's initial state at
   * time 0.
   */
  static absl::StatusOr<std::unique_ptr<CentroidalMpcLoop>> Create(ShippedCentroidalRobot robot, const CentroidalMpcLoopOptions& options);

  CentroidalMpcLoop(const CentroidalMpcLoop&) = delete;
  CentroidalMpcLoop& operator=(const CentroidalMpcLoop&) = delete;
  CentroidalMpcLoop(CentroidalMpcLoop&&) = delete;
  CentroidalMpcLoop& operator=(CentroidalMpcLoop&&) = delete;
  ~CentroidalMpcLoop() = default;

  /** The operator's command, applied as given: [m/s] forward and to the left in the heading frame, [rad/s] about z. */
  void command(scalar_t forward, scalar_t lateral, scalar_t yawRate);

  /**
   * Pushes the plant in the next step(), after its solve: adds `momentumChange` to the normalized centroidal momentum
   * [v_com, L / m] the period starts from, as an impulse would, so that the plant leaves the plan and the policy's
   * feedback acts over the period.
   */
  void push(const vector6_t& momentumChange) { pendingPush_ = momentumChange; }

  /** One solve from the plant's state, then one MPC period of the plant; the solve's status. */
  absl::StatusOr<CentroidalMpcLoopStep> step();

  CentroidalMpcInterface& interface() { return *interface_; }
  const ProceduralMpcMotionManager& motionManager() const { return *motionManager_; }
  const SystemObservation& observation() const { return observation_; }
  /** Whether a solve has given a policy yet; policy() needs one. */
  bool hasPolicy() const { return mrt_->initialPolicyReceived(); }
  const PrimalSolution& policy() const { return mrt_->getPolicy(); }
  /** [s] The plant's step: the MPC's period, 1 / mpc_desired_frequency. */
  scalar_t period() const { return period_; }

 private:
  CentroidalMpcLoop() = default;

  std::unique_ptr<CentroidalMpcInterface> interface_;
  std::unique_ptr<SqpMpc> mpc_;
  std::unique_ptr<CentroidalMpcTargetTrajectoriesCalculator> calculator_;
  std::shared_ptr<ProceduralMpcMotionManager> motionManager_;
  std::unique_ptr<MPC_MRT_Interface> mrt_;
  SystemObservation observation_;
  scalar_t period_ = 0.0;
  /** The push of the next period (push()); zero: none. */
  vector6_t pendingPush_ = vector6_t::Zero();
};

}  // namespace ocs2::humanoid
