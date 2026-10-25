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
#include <utility>

#include "absl/base/nullability.h"
#include "ocs2_mpc/CommandData.h"
#include "ocs2_mpc/MRT_BASE.h"
#include "ocs2_mpc/SystemObservation.h"
#include "ocs2_oc/oc_data/PerformanceIndex.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"

#include "humanoid_common_mpc/mrt/MpcLink.h"
#include "humanoid_common_mpc/mrt/MpcResetSupervisor.h"

namespace ocs2::humanoid::test_support {

/**
 * An MpcLink whose policies the test hands it (deliver()), with no solver
 * behind it: for tests of what an MRT joint controller does with a policy in
 * WB_MPC - one it evaluates, one it refuses - without a solver thread or a bus.
 * Not thread-safe: the test delivers between control cycles, on the thread that
 * runs them.
 */
class ScriptedPolicyMpcLink final : public MpcLink {
 public:
  /** A link whose resets the controller would serve from `resetTarget` (its own
   * reset target, kept for resetTargetOf()). */
  explicit ScriptedPolicyMpcLink(MpcLink::ResetTargetFunction resetTarget)
      : resetTarget_(std::move(resetTarget)) {}

  void start(const SystemObservation& /*initialObservation*/) override {}
  void stop() override {}
  MRT_BASE& mrt() override { return mrt_; }
  const MRT_BASE& mrt() const override { return mrt_; }

  /**
   * Serves the outstanding reset, if any (as a solver thread does before it
   * solves: the buffered policy dropped, the reset completed), then moves a
   * copy of `policy` to the MRT's buffer, which the controller's next
   * updatePolicy() takes into use as a policy solved after that reset.
   */
  void deliver(const PrimalSolution& policy) {
    const std::optional<MpcResetSupervisor::ResetTicket> ticket =
        resetSupervisor().takeResetRequest();
    if (ticket.has_value()) {
      mrt_.discardBufferedPolicy();
      resetSupervisor().completeReset(*ticket);
    }
    mrt_.deliver(policy);
  }

  /** The observation of the controller's last cycle (setCurrentObservation()).
   */
  const SystemObservation& lastObservation() const {
    return mrt_.lastObservation();
  }

  /** The target the controller's MPC would restart from after a reset at
   * `observation`: the controller's reset target. */
  TargetTrajectories resetTargetOf(const SystemObservation& observation) const {
    return resetTarget_(observation);
  }

  /**
   * A factory for the controllers' constructors that leaves the address of the
   * link it makes in `*link`; the controller owns the link, so `*link` lives as
   * long as the controller.
   */
  static MpcLinkFactory factory(
      ScriptedPolicyMpcLink* absl_nullable* absl_nonnull link) {
    return [link](MpcLink::ResetTargetFunction resetTarget) {
      std::unique_ptr<ScriptedPolicyMpcLink> created =
          std::make_unique<ScriptedPolicyMpcLink>(std::move(resetTarget));
      *link = created.get();
      return std::unique_ptr<MpcLink>(std::move(created));
    };
  }

 private:
  class ScriptedMrt final : public MRT_BASE {
   public:
    void resetMpcNode(
        const TargetTrajectories& /*initTargetTrajectories*/) override {}
    // An assignment of equal sizes after the first cycle: no allocation on the
    // control thread.
    void setCurrentObservation(const SystemObservation& observation) override {
      observation_ = observation;
    }
    void deliver(const PrimalSolution& policy) {
      moveToBuffer(std::make_unique<CommandData>(),
                   std::make_unique<PrimalSolution>(policy),
                   std::make_unique<PerformanceIndex>());
    }
    const SystemObservation& lastObservation() const { return observation_; }

   private:
    SystemObservation observation_;
  };

  MpcLink::ResetTargetFunction resetTarget_;
  ScriptedMrt mrt_;
};

}  // namespace ocs2::humanoid::test_support
