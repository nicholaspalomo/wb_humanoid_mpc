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

#include <atomic>
#include <memory>

#include <ocs2_core/Types.h>
#include <ocs2_oc/synchronized_module/SolverSynchronizedModule.h>

#include <humanoid_common_mpc/gait/GaitSchedule.h>
#include <humanoid_common_mpc/gait/ModeSequenceTemplate.h>
#include <humanoid_common_mpc/gait/MotionPhaseDefinition.h>

namespace ocs2::humanoid {

/**
 * Inserts a gait received from outside the solver (updateModeSequence) into the gait schedule before the next solve.
 *
 * The "a gait is waiting" flag is the one flag of this class and its subclasses, atomic, so that a subscriber thread
 * may set it while the solver thread consumes it in preSolverRun() and clears it in reset(). The received template
 * itself is not atomic: a subclass that receives on another thread serializes it with getReceivedGait(), for instance
 * with a mutex.
 */
class GaitScheduleUpdater : public SolverSynchronizedModule {
 public:
  explicit GaitScheduleUpdater(std::shared_ptr<GaitSchedule> gaitSchedulePtr);

  /** Inserts the gait received since the last solve, if any, through getReceivedGait() (updateGaitSchedule). */
  void preSolverRun(scalar_t initTime,
                    scalar_t finalTime,
                    const vector_t& currentState,
                    const ReferenceManagerInterface& referenceManager) override;

  void postSolverRun(const PrimalSolution&) override {};

  /** Drops a gait received but not yet inserted. The gait schedule itself is reset by the reference manager. */
  void reset() override;

  // Override this function in case you need to e.g. access the received gait in a mutex protected way.
  virtual ModeSequenceTemplate getReceivedGait() { return receivedGait_; }

  /** Stores `modeSequenceTemplate` and marks it for insertion by the next preSolverRun(). */
  void updateModeSequence(const ModeSequenceTemplate& modeSequenceTemplate);

  /**
   * Inserts `updatedGait` into the schedule at the first event after 70% of the horizon [initTime, finalTime] (or where
   * the left swing that ends there starts), tiled up to the absolute time initTime + 1.5 * (finalTime - initTime).
   */
  static void updateGaitSchedule(std::shared_ptr<GaitSchedule>& gaitSchedulePtr,
                                 const ModeSequenceTemplate& updatedGait,
                                 scalar_t initTime,
                                 scalar_t finalTime);

 protected:
  std::shared_ptr<GaitSchedule> gaitSchedulePtr_;
  // Set by updateModeSequence() on any thread; consumed by preSolverRun() and cleared by reset() on the solver thread.
  std::atomic<bool> gaitUpdated_{false};
  ModeSequenceTemplate receivedGait_;
};

}  // namespace ocs2::humanoid
