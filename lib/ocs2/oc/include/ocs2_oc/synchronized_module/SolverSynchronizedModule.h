/******************************************************************************
Copyright (c) 2020, Farbod Farshidian. All rights reserved.

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

#include <ocs2_core/Types.h>

#include "ocs2_oc/oc_data/PrimalSolution.h"
#include "ocs2_oc/synchronized_module/ReferenceManagerInterface.h"

namespace ocs2 {

/**
 * A Solver synchronized module is updated once before and once after a problem is solved.
 */
class SolverSynchronizedModule {
 public:
  /**
   * Default destructor
   */
  virtual ~SolverSynchronizedModule() = default;

  /**
   * Method called right before the solver runs
   *
   * @param initTime : start time of the MPC horizon
   * @param finalTime : Final time of the MPC horizon
   * @param initState : State at the start of the MPC horizon
   * @param referenceManager : The ReferenceManager which manages both ModeSchedule and TargetTrajectories.
   */
  virtual void preSolverRun(scalar_t initTime,
                            scalar_t finalTime,
                            const vector_t& initState,
                            const ReferenceManagerInterface& referenceManager) = 0;

  /**
   * Method called right after the solver runs
   *
   * @param primalSolution : primalSolution
   */
  virtual void postSolverRun(const PrimalSolution& primalSolution) = 0;

  /**
   * Returns the module to the state it had right after construction, keeping only its configuration: whatever it
   * carries from one solve to the next (a command ramp, a pending plan, a warm start) is dropped, so that the next
   * preSolverRun() behaves exactly as the first one of a freshly constructed module would.
   *
   * Called by MPC_BASE::reset() on the thread that runs the solver, between two solves, after the reference manager's
   * reset() and before the solver's. A module that owns another thread must make sure nothing that thread computed from
   * the state before the reset reaches the reference manager after it.
   */
  virtual void reset() {}
};

}  // namespace ocs2
