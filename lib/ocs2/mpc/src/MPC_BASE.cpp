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

#include <algorithm>

#include <ocs2_mpc/MPC_BASE.h>

#include "absl/log/log.h"

namespace ocs2 {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
MPC_BASE::MPC_BASE(mpc::Settings mpcSettings) : mpcSettings_(std::move(mpcSettings)) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void MPC_BASE::reset() {
  // The references and the synchronized modules first: they carry the schedules, the latched measurements and the
  // command state that the solver's own reset() does not know about, and that a reset has to clear as surely as the
  // warm start (see ReferenceManagerInterface::reset()).
  getSolverPtr()->resetReferenceManagerAndModules();
  resetSolver();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void MPC_BASE::resetSolver() {
  initRun_ = true;
  mpcTimer_.reset();
  getSolverPtr()->reset();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
bool MPC_BASE::run(scalar_t currentTime, const vector_t& currentState, size_t currentMode) {
  // check if the current time exceeds the solver final limit
  if (!initRun_ && currentTime >= getSolverPtr()->getFinalTime()) {
    LOG_EVERY_N_SEC(WARNING, 1.0) << "The MPC time horizon is smaller than the MPC starting time: currentTime " << currentTime
                                  << " is past the final time " << getSolverPtr()->getFinalTime()
                                  << " of the previous solution. The MPC has to be reset.";
    return false;
  }

  const scalar_t finalTime = currentTime + mpcSettings_.timeHorizon_;

  // display
  if (mpcSettings_.debugPrint_) {
    LOG(INFO) << "\n#####################################################"
              << "\n#####################################################"
              << "\n#####################################################"
              << "\n### MPC is called at time:  " << currentTime << " [s]."
              << "\n### MPC final Time:         " << finalTime << " [s]."
              << "\n### MPC time horizon:       " << mpcSettings_.timeHorizon_ << " [s].";
    mpcTimer_.startTimer();
  }

  // calculate the MPC policy
  calculateController(currentTime, currentState, currentMode, finalTime);

  // set initRun flag to false
  initRun_ = false;

  // display
  if (mpcSettings_.debugPrint_) {
    mpcTimer_.endTimer();
    LOG(INFO) << "\n### MPC Benchmarking"
              << "\n###   Maximum : " << mpcTimer_.getMaxIntervalInMilliseconds() << "[ms]."
              << "\n###   Average : " << mpcTimer_.getAverageInMilliseconds() << "[ms]."
              << "\n###   Latest  : " << mpcTimer_.getLastIntervalInMilliseconds() << "[ms].";
  }

  return true;
}

}  // namespace ocs2
