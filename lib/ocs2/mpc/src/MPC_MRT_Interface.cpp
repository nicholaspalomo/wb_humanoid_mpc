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

#include "ocs2_mpc/MPC_MRT_Interface.h"

#include <ocs2_core/control/FeedforwardController.h>
#include <ocs2_core/control/LinearController.h>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace ocs2 {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
MPC_MRT_Interface::MPC_MRT_Interface(MPC_BASE& mpc) : mpc_(mpc) {
  mpcTimer_.reset();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void MPC_MRT_Interface::resetMpcNode(const TargetTrajectories& initTargetTrajectories) {
  // A policy solved before the reset is dropped from the buffer; the one in use is replaced by the thread that calls
  // updatePolicy(), which isActivePolicyCurrent() lets tell the two apart.
  discardBufferedPolicy();
  mpc_.reset();
  mpc_.getSolverPtr()->getReferenceManager().setTargetTrajectories(initTargetTrajectories);
  mpcTimer_.reset();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void MPC_MRT_Interface::resetMpcSolver(const TargetTrajectories& initTargetTrajectories) {
  discardBufferedPolicy();
  mpc_.resetSolver();
  mpc_.getSolverPtr()->getReferenceManager().setTargetTrajectories(initTargetTrajectories);
  mpcTimer_.reset();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void MPC_MRT_Interface::setCurrentObservation(const SystemObservation& currentObservation) {
  std::lock_guard<std::mutex> lock(observationMutex_);
  currentObservation_ = currentObservation;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

SystemObservation MPC_MRT_Interface::getCurrentObservation() {
  std::lock_guard<std::mutex> lock(observationMutex_);
  return currentObservation_;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
ReferenceManagerInterface& MPC_MRT_Interface::getReferenceManager() {
  return mpc_.getSolverPtr()->getReferenceManager();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
const ReferenceManagerInterface& MPC_MRT_Interface::getReferenceManager() const {
  return mpc_.getSolverPtr()->getReferenceManager();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
absl::Status MPC_MRT_Interface::advanceMpc() {
  // measure the delay in running MPC
  mpcTimer_.startTimer();

  SystemObservation currentObservation;
  {
    std::lock_guard<std::mutex> lock(observationMutex_);
    currentObservation = currentObservation_;
  }

  bool controllerIsUpdated;

  try {
    controllerIsUpdated = mpc_.run(currentObservation.time, currentObservation.state, currentObservation.mode);
  } catch (const std::exception& e) {
    // The state and the targets of the solve that crashed are dumped once per run of failures, not once per failure:
    // a caller that retries at its solve rate would otherwise flood the log with them. What to report, and how often,
    // is the caller's decision; the returned status carries the reason.
    if (consecutiveCrashes_++ == 0) {
      const vector_t& state = currentObservation.state;
      LOG(WARNING) << "MPC solver crashed at t = " << currentObservation.time << ": " << e.what()
                   << "\nState: " << absl::StrJoin(state.data(), state.data() + state.size(), " ")
                   << "\nDesired trajectories: " << mpc_.getSolverPtr()->getReferenceManager().getTargetTrajectories();
    }
    return absl::InternalError(absl::StrCat("MPC solver crashed at t = ", currentObservation.time, ": ", e.what()));
  }
  consecutiveCrashes_ = 0;

  if (!controllerIsUpdated) {
    // MPC_BASE::run() refuses to run once the observation time has passed the end of the previous solution (a clock
    // that jumped forward, or a solver thread starved for a whole horizon). Nothing is solved and nothing reaches the
    // buffer until the MPC is reset, so this is a failure for the caller to act on, not a quiet success.
    return absl::FailedPreconditionError(absl::StrCat("MPC not run: the observation time ", currentObservation.time,
                                                      " is past the final time ", mpc_.getSolverPtr()->getFinalTime(),
                                                      " of the previous solution; the MPC has to be reset."));
  }
  copyToBuffer(currentObservation);

  // measure the delay for sending ROS messages
  mpcTimer_.endTimer();

  // check MPC delay and solution window compatibility
  scalar_t timeWindow = mpc_.settings().solutionTimeWindow_;
  if (mpc_.settings().solutionTimeWindow_ < 0) {
    timeWindow = mpc_.getSolverPtr()->getFinalTime() - currentObservation.time;
  }
  if (timeWindow < 2.0 * mpcTimer_.getAverageInMilliseconds() * 1e-3) {
    LOG(WARNING) << "The solution time window might be shorter than the MPC delay!";
  }

  // measure the delay
  if (mpc_.settings().debugPrint_) {
    LOG(INFO) << "MPC_MRT Benchmarking — Max: " << mpcTimer_.getMaxIntervalInMilliseconds()
              << "ms, Avg: " << mpcTimer_.getAverageInMilliseconds() << "ms, Latest: " << mpcTimer_.getLastIntervalInMilliseconds() << "ms";
  }
  return absl::OkStatus();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void MPC_MRT_Interface::copyToBuffer(const SystemObservation& mpcInitObservation) {
  // policy
  auto primalSolutionPtr = std::make_unique<PrimalSolution>();
  const scalar_t startTime = mpcInitObservation.time;
  const scalar_t finalTime =
      (mpc_.settings().solutionTimeWindow_ < 0) ? mpc_.getSolverPtr()->getFinalTime() : startTime + mpc_.settings().solutionTimeWindow_;
  mpc_.getSolverPtr()->getPrimalSolution(finalTime, primalSolutionPtr.get());

  // command
  auto commandPtr = std::make_unique<CommandData>();
  commandPtr->mpcInitObservation_ = mpcInitObservation;
  commandPtr->mpcTargetTrajectories_ = mpc_.getSolverPtr()->getReferenceManager().getTargetTrajectories();

  // performance indices
  auto performanceIndicesPtr = std::make_unique<PerformanceIndex>();
  *performanceIndicesPtr = mpc_.getSolverPtr()->getPerformanceIndeces();

  this->moveToBuffer(std::move(commandPtr), std::move(primalSolutionPtr), std::move(performanceIndicesPtr));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
matrix_t MPC_MRT_Interface::getLinearFeedbackGain(scalar_t time) {
  auto controller = dynamic_cast<LinearController*>(this->getPolicy().controllerPtr_.get());
  if (controller == nullptr) {
    throw std::runtime_error("[MPC_MRT_Interface::getLinearFeedbackGain] Feedback gains only available with linear controller!");
  }
  matrix_t K;
  controller->getFeedbackGain(time, K);
  return K;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
ScalarFunctionQuadraticApproximation MPC_MRT_Interface::getValueFunction(scalar_t time, const vector_t& state) const {
  return mpc_.getSolverPtr()->getValueFunction(time, state);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
vector_t MPC_MRT_Interface::getStateInputEqualityConstraintLagrangian(scalar_t time, const vector_t& state) const {
  return mpc_.getSolverPtr()->getStateInputEqualityConstraintLagrangian(time, state);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
MultiplierCollection MPC_MRT_Interface::getIntermediateDualSolution(scalar_t time) const {
  return mpc_.getSolverPtr()->getIntermediateDualSolution(time);
}

}  // namespace ocs2
