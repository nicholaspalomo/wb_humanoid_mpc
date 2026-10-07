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

#include "humanoid_mpc_ipc/SolutionTimeWindow.h"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "absl/base/nullability.h"
#include "ocs2_core/control/ControllerBase.h"
#include "ocs2_core/control/FeedforwardController.h"
#include "ocs2_core/control/LinearController.h"

#include "humanoid_mpc_ipc/PolicyControllers.h"

namespace ocs2::humanoid::ipc {
namespace {

template <typename Array>
void keepFirst(size_t length, Array* absl_nonnull array) {
  if (array->size() > length) {
    array->erase(array->begin() + static_cast<ptrdiff_t>(length), array->end());
  }
}

void trimController(scalar_t finalTime, ControllerBase* absl_nonnull controller) {
  if (FeedforwardController* absl_nullable feedforward = asFeedforwardController(*controller); feedforward != nullptr) {
    const size_t length = solutionWindowLength(feedforward->timeStamp_, finalTime);
    keepFirst(length, &feedforward->timeStamp_);
    keepFirst(length, &feedforward->uffArray_);
    return;
  }
  if (LinearController* absl_nullable linear = asLinearController(*controller); linear != nullptr) {
    const size_t length = solutionWindowLength(linear->timeStamp_, finalTime);
    keepFirst(length, &linear->timeStamp_);
    keepFirst(length, &linear->biasArray_);
    keepFirst(length, &linear->deltaBiasArray_);
    keepFirst(length, &linear->gainArray_);
  }
}

}  // namespace

size_t solutionWindowLength(const scalar_array_t& timeTrajectory, scalar_t finalTime) {
  const scalar_array_t::const_iterator firstLater = std::upper_bound(timeTrajectory.cbegin(), timeTrajectory.cend(), finalTime);
  size_t length = static_cast<size_t>(firstLater - timeTrajectory.cbegin());
  if (length < timeTrajectory.size()) {
    ++length;
  }
  return length;
}

void trimToSolutionWindow(scalar_t finalTime, PrimalSolution* absl_nonnull solution) {
  const size_t length = solutionWindowLength(solution->timeTrajectory_, finalTime);
  keepFirst(length, &solution->timeTrajectory_);
  keepFirst(length, &solution->stateTrajectory_);
  keepFirst(length, &solution->inputTrajectory_);
  // An event whose post-event node is cut off is cut off with it.
  const size_array_t::const_iterator firstCut =
      std::upper_bound(solution->postEventIndices_.cbegin(), solution->postEventIndices_.cend(), length == 0 ? 0 : length - 1);
  solution->postEventIndices_.erase(firstCut, solution->postEventIndices_.cend());
  if (solution->controllerPtr_ != nullptr) {
    trimController(finalTime, solution->controllerPtr_.get());
  }
}

}  // namespace ocs2::humanoid::ipc
