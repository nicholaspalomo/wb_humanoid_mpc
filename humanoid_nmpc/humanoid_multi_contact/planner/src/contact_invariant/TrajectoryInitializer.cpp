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

#include "humanoid_multi_contact/planner/contact_invariant/TrajectoryInitializer.h"

#include <algorithm>
#include <memory>
#include <utility>

#include "absl/log/absl_check.h"
#include "ocs2_core/misc/LinearInterpolation.h"
#include "ocs2_oc/oc_data/StateTrajectoryInterpolation.h"

namespace ocs2::humanoid::multi_contact {

TrajectoryInitializer::TrajectoryInitializer(
    GuessTrajectory guess, std::shared_ptr<const StateManifold> manifold)
    : guess_(std::move(guess)), manifold_(std::move(manifold)) {
  ABSL_CHECK(!guess_.times.empty());
  ABSL_CHECK_EQ(guess_.times.size(), guess_.states.size());
  ABSL_CHECK_EQ(guess_.times.size(), guess_.inputs.size());
}

void TrajectoryInitializer::compute(scalar_t time,
                                    const vector_t& /*state*/,
                                    scalar_t nextTime,
                                    vector_t& input,
                                    vector_t& nextState) {
  const scalar_t start = guess_.times.front();
  const scalar_t end = guess_.times.back();
  input = LinearInterpolation::interpolate(std::clamp(time, start, end),
                                           guess_.times, guess_.inputs);
  nextState = interpolateStateTrajectory(manifold_.get(),
                                         std::clamp(nextTime, start, end),
                                         guess_.times, guess_.states);
}

}  // namespace ocs2::humanoid::multi_contact
