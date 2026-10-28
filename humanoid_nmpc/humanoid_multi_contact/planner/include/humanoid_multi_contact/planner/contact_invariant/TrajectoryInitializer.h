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

#include "absl/base/nullability.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/initialization/Initializer.h"
#include "ocs2_core/manifold/StateManifold.h"

namespace ocs2::humanoid::multi_contact {

/** A state and input trajectory over a time grid. Passive data. */
struct GuessTrajectory {
  scalar_array_t times;
  vector_array_t states;
  // One per time; the last may repeat the one before.
  vector_array_t inputs;
};

/**
 * The initializer of an offline solve over a whole motion: the input of a
 * guess trajectory at the node's time (interpolated linearly) and its state
 * at the next node's time (interpolated on the state manifold), whatever the
 * node's state. Times outside the guess clamp to its ends.
 *
 * A value type; the manifold is shared and immutable.
 */
class TrajectoryInitializer final : public Initializer {
 public:
  /** The guess `guess`, its states on `manifold` (nullptr: Euclidean). */
  TrajectoryInitializer(GuessTrajectory guess,
                        std::shared_ptr<const StateManifold> manifold);
  ~TrajectoryInitializer() override = default;
  TrajectoryInitializer* absl_nonnull clone() const override {
    return new TrajectoryInitializer(*this);
  }
  TrajectoryInitializer& operator=(const TrajectoryInitializer&) = delete;
  TrajectoryInitializer(TrajectoryInitializer&&) = delete;
  TrajectoryInitializer& operator=(TrajectoryInitializer&&) = delete;

  void compute(scalar_t time,
               const vector_t& state,
               scalar_t nextTime,
               vector_t& input,
               vector_t& nextState) override;

  const GuessTrajectory& guess() const { return guess_; }

 private:
  TrajectoryInitializer(const TrajectoryInitializer& other) = default;

  GuessTrajectory guess_;
  std::shared_ptr<const StateManifold> manifold_;
};

}  // namespace ocs2::humanoid::multi_contact
