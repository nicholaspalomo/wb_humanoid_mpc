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

#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"

namespace ocs2::humanoid {

struct LocomotionHeuristicConfig;
struct LocomotionHeuristicModelParameters;

/**
 * One regularization heuristic of Gerardo Bledt's Regularized Predictive Control, "Regularized Predictive Control
 * Framework for Robust Dynamic Legged Locomotion", MIT 2020, Appendix C.
 *
 * RPC does not add a cost term. It SHAPES THE REFERENCE that a quadratic cost already regularizes against, so that the
 * optimizer starts from, and is biased towards, a solution a legged robot is known to want - while remaining free to
 * depart from it where the dynamics pay better (dissertation section 3.1). This class is that idea: a pure function
 * from the robot's situation to an OFFSET of one reference channel, summed with the offsets of every other listed
 * heuristic and added to the reference the controller would have used anyway.
 *
 * Three consequences of "offset" that are load-bearing and are relied on throughout the layer:
 *  - an empty list is EXACTLY the previous behaviour, bit for bit, because the sum of no offsets is zero and nothing
 *    is recomputed;
 *  - the order of the names in the task file is documentary, because addition is commutative (unlike the contact
 *    planner's `costs` list, whose order is the floating-point accumulation order);
 *  - a heuristic can be tuned to nothing by zeroing its coefficients, so a name may be listed while its block is
 *    still at zero and the layer is still a no-op. That is deliberate: it lets the list be turned on first and the
 *    numbers found second.
 *
 * Every implementation is IMMUTABLE after configure() and holds no mutable state, because the base-pose and wrench
 * evaluations run once per shooting node per SQP iteration on every worker thread of the solver. Nothing here may
 * perform forward kinematics for the same reason: everything that needs the model is derived once into
 * LocomotionHeuristicModelParameters, and everything that needs a measurement is latched once per solve by
 * SwitchedModelReferenceManager::captureMeasuredState() and handed over in the per-call context.
 */
class LocomotionHeuristic {
 public:
  virtual ~LocomotionHeuristic() = default;

  LocomotionHeuristic(const LocomotionHeuristic&) = delete;
  LocomotionHeuristic& operator=(const LocomotionHeuristic&) = delete;

  /** The canonical registry name of this heuristic, i.e. what the task file lists. */
  virtual absl::string_view name() const = 0;

  /**
   * Reads this heuristic's own parameter block out of `config` and the constants it needs out of `model`.
   *
   * This is also the hot-reload entry point: the parameter updater re-reads the task file and calls it again on the
   * live layer, so an implementation must overwrite every member it owns rather than accumulating into them.
   */
  virtual absl::Status configure(const LocomotionHeuristicConfig& config, const LocomotionHeuristicModelParameters& model) = 0;

  /**
   * One line naming the formula and the coefficients it is currently carrying, for the start-up banner.
   *
   * It exists because a heuristic whose coefficients are all zero is indistinguishable at run time from one that is
   * not listed at all, and the operator has to be able to tell those apart without reading the task file.
   */
  virtual std::string describe() const = 0;

 protected:
  LocomotionHeuristic() = default;
};

}  // namespace ocs2::humanoid
