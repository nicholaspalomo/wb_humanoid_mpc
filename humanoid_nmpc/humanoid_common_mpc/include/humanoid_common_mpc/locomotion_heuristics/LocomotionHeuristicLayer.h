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
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "humanoid_common_mpc/locomotion_heuristics/BasePoseHeuristic.h"
#include "humanoid_common_mpc/locomotion_heuristics/FootholdHeuristic.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicConfig.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicModelParameters.h"
#include "humanoid_common_mpc/locomotion_heuristics/WrenchHeuristic.h"

namespace ocs2::humanoid {

/**
 * The assembled reference-shaping layer: the heuristics a task file listed, held by kind, summed on demand.
 *
 * This is the object SwitchedModelReferenceManager holds and the three seams call into. It is the only thing in the
 * subsystem the rest of the controller needs to know about.
 *
 * THREADING. The layer is built once on the construction thread and is thereafter read from every SQP worker thread,
 * concurrently and without a lock. That is sound because the heuristics are immutable between reconfigure() calls and
 * hold no mutable state, and because reconfigure() - the hot-reload path - is called from the solver thread's
 * pre-solve hook, which OCS2 runs before any worker exists for that solve. An empty layer is not merely cheap but
 * free: every accessor below short-circuits on an empty list before touching anything.
 */
class LocomotionHeuristicLayer {
 public:
  /**
   * Builds the layer from a validated configuration and the model constants, and configures every listed heuristic.
   *
   * Fails on a configuration validate() rejects, on a name no factory can build, and on the two combinations that
   * cannot work: a foothold heuristic under an online contact planner, and `centripetal_acceleration` under the
   * basis-vector input parameterization without a way to rotate its force. The warnings that are not errors are
   * emitted here, once, with LOG(WARNING).
   */
  static absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> Create(const LocomotionHeuristicConfig& config,
                                                                          const LocomotionHeuristicModelParameters& model,
                                                                          bool usesContactPlanning,
                                                                          bool usesContactBasisVectorInputs,
                                                                          bool verbose = false);

  /**
   * An empty layer, which is an exact no-op on every channel.
   *
   * Public so that Create() can build one, and because it is also the right thing for a caller that has no
   * configuration to hand - a test, or a reference manager constructed before the task file has been read. There is no
   * invalid state to protect: every accessor short-circuits on an empty list.
   */
  LocomotionHeuristicLayer() = default;

  LocomotionHeuristicLayer(const LocomotionHeuristicLayer&) = delete;
  LocomotionHeuristicLayer& operator=(const LocomotionHeuristicLayer&) = delete;

  /** True when no heuristic at all is listed, which is the shipped state of every robot. */
  bool empty() const { return basePose_.empty() && foothold_.empty() && wrench_.empty(); }
  bool basePoseEmpty() const { return basePose_.empty(); }
  bool footholdEmpty() const { return foothold_.empty(); }
  bool wrenchEmpty() const { return wrench_.empty(); }

  /**
   * True when a listed foothold heuristic re-anchors the landing target rather than nudging it, i.e. when
   * `hip_centered_stepping` is listed.
   *
   * The reference manager needs it because the anchor it would otherwise add the offsets to - a step width to the
   * side of the STANCE foot - is then the wrong starting point, and the two would be added together.
   */
  bool footholdMovesAnchor() const { return footholdMovesAnchor_; }

  /**
   * True when any listed wrench heuristic can produce a horizontal force, and therefore when the whole contact-force
   * reference has to be written through the state-aware, forward-kinematics path.
   */
  bool wrenchNeedsWorldFrame() const { return wrenchNeedsWorldFrame_; }

  /** The summed base-pose offset of every listed base-pose heuristic. Returns zero, cheaply, when none is listed. */
  BasePoseOffset basePoseOffset(const BasePoseHeuristicContext& context) const;

  /** The summed world-frame ground-plane foothold offset. Returns zero, cheaply, when none is listed. */
  vector2_t footholdOffset(const FootholdHeuristicContext& context) const;

  /** The summed world-frame contact-force offset for one stance foot. Returns zero, cheaply, when none is listed. */
  vector3_t wrenchOffset(const WrenchHeuristicContext& context, size_t contactIndex) const;

  /**
   * Re-reads every listed heuristic's coefficients from a freshly loaded configuration, IN PLACE.
   *
   * The formulation itself is not re-read: which heuristics are listed is a structural choice that decides how the
   * reference manager and the two input costs are wired, and changing it under a running solver would be a different
   * controller rather than a retuned one. A task file whose lists have changed is reported and otherwise ignored, so
   * that a slider drag on a coefficient does not silently pick up an edit to a list made an hour earlier.
   *
   * Called from the solver thread's pre-solve hook only; see the class comment.
   */
  absl::Status reconfigure(const LocomotionHeuristicConfig& config);

  /** The listed heuristics and their current coefficients, one per line, for the start-up banner. */
  std::string summary() const;

 private:
  LocomotionHeuristicFormulation formulation_;
  LocomotionHeuristicModelParameters model_;
  std::vector<std::unique_ptr<BasePoseHeuristic>> basePose_;
  std::vector<std::unique_ptr<FootholdHeuristic>> foothold_;
  std::vector<std::unique_ptr<WrenchHeuristic>> wrench_;
  bool footholdMovesAnchor_ = false;
  bool wrenchNeedsWorldFrame_ = false;
};

}  // namespace ocs2::humanoid
