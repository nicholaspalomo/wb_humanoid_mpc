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

#include "humanoid_common_mpc/common/StateManifoldFactory.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "ocs2_core/manifold/ProductStateManifold.h"
#include "ocs2_core/manifold/StateManifold.h"
#include "ocs2_core/manifold/StateManifoldSegment.h"

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"

namespace ocs2::humanoid {

absl::StatusOr<std::shared_ptr<const StateManifold>> stateManifoldOf(
    const TuningLayout& placedLayout, size_t stateDim) {
  size_t stateStart = 0;
  std::vector<StateManifoldSegment> segments;
  segments.reserve(placedLayout.segments.size());
  for (const TuningSegment& segment : placedLayout.segments) {
    if (segment.stateStart != stateStart) {
      return absl::InvalidArgumentError(absl::StrCat(
          "[stateManifoldOf] the block '", segment.name, "' of the ",
          describeTuningLayout(placedLayout), " starts at state row ",
          segment.stateStart, ", not at ", stateStart,
          ": the layout is not placed (placeTuningLayout())."));
    }
    stateStart += segment.stateDim;
    if (segment.kind != TuningSegmentKind::kEulerZyxOrientation) {
      if (segment.stateDim > 0)
        segments.push_back(StateManifoldSegment::euclidean(segment.stateDim));
      continue;
    }
    if (segment.stateDim != kQuaternionOrientationStateDim) {
      return absl::InvalidArgumentError(absl::StrCat(
          "[stateManifoldOf] the orientation block '", segment.name,
          "' of the ", describeTuningLayout(placedLayout), " is placed on ",
          segment.stateDim,
          " state rows; the state stores the base orientation as a quaternion, "
          "on ",
          kQuaternionOrientationStateDim, " rows."));
    }
    segments.push_back(StateManifoldSegment::unitQuaternion());
  }
  if (stateStart != stateDim) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[stateManifoldOf] the ", describeTuningLayout(placedLayout),
        " is placed on ", stateStart, " state rows, but the state has ",
        stateDim, "."));
  }
  ASSIGN_OR_RETURN(std::shared_ptr<const ProductStateManifold> manifold,
                   ProductStateManifold::create(std::move(segments)));
  return std::shared_ptr<const StateManifold>(std::move(manifold));
}

absl::StatusOr<std::shared_ptr<const StateManifold>> stateManifoldOf(
    const MpcRobotModelBase<scalar_t>& model) {
  return stateManifoldOf(model.getTuningLayout(), model.getStateDim());
}

}  // namespace ocs2::humanoid
