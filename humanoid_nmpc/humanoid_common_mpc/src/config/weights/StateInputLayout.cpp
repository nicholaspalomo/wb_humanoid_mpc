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

#include "humanoid_common_mpc/config/weights/StateInputLayout.h"

#include <cstddef>

#include "ocs2_core/Dimensions.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"

namespace ocs2::humanoid {

// LINT.IfChange(state_input_dimensions)
StateInputLayout stateInputLayout(const ModelSettings& settings, StateInputLayout::Mpc mpc) {
  StateInputLayout layout;
  layout.mpc = mpc;
  layout.jointNames = settings.mpcModelJointNames;
  layout.fixedJointNames = settings.fixedJointNames;
  layout.contactNames = settings.contactNames;
  return layout;
}

TuningLayout tuningLayoutOf(const StateInputLayout& layout) {
  const size_t joints = layout.jointNames.size();
  switch (layout.mpc) {
    case StateInputLayout::Mpc::kCentroidal:
      // The normalized centroidal momentum, the base pose, the joint positions.
      return TuningLayout{
          .name = "centroidal",
          .segments =
              {
                  {.name = "normalized_linear_momentum", .kind = TuningSegmentKind::kEuclidean, .tuningDim = kPositionDim},
                  {.name = "normalized_angular_momentum", .kind = TuningSegmentKind::kEuclidean, .tuningDim = kRotationTangentDim},
                  {.name = "base_position", .kind = TuningSegmentKind::kEuclidean, .tuningDim = kPositionDim},
                  {.name = "base_orientation", .kind = TuningSegmentKind::kEulerZyxOrientation, .tuningDim = kRotationTuningDim},
                  {.name = "joint_positions", .kind = TuningSegmentKind::kEuclidean, .tuningDim = joints},
              },
      };
    case StateInputLayout::Mpc::kWholeBody:
      // The generalized coordinates and their velocities.
      return TuningLayout{
          .name = "whole-body",
          .segments =
              {
                  {.name = "base_position", .kind = TuningSegmentKind::kEuclidean, .tuningDim = kPositionDim},
                  {.name = "base_orientation", .kind = TuningSegmentKind::kEulerZyxOrientation, .tuningDim = kRotationTuningDim},
                  {.name = "joint_positions", .kind = TuningSegmentKind::kEuclidean, .tuningDim = joints},
                  {.name = "base_linear_velocity", .kind = TuningSegmentKind::kEuclidean, .tuningDim = kPositionDim},
                  {.name = "base_angular_velocity", .kind = TuningSegmentKind::kZyxOrderedAngularVelocity, .tuningDim = kRotationTuningDim},
                  {.name = "joint_velocities", .kind = TuningSegmentKind::kEuclidean, .tuningDim = joints},
              },
      };
  }
  return TuningLayout{};
}

size_t tuningDimension(const StateInputLayout& layout) {
  return getTuningLayoutDim(tuningLayoutOf(layout));
}

size_t inputDimension(const StateInputLayout& layout) {
  // The wrench of each contact (kWrenchDim: its force, then its moment), then the joints.
  return kWrenchDim * layout.contactNames.size() + layout.jointNames.size();
}
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/config/weights/StateInputWeightsFromConfig.cpp:input_offsets)

}  // namespace ocs2::humanoid
