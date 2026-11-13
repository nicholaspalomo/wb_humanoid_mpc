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
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "urdf_model/model.h"
#include "urdf_world/types.h"

namespace robot::model {

/**
 * Turns every CONTINUOUS joint of `model` into a REVOLUTE joint whose position
 * limits are -/+infinity, keeping its effort and velocity limits (a joint
 * without a <limit> element gets infinite ones). Pinocchio then builds the
 * joint with one coordinate, the angle (nq = nv = 1), instead of the (cos, sin)
 * pair of its unbounded revolute joint, and with no position bound: a wheel
 * keeps one entry in the MPC state and may turn without end.
 *
 * Every Pinocchio build of a URDF in this repository goes through it, so that
 * a continuous joint has the same coordinates in every model. A URDF without a
 * continuous joint is left as it is. The joints are changed in place: a copy of
 * the ModelInterface shares them.
 *
 * @return the names of the converted joints, in name order.
 */
std::vector<std::string> makeContinuousJointsUnboundedRevolute(
    urdf::ModelInterface& model);

/**
 * Parses the URDF file `urdfPath` and applies
 * makeContinuousJointsUnboundedRevolute() to it: the tree to build a Pinocchio
 * model from (pinocchio::urdf::buildModel()).
 *
 * @return InvalidArgument naming the file when it cannot be read or parsed.
 */
absl::StatusOr<urdf::ModelInterfaceSharedPtr> parseUrdfFileForPinocchio(
    absl::string_view urdfPath);

}  // namespace robot::model
