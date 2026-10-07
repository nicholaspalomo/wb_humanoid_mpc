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

#include "absl/base/nullability.h"
#include "ocs2_core/control/ControllerBase.h"
#include "ocs2_core/control/FeedforwardController.h"
#include "ocs2_core/control/LinearController.h"

namespace ocs2::humanoid::ipc {

/**
 * Returns the controller of a policy as one of the two controller types the MPC link carries (MpcMessageConversions.h),
 * when it is of exactly that type, and null when it is of another type. The conversions, the solution time window and
 * the realtime evaluator tell the two apart through these; nothing else in the package asks for a controller's type.
 *
 * ControllerBase::getType() cannot tell them apart: a StateBasedLinearController reports the type of the controller it
 * wraps. Both classes are final, so the exact type is the one dynamic_cast would find. A call compares two
 * std::type_info objects, allocates nothing and does not log, so the realtime thread may make it. Each pair of overloads
 * is the same function for a const and a mutable controller, and returns `controller` itself or null.
 */
const FeedforwardController* absl_nullable asFeedforwardController(const ControllerBase& controller);
FeedforwardController* absl_nullable asFeedforwardController(ControllerBase& controller);
const LinearController* absl_nullable asLinearController(const ControllerBase& controller);
LinearController* absl_nullable asLinearController(ControllerBase& controller);

}  // namespace ocs2::humanoid::ipc
