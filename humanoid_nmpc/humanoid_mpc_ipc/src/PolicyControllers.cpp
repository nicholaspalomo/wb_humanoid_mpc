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

#include "humanoid_mpc_ipc/PolicyControllers.h"

#include <typeinfo>

#include "absl/base/nullability.h"
#include "ocs2_core/control/ControllerBase.h"
#include "ocs2_core/control/FeedforwardController.h"
#include "ocs2_core/control/LinearController.h"

namespace ocs2::humanoid::ipc {
namespace {

// `controller` as a `Concrete` when that is its exact type, else null; `Concrete` carries the const of `Base`.
template <typename Concrete, typename Base>
Concrete* absl_nullable asExactly(Base& controller) {
  // NOLINTNEXTLINE(rtti): OCS2's ControllerBase has no visitor and no exact type tag (PolicyControllers.h says why).
  return typeid(controller) == typeid(Concrete) ? static_cast<Concrete*>(&controller) : nullptr;
}

}  // namespace

const FeedforwardController* absl_nullable asFeedforwardController(const ControllerBase& controller) {
  return asExactly<const FeedforwardController>(controller);
}

FeedforwardController* absl_nullable asFeedforwardController(ControllerBase& controller) {
  return asExactly<FeedforwardController>(controller);
}

const LinearController* absl_nullable asLinearController(const ControllerBase& controller) {
  return asExactly<const LinearController>(controller);
}

LinearController* absl_nullable asLinearController(ControllerBase& controller) {
  return asExactly<LinearController>(controller);
}

}  // namespace ocs2::humanoid::ipc
