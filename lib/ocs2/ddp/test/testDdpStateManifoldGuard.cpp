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

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>

#include <ocs2_core/initialization/DefaultInitializer.h>
#include <ocs2_oc/rollout/TimeTriggeredRollout.h>
#include <ocs2_oc/test/RigidBodyAttitudeDynamics.h>

#include "ocs2_ddp/SLQ.h"

namespace ocs2 {
namespace {

// DDP rolls out, stores a value function and builds a LinearController on the ambient state: a state manifold would be
// silently ignored, so it is refused.
TEST(DdpStateManifoldGuard, ConstructorThrowsOnAStateManifold) {
  const manifold_test::RigidBodyAttitudeDynamics dynamics(/*withPosition=*/false);
  OptimalControlProblem problem;
  problem.dynamicsPtr.reset(dynamics.clone());
  problem.stateManifoldPtr = dynamics.getStateManifold();
  const TimeTriggeredRollout rollout(dynamics);
  const DefaultInitializer initializer(3);
  EXPECT_THROW({ SLQ slq(ddp::Settings(), rollout, problem, initializer); }, std::runtime_error);

  problem.stateManifoldPtr = nullptr;
  EXPECT_NO_THROW({ SLQ slq(ddp::Settings(), rollout, problem, initializer); });
}

}  // namespace
}  // namespace ocs2
