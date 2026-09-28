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

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid::test {

/**
 * The assembled optimal control problem of an interface, as the solver sees it, as labeled matrices compared exactly:
 * every collection's term names, and the LQ approximation of the whole problem (cost with its soft constraints,
 * dynamics, equality constraints) at points along the walking schedule of setWalkingReferences(), plus the terminal
 * approximation and the reference manager's desired state.
 *
 * It is how a formulation switch that became a name is shown to assemble exactly the problem the switch assembled: the
 * old switch's effects are re-applied by hand to a problem built without the name, and the two fingerprints compared.
 */
using Fingerprint = std::vector<std::pair<std::string, matrix_t>>;

/**
 * Puts `interface` on a walking schedule with a forward-moving target, the way the MPC loop does before a solve, and
 * points its problem at that target. The walking command is what exercises the procedural arm swing and every
 * schedule-gated term.
 */
void setWalkingReferences(CentroidalMpcInterface& interface);

/** The fingerprint of `interface`'s problem; it calls setWalkingReferences() first. */
Fingerprint fingerprintOf(CentroidalMpcInterface& interface);

/** Bit-for-bit equality of two fingerprints, naming the first entry that differs. */
::testing::AssertionResult identical(const Fingerprint& a, const Fingerprint& b);

}  // namespace ocs2::humanoid::test
