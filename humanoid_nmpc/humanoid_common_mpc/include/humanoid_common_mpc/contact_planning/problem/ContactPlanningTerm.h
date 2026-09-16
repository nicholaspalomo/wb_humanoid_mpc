/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/problem/Layout.h"

namespace ocs2::humanoid {

/**
 * Base of every term the contact planner's problem is assembled from (ContactPlanningProblem): model blocks, costs,
 * constraints, logic rules, assignment costs, and, around the problem, search stages and execution rules.
 *
 * A term has a canonical name (ContactPlanningFormulation.h), reads its parameter block from the configuration in
 * configure() (the hot-reload path), resolves the variable indices it uses once in bind(), and describes its
 * mathematics in one line for the start-up print.
 */
class ContactPlanningTerm {
 public:
  virtual ~ContactPlanningTerm() = default;

  /** One line of math with the current parameter values, printed at start-up and after a structural reload. */
  virtual std::string describe() const = 0;
  /** Model blocks this term needs in the layout; the problem rejects a formulation that lists the term without them. */
  virtual std::vector<std::string> requiredBlocks() const { return {}; }
  /** Resolves the variable indices the term uses. Called once after the layout is final and after every re-assembly. */
  virtual void bind(const Layout& /*layout*/) {}
  /** Reads the term's parameter block. Validates it and throws std::invalid_argument on bad values. */
  virtual void configure(const ContactPlanningConfig& config) = 0;
};

/** Nodes of the horizon a term acts on. Node N (the terminal one) has no inputs, dynamics or ZMP. */
enum class NodeSet { RUNNING, TERMINAL, ALL, LAST_RUNNING };
bool nodeSetContains(NodeSet set, int node, int numNodes);
std::string nodeSetName(NodeSet set);

}  // namespace ocs2::humanoid
