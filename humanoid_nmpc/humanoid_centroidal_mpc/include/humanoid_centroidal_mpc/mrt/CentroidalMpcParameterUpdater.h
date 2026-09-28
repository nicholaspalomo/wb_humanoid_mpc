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

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <ocs2_mpc/MPC_BASE.h>

#include "absl/status/statusor.h"

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_centroidal_mpc/mrt/MpcParameterUpdaterModule.h"

namespace ocs2::humanoid {

/** A consumer of reference.yaml, called with the file's path whenever it changes (MpcParameterUpdaterModule). */
using ReferenceFileReloader = std::function<void(const std::string&)>;

/**
 * The parameter updater of a node that runs a centroidal MPC, wired to everything `interface` built that a hot reload
 * has to reach:
 *  - sized to the OCP input, with the basis-space transform of R when the contact inputs are basis vectors;
 *  - the reference manager (swing trajectories, the ground);
 *  - the contact planner module, under contactScheduleSource: contact_planner (contact_planning.yaml);
 *  - the locomotion-heuristic layer, whose coefficients are otherwise launch-time only;
 *  - `referenceFileReloaders`, the consumers of reference.yaml the node built (the target trajectories calculator and
 *    the procedural motion manager), so that the Command Limits tab of the remote control reaches the running node.
 *
 * Every node that runs the MPC builds its updater with this one function, so the nodes cannot drift apart in what they
 * register: they used to repeat the wiring by hand, and the MuJoCo simulator node never registered the reference.yaml
 * reloaders the SQP node did, so a command limit saved from the GUI reached one node and not the other.
 *
 * The node still subscribes the updater to the parameter topic (subscribe()) and registers it with the solver
 * (addSynchronizedModule), which need its ROS node and its solver. Returns the InvalidArgument of an updater that
 * cannot be sized to the interface's input.
 */
absl::StatusOr<std::shared_ptr<MpcParameterUpdaterModule>> makeCentroidalMpcParameterUpdater(
    MPC_BASE* mpc,
    const CentroidalMpcInterface& interface,
    const std::string& taskFile,
    const std::string& urdfFile,
    const std::string& referenceFile,
    std::vector<ReferenceFileReloader> referenceFileReloaders);

}  // namespace ocs2::humanoid
