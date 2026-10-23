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

#include "humanoid_mpc_config/config_registries.nproto.h"

namespace ocs2::humanoid::config_registries {

/** Where config_registries.textproto is, from the repository root (and in the runfiles of a target that has it). */
inline constexpr char kConfigRegistriesFile[] = "humanoid_nmpc/humanoid_mpc_config/config_registries.textproto";

/**
 * Collects the names each registry of the stack accepts, read from the registries themselves (each one's own list of
 * names): the MPC's terms, contact inputs, contact schedule sources, basis vectors, centroidal models, stance
 * constraints and foot cost phases, the contact planner, its threading, its terms and its terminal DCM targets, the
 * locomotion heuristics, the robot process's telemetry sinks, contact estimators and whole-body feedforwards, and the
 * simulator's gantry holds, projectiles and visualizations. A registry is named as the schemas'
 * (humanoid_mpc_config.tuning) options name it; the registries are sorted by name, and each keeps its own order.
 * Allocates; touches no global state but the registries' own constants.
 */
mpc_config::ConfigRegistries collectConfigRegistries();

/**
 * Formats `registries` as config_registries.textproto: its schema header, a comment saying how it is written, and one
 * `registries { name: ... names: ... }` block per registry, one name per line.
 */
std::string configRegistriesText(const mpc_config::ConfigRegistries& registries);

}  // namespace ocs2::humanoid::config_registries
