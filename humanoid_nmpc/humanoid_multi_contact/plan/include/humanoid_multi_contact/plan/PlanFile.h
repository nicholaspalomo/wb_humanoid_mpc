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

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_multi_contact/contact_set/contact_set.nproto.h"
#include "humanoid_multi_contact/plan/multi_contact_plan.nproto.h"
#include "humanoid_multi_contact/terrain/terrain_scene.nproto.h"

/*
 * Reading and writing multi-contact plans (design section 7.1). A plan names
 * the contact set and the terrain scene it was made for by their hashes, and a
 * plan whose hashes do not match the ones loaded is refused: its knots' contact
 * bits and contact positions would mean something else.
 *
 * The hash is the 64-bit FNV-1a hash of the deterministic binary serialization
 * of the message rebuilt from its struct (ToProto() of FromProto()), so that a
 * field written out at its default and a field left out hash alike, and a
 * comment or the order of the keys in the textproto changes nothing.
 */

namespace ocs2::humanoid::multi_contact {

/** The suffix of a plan's binary file, and that of its textproto summary,
 * which replaces it. */
// LINT.IfChange(plan_binary_suffix)
inline constexpr char kPlanBinarySuffix[] = ".binpb";
// LINT.ThenChange(//humanoid_nmpc/humanoid_multi_contact/app/python/multi_contact_playback/plan_playback.py:plan_binary_suffix)
inline constexpr char kPlanSummarySuffix[] = ".summary.textproto";

/** The hash of a contact set (a mask of bits, hence unsigned). */
uint64_t contactSetHash(const config::ContactSet& contactSet);

/** The hash of a terrain scene. */
uint64_t terrainSceneHash(const config::TerrainScene& scene);

/** Sets the hashes of `plan` to those of `contactSet` and `scene`. */
void stampPlanHashes(const config::ContactSet& contactSet,
                     const config::TerrainScene& scene,
                     config::MultiContactPlan& plan);

/** FailedPrecondition, with both hashes, when the hashes of `plan` are not
 * those of `contactSet` and `scene`. */
absl::Status checkPlanHashes(const config::MultiContactPlan& plan,
                             const config::ContactSet& contactSet,
                             const config::TerrainScene& scene);

/**
 * Reads the plan at `path`: a binary proto when the path ends in
 * kPlanBinarySuffix, otherwise a textproto (whose header names
 * humanoid_multi_contact.MultiContactPlan), as hand-written test plans are.
 * Then checks its hashes against `contactSet` and `scene` (checkPlanHashes()).
 * NotFound for a missing file, InvalidArgument for one that does not parse.
 */
absl::StatusOr<config::MultiContactPlan> loadPlanFile(
    absl::string_view path,
    const config::ContactSet& contactSet,
    const config::TerrainScene& scene);

/** The path of the textproto summary of the binary plan at `binaryPath`. */
std::string planSummaryPath(absl::string_view binaryPath);

/**
 * Writes `plan` as a binary proto to `binaryPath`, which must end in
 * kPlanBinarySuffix, and its summary, the plan without its knots (the hashes,
 * phases and certifier report), as a textproto beside it (planSummaryPath()).
 * Creates the directory. InvalidArgument for another suffix, Internal when a
 * file cannot be written.
 */
absl::Status savePlanFile(const config::MultiContactPlan& plan,
                          absl::string_view binaryPath);

}  // namespace ocs2::humanoid::multi_contact
