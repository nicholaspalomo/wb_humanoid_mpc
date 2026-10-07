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

namespace ocs2::humanoid {

/**
 * How the MPC parameterizes the contact part of its input, selected by name with the top-level task-file field
 * `contact_input_parameterization` (kContactInputParameterizationKey).
 *
 *   wrench         (default) each contact's input is its six-dimensional wrench in the world frame, and the contact
 *                  wrench cone is a soft constraint of the formulation lists (`contact_wrench_cone`, or
 *                  `friction_force_cone` with `contact_moment_xy`).
 *   basis_vectors  each contact's input is a vector of scalings lambda of wrench-cone generators written in the local
 *                  contact frame (BasisInputsModelDecorator), so W = B lambda lies inside the cone whenever
 *                  lambda >= 0; that bound is a barrier built for every contact. Which generators make up B and how
 *                  the input cost is regularized are the task-file fields contacts.basis_generator_set and
 *                  contacts.basis_regularization. Centroidal MPC only: the whole-body MPC refuses it.
 *
 * The choice changes the input dimension of the problem, so it is structural: it takes effect at start-up only.
 * See humanoid_nmpc/docs/contact_basis_vectors/README.md.
 */
enum class ContactInputParameterization { kWrench, kBasisVectors };

// LINT.IfChange(contact_input_parameterization_names)
/** The top-level task-file field that names the parameterization. */
inline constexpr absl::string_view kContactInputParameterizationKey = "contact_input_parameterization";
inline constexpr absl::string_view kWrenchContactInputParameterization = "wrench";
inline constexpr absl::string_view kBasisVectorsContactInputParameterization = "basis_vectors";
/**
 * The boolean the field replaced (true meant basis_vectors). A task file that still carries it is refused by the strict
 * parser whatever its value (a retired field of TaskFile), so that a stale file cannot silently run the other
 * parameterization.
 */
inline constexpr absl::string_view kRetiredContactBasisVectorInputsKey = "useContactBasisVectorInputs";
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/common/ContactInputParameterization.cpp:contact_input_parameterization_registry, //humanoid_nmpc/docs/contact_basis_vectors/README.md:contact_input_parameterization_names, //robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto:contact_input_parameterization_config, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.textproto:contact_input_parameterization_config, //robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto:contact_input_parameterization_config, //robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/mpc/task.textproto:contact_input_parameterization_config, //tools/locomotion_heuristics/derive_parameters.py:contact_input_parameterization_name, //humanoid_nmpc/humanoid_mpc_config/task_file.proto:contact_input_parameterization)
// clang-format on

/** The parameterization of a task file that does not name one. It is what every robot ran before the field existed. */
inline constexpr ContactInputParameterization kDefaultContactInputParameterization = ContactInputParameterization::kWrench;

/** Every registered parameterization name, in registration order. */
std::vector<std::string> contactInputParameterizationNames();

/** The task-file name of a parameterization. */
absl::string_view contactInputParameterizationName(ContactInputParameterization parameterization);

/**
 * Resolves a name to its parameterization. An unknown name is an InvalidArgumentError that names
 * kContactInputParameterizationKey and lists every valid name.
 */
absl::StatusOr<ContactInputParameterization> contactInputParameterizationFromName(absl::string_view name);

}  // namespace ocs2::humanoid
