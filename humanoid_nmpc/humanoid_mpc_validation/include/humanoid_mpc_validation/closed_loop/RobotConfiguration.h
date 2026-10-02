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

namespace ocs2::humanoid::validation {

/** Which MPC a configuration runs. */
enum class MpcFormulation {
  kCentroidal,  ///< humanoid_centroidal_mpc
  kWholeBody,   ///< humanoid_wb_mpc
};

/** "centroidal" or "whole_body", as the metrics documents name a formulation. */
std::string formulationName(MpcFormulation formulation);

/**
 * One of the five configurations the closed-loop metrics and the solve benchmark run: the files the robot's launch files
 * pass to its robot binary and its MPC node (robot_models/<robot>/<package>/launch/robot.textproto and mpc.textproto),
 * as paths relative to the workspace root, which is the working directory of a Bazel test.
 */
struct RobotConfiguration {
  std::string name;  ///< the registry name, e.g. drc_atlas
  MpcFormulation formulation = MpcFormulation::kCentroidal;
  std::string taskFile;
  std::string referenceFile;
  std::string gaitFile;
  std::string urdfFile;
  std::string sceneFile;  ///< the MuJoCo scene
  std::string pdGainsFile;
  /// The walking scenario the robot survives, whose recorded states the solve benchmark replays by default: walk_0p5,
  /// or walk_0p3 for a robot that falls at 0.5 m/s (EngineAI SA01 and Unitree R1 in M0).
  std::string walkingScenario = "walk_0p5";

  /** Every file the run depends on, whose hashes go into the provenance: the files above and contact_planning.yaml. */
  std::vector<std::string> configurationFiles() const;
};

// LINT.IfChange(robot_configurations)
/** drc_atlas, engineai_sa01, unitree_g1, unitree_r1 (centroidal) and unitree_g1_wb (whole-body). */
const std::vector<RobotConfiguration>& robotConfigurations();
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_validation/BUILD.bazel:robot_configurations, //Makefile:closed_loop_targets)

/** The configuration named `name`; NotFound listing the names when there is none. */
absl::StatusOr<RobotConfiguration> findRobotConfiguration(absl::string_view name);

}  // namespace ocs2::humanoid::validation
