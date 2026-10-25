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

#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_state_estimator/KalmanBaseEstimator.h"
#include "humanoid_state_estimator/StateEstimator.h"
#include "humanoid_state_estimator/StateEstimatorNames.h"
#include "robot_model/RobotDescription.h"

namespace ocs2::humanoid::estimation {

/** What a state estimator is built from. */
struct StateEstimatorContext {
  /** The robot's URDF: the leg kinematics. */
  std::string urdfFile;
  /**
   * The contact points, in contact order: the MPC's contact frames
   * (model_settings), and the joints they hang from. A contact frame the URDF
   * does not have (the MPC adds its contact frames from the contact rectangles
   * of the task file) is replaced by its parent joint's frame, whose origin is
   * the contact point then: kalman_state_estimator's contact_height is that
   * joint's height while its foot stands. Empty parent names: the contact
   * frames only.
   */
  std::vector<std::string> contactFrameNames;
  std::vector<std::string> contactParentJointNames;
  /** The robot whose states are estimated; it must outlive the estimator. */
  const robot::model::RobotDescription* absl_nonnull description;
  /** The `kalman` estimator's noise model. */
  KalmanBaseEstimator::Config kalman;
};

/**
 * The state estimators of the robot process by name (the task file's
 * state_estimator): `ground_truth` and `kalman`. An unknown name is refused
 * with the list of the available ones. Immutable once built; its const methods
 * are thread-safe.
 */
class StateEstimatorRegistry {
 public:
  StateEstimatorRegistry() = default;

  /** The names, in registration order. */
  std::vector<std::string> names() const;
  /** "ground_truth (the backend's state), kalman (...)". */
  std::string availableNames() const;
  bool has(absl::string_view name) const;

  /**
   * The estimator `name` names, built from `context`. InvalidArgument listing
   * the available names for an unknown one, and the errors of building it (a
   * URDF without the contact frames, a noise model out of range).
   */
  absl::StatusOr<std::unique_ptr<StateEstimator>> create(
      absl::string_view name, const StateEstimatorContext& context) const;
};

}  // namespace ocs2::humanoid::estimation
