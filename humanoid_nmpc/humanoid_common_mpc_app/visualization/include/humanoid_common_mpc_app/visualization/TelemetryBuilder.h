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

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "ocs2_mpc/SystemObservation.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc_app/visualization/PolicySnapshot.h"
#include "humanoid_common_mpc_app/visualization/RobotStateDecoder.h"
#include "humanoid_common_mpc_app/visualization/VisualizationConfig.h"
#include "humanoid_common_mpc_app/visualization/VisualizationModel.h"
#include "humanoid_mpc_msgs/telemetry_series.pb.h"

namespace ocs2::humanoid::visualization {

/**
 * Computes viz/telemetry for one robot/state sample: every group of the telemetry contract of the Rerun bridge
 * (humanoid_nmpc/humanoid_rerun_viewer/README.md, "The plots: the telemetry contract"), in its order, with the MPC's
 * robot model and Pinocchio.
 *
 * Three sources, all at the sample's time t:
 *   - measured: the sample, in the MPC's generalized coordinates (RobotStateDecoder);
 *   - reference: the target trajectories of the latest policy's command at t (the old "mpc/desired" topics, which
 *     were the reference and not the plan). Without a policy the reference is the measured robot at rest;
 *   - plan: the latest policy's state and input at t, held at the ends of its horizon; "mpc" in the contact force
 *     groups. Without a policy the plan is the measured robot at rest, with zero wrenches.
 * The mpc_observation groups are the latest observation and the policy's input at the observation's time (0 before
 * the first observation or policy).
 *
 * Accelerations: the measured one is the difference of consecutive samples' velocities (none for the first sample,
 * after reset(), when the robot's clock went back, or across a gap of a second or more); the reference's and the
 * plan's are central differences along their own trajectories over kAccelerationHalfInterval, so that they do not
 * jump when a new policy arrives.
 *
 * The group layout (paths, names) is fixed at construction; build() only writes values. Not thread-safe.
 */
class TelemetryBuilder {
 public:
  /** [s] Half the interval of the central differences of the reference's and the plan's velocities. */
  static constexpr scalar_t kAccelerationHalfInterval = 2.5e-3;
  /** [s] Consecutive samples closer than the first or further apart than the second give no measured acceleration. */
  static constexpr scalar_t kMinDifferenceInterval = 1.0e-5;
  static constexpr scalar_t kMaxDifferenceInterval = 1.0;

  /**
   * @return InvalidArgument when the model is not the MPC's (checkVisualizationModel()), when a frame of
   *         config.telemetryFrames or a contact frame is not a frame of the MPC's model.
   */
  static absl::StatusOr<std::unique_ptr<TelemetryBuilder>> Create(const VisualizationModel& model, const VisualizationConfig& config);

  /**
   * The series of `measured`, with the latest `observation` and `policy` (either may be null; one of other dimensions
   * than the model's is ignored). The message is the builder's and the next call overwrites it.
   */
  const humanoid_mpc_msgs::TelemetrySeries& build(const DecodedRobotState& measured,
                                                  const SystemObservation* absl_nullable observation,
                                                  const PolicySnapshot* absl_nullable policy);

  /** Forgets the previous sample, so that the next one has no measured acceleration. */
  void reset() { hasPreviousSample_ = false; }

  /** The paths of the groups, in the order of the message. */
  std::vector<std::string> groupPaths() const;

 private:
  /** The groups of one tracked frame, per kind (pose, twist, acceleration, wrench) and source. */
  struct FrameGroups {
    pinocchio::FrameIndex frame;
    /** The contact the frame is, or -1. */
    int contact;
    std::array<std::array<int, 3>, 4> groups;
  };

  /** The state of one source at the sample's time, and the frame kinematics computed from it. */
  struct Source {
    vector_t q;
    vector_t v;
    vector_t a;
    feet_array_t<vector6_t> contactWrenches = makeFeetArray<vector6_t>(vector6_t::Zero());
    std::unique_ptr<PinocchioInterface::Data> data;
  };

  TelemetryBuilder(const VisualizationModel& model, const VisualizationConfig& config);
  absl::Status initialize(const VisualizationConfig& config);

  int addGroup(const std::string& path, const std::vector<std::string>& names);
  /** The values of `group`, which must have at least one name: an empty RepeatedField's mutable_data() may be null. */
  double* absl_nonnull values(int group) { return groupValues_[static_cast<size_t>(group)]->mutable_data(); }
  void setValues(int group, const vector_t& vector);

  /** The generalized velocities of `state` and `input` (MpcRobotModelBase::getGeneralizedVelocities()). */
  vector_t generalizedVelocities(const vector_t& state, const vector_t& input);
  void computeReference(const DecodedRobotState& measured, const PolicySnapshot* absl_nullable policy);
  void computePlan(const DecodedRobotState& measured, const PolicySnapshot* absl_nullable policy);
  void updateKinematics(Source* absl_nonnull source);

  PinocchioInterface pinocchioInterface_;
  std::unique_ptr<MpcRobotModelBase<scalar_t>> robotModel_;
  const ModelSettings& modelSettings_;
  feet_array_t<pinocchio::FrameIndex> contactFrames_{};

  humanoid_mpc_msgs::TelemetrySeries series_;
  std::vector<google::protobuf::RepeatedField<double>* absl_nonnull> groupValues_;

  // The groups, by what they plot. Panel groups:
  std::array<int, 3> basePosition_{};
  std::array<int, 3> baseRollPitchYaw_{};
  std::array<int, 3> baseLinearVelocity_{};
  std::array<int, 3> baseAngularVelocity_{};
  feet_array_t<int> normalForce_{};
  feet_array_t<int> tangentialForce_{};
  std::array<int, 3> generalizedBaseCoordinate_{};
  std::array<int, 3> generalizedBaseVelocity_{};
  feet_array_t<int> footAcceleration_{};
  feet_array_t<int> footVelocity_{};
  // Complete groups:
  std::array<int, 6> joints_{};
  std::array<int, 3> dofPositions_{};
  std::array<int, 3> dofVelocities_{};
  std::array<int, 2> dofForces_{};
  feet_array_t<int> contactWrenchMpc_{};
  feet_array_t<int> contactWrenchMeasured_{};
  int observationState_ = 0;
  int observationInput_ = 0;
  int observationMode_ = 0;
  std::vector<FrameGroups> frames_;

  // The three sources: measured, reference, plan.
  std::array<Source, 3> sources_;
  // The reference's base, as the base panels plot it.
  vector3_t referenceBasePosition_ = vector3_t::Zero();
  vector3_t referenceRollPitchYaw_ = vector3_t::Zero();
  vector3_t referenceBaseLinearVelocity_ = vector3_t::Zero();

  bool hasPreviousSample_ = false;
  scalar_t previousTime_ = 0.0;
  vector_t previousVelocities_;
};

}  // namespace ocs2::humanoid::visualization
