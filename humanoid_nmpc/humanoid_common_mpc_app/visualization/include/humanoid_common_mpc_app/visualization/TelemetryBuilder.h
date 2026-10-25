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
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/manifold/StateManifold.h"
#include "ocs2_mpc/SystemObservation.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "ocs2_robotic_tools/common/RotationTransforms.h"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"

#include "humanoid_common_mpc/acom/AngularCenterOfMass.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc_app/visualization/PolicySnapshot.h"
#include "humanoid_common_mpc_app/visualization/RobotStateDecoder.h"
#include "humanoid_common_mpc_app/visualization/SolutionTermEvaluator.h"
#include "humanoid_common_mpc_app/visualization/VisualizationConfig.h"
#include "humanoid_common_mpc_app/visualization/VisualizationModel.h"
#include "humanoid_mpc_msgs/telemetry_series.pb.h"

namespace ocs2::humanoid::visualization {

/**
 * Computes viz/telemetry for one robot/state sample: every group of the
 * telemetry contract of the Rerun bridge
 * (humanoid_nmpc/humanoid_rerun_viewer/README.md, "The plots: the telemetry
 * contract"), in its order, with the MPC's robot model and Pinocchio.
 *
 * Three sources, all at the sample's time t:
 *   - measured: the sample, in the MPC's generalized coordinates
 * (RobotStateDecoder);
 *   - reference: the target trajectories of the latest policy's command at t
 * (the old "mpc/desired" topics, which were the reference and not the plan).
 * Without a policy the reference is the measured robot at rest;
 *   - plan: the latest policy's state and input at t, held at the ends of its
 * horizon; "mpc" in the contact force groups. Without a policy the plan is the
 * measured robot at rest, with zero wrenches. The mpc_observation groups are
 * the latest observation and the policy's input at the observation's time (0
 * before the first observation or policy).
 *
 * Accelerations: the measured one is the difference of consecutive samples'
 * velocities (none for the first sample, after reset(), when the robot's clock
 * went back, or across a gap of a second or more); the reference's and the
 * plan's are central differences along their own trajectories over
 * kAccelerationHalfInterval, so that they do not jump when a new policy
 * arrives.
 *
 * Orientation: the MPC's q holds the base quaternion and v the body angular
 * velocity w_B (Composite(Translation, Spherical) root). The plots show the ZYX
 * Euler angles, their rates T_B(Theta)^-1 w_B (eulerZyxRatesForDisplay(), NaN
 * near a pitch of +-90 degrees) and w_B itself; the groups of generalized
 * coordinates (dofs/position, the coordinates of generalized_base) plot a
 * display buffer [p_W, (yaw, pitch, roll), q_j] of each source's q, and the
 * groups of generalized velocities and forces plot v as it is
 * (getBaseVelocityDofNames()). These are display values only, computed here on
 * the MPC node's visualization thread into buffers sized at construction.
 *
 * Balance, per source (BalanceQuantities.h): the whole-body CoM c and its
 * velocity c_dot (Pinocchio's, of the source's q and v); the DCM xi = c_xy +
 * c_dot_xy / omega, omega = sqrt(g / z0) with g the Pinocchio model's gravity
 * and z0 the pendulum height of the nominal state
 * (VisualizationModel::nominalState; NaN without one); the ZMP of the source's
 * contact wrenches at its contact frames (NaN below kMinNormalForceForZmp; the
 * measured wrenches carry no torque, so the measured ZMP is the
 * normal-force-weighted mean of the contact frames); and the ACoM attitude
 * (AngularCenterOfMass::acomAttitude() of the source's base quaternion and
 * joints) as roll, pitch, yaw (NaN for a robot without an ACoM network).
 *
 * The MPC solution: the solver's performance index and the cost and constraint
 * terms of the latest solution (SolutionTerms, computed on the solver thread by
 * SolutionTermEvaluator), held until the next one; NaN before it.
 *
 * The group layout (paths, names) is fixed at construction; build() only writes
 * values. Not thread-safe.
 */
class TelemetryBuilder {
 public:
  /** [s] Half the interval of the central differences of the reference's and
   * the plan's velocities. */
  static constexpr scalar_t kAccelerationHalfInterval = 2.5e-3;
  /** [s] Consecutive samples closer than the first or further apart than the
   * second give no measured acceleration. */
  static constexpr scalar_t kMinDifferenceInterval = 1.0e-5;
  static constexpr scalar_t kMaxDifferenceInterval = 1.0;
  /** [N] Below this net normal force a source has no ZMP (NaN): a flight phase.
   */
  static constexpr scalar_t kMinNormalForceForZmp = 1.0;

  /** The series of the mpc_solution/costs and mpc_solution/constraints groups
   * (SolutionTermEvaluator's names). */
  struct SolutionTermNames {
    std::vector<std::string> costs;
    std::vector<std::string> constraints;
  };

  /**
   * @return InvalidArgument when the model is not the MPC's
   * (checkVisualizationModel()), when a frame of config.telemetryFrames or a
   * contact frame is not a frame of the MPC's model. `termNames` names the
   * series of the solution's term groups (empty: groups without series).
   */
  static absl::StatusOr<std::unique_ptr<TelemetryBuilder>> Create(
      const VisualizationModel& model,
      const VisualizationConfig& config,
      const SolutionTermNames& termNames);
  static absl::StatusOr<std::unique_ptr<TelemetryBuilder>> Create(
      const VisualizationModel& model, const VisualizationConfig& config) {
    return Create(model, config, SolutionTermNames());
  }

  /**
   * The series of `measured`, with the latest `observation`, `policy` and
   * `solutionTerms` (any may be null; an observation or a policy of other
   * dimensions than the model's is ignored, and so are solution terms of other
   * sizes than the term names'). The message is the builder's and the next call
   * overwrites it.
   */
  const humanoid_mpc_msgs::TelemetrySeries& build(
      const DecodedRobotState& measured,
      const SystemObservation* absl_nullable observation,
      const PolicySnapshot* absl_nullable policy,
      const SolutionTerms* absl_nullable solutionTerms = nullptr);

  /** Forgets the previous sample, so that the next one has no measured
   * acceleration. */
  void reset() { hasPreviousSample_ = false; }

  /** The paths of the groups, in the order of the message. */
  std::vector<std::string> groupPaths() const;

 private:
  /** The sources of the plots: measured, reference and plan (sources_). */
  static constexpr size_t kNumSources = 3;
  /** The kinds of the groups of a tracked frame: pose, twist, acceleration and
   * wrench. */
  static constexpr size_t kNumFrameKinds = 4;
  /** The joint groups: the position, the velocity and the effort, each measured
   * and target. */
  static constexpr size_t kNumJointGroups = 6;
  /** The sources of the generalized forces: measured and reference. */
  static constexpr size_t kNumForceSources = 2;
  /** The panels of the generalized base: base_z, base_pitch and base_roll, and
   * their velocities. */
  static constexpr size_t kNumGeneralizedBasePanels = 3;

  /** The groups of one tracked frame, per kind (pose, twist, acceleration,
   * wrench) and source. */
  struct FrameGroups {
    pinocchio::FrameIndex frame;
    /** The contact the frame is, or -1. */
    int contact;
    std::array<std::array<int, kNumSources>, kNumFrameKinds> groups;
  };

  /** The state of one source at the sample's time, and the frame kinematics
   * computed from it. */
  struct Source {
    vector_t q;
    vector_t v;
    vector_t a;
    /**
     * For the plots: the ZYX Euler angles (yaw, pitch, roll) of the base
     * orientation in q, their rates of the body angular velocity in v
     * (eulerZyxRatesForDisplay(): NaN near a pitch of +-90 degrees), and q with
     * the angles in place of the quaternion, [p_W, (yaw, pitch, roll), q_j],
     * sized at construction.
     */
    vector3_t eulerAnglesZyx = vector3_t::Zero();
    vector3_t eulerZyxRates = vector3_t::Zero();
    vector_t displayCoordinates;
    feet_array_t<vector6_t> contactWrenches =
        makeFeetArray<vector6_t>(vector6_t::Zero());
    std::unique_ptr<PinocchioInterface::Data> data;
    // The balance quantities (updateBalance()); NaN where they are undefined.
    vector3_t com = vector3_t::Zero();
    vector2_t dcm = vector2_t::Zero();
    vector2_t zmp = vector2_t::Zero();
    /** The ACoM attitude as roll, pitch, yaw. */
    vector3_t acomRollPitchYaw = vector3_t::Zero();
    /** The joints of q, the ACoM network's input, sized at construction. */
    vector_t jointPositions;
  };

  TelemetryBuilder(const VisualizationModel& model,
                   const VisualizationConfig& config);
  absl::Status initialize(const VisualizationConfig& config,
                          const SolutionTermNames& termNames);

  int addGroup(const std::string& path, const std::vector<std::string>& names);
  /** The values of `group`, which must have at least one name: an empty
   * RepeatedField's mutable_data() may be null. */
  double* absl_nonnull values(int group) {
    return groupValues_[static_cast<size_t>(group)]->mutable_data();
  }
  void setValues(int group, const vector_t& vector);

  /** The generalized velocities of `state` and `input`
   * (MpcRobotModelBase::getGeneralizedVelocities()). */
  vector_t generalizedVelocities(const vector_t& state, const vector_t& input);
  void computeReference(const DecodedRobotState& measured,
                        const PolicySnapshot* absl_nullable policy);
  void computePlan(const DecodedRobotState& measured,
                   const PolicySnapshot* absl_nullable policy);
  void updateKinematics(Source* absl_nonnull source);
  /**
   * Writes the Euler rates and the display coordinates of `source` from its q,
   * v and Euler angles, which the caller wrote. Allocates nothing.
   */
  static void updateDisplayValues(Source* absl_nonnull source);
  /** Writes the balance quantities of `source` from its kinematics
   * (updateKinematics()) and contact wrenches. */
  void updateBalance(Source* absl_nonnull source);
  /** Writes the mpc_solution groups from `terms` (NaN without terms of the
   * layout's sizes). */
  void writeSolutionTerms(const SolutionTerms* absl_nullable terms);
  /** Takes the measured source's q, v and contact wrenches from `measured`, and
   * its accelerations from the previous sample. */
  void updateMeasuredSource(const DecodedRobotState& measured);
  /** Writes the base, contact force, generalized base and foot panels from the
   * sources and `measured`. */
  void writeBasePanels(const DecodedRobotState& measured);
  /** Writes the balance panels (CoM, ACoM angles, DCM, ZMP) of every source. */
  void writeBalancePanels();
  /** Writes the joint, generalized coordinate, force and contact wrench groups.
   */
  void writeCompleteGroups(const DecodedRobotState& measured);
  /** Writes the observation groups: zero without an observation of the model's
   * size; the input sampled from `plan`. */
  void writeObservation(const SystemObservation* absl_nullable observation,
                        const PolicySnapshot* absl_nullable plan);
  /** Writes the pose, twist, acceleration and wrench of every frame group, for
   * every source. */
  void writeFramePanels();

  /** The settings of robotModel_, which is never null. */
  const ModelSettings& modelSettings() const {
    return robotModel_->modelSettings;
  }

  PinocchioInterface pinocchioInterface_;
  std::unique_ptr<MpcRobotModelBase<scalar_t>> robotModel_;
  // VisualizationModel::stateManifold, retained (Create() refuses a null one).
  const StateManifold* absl_nonnull stateManifold_;
  feet_array_t<pinocchio::FrameIndex> contactFrames_{};
  /** [1/s] sqrt(g / z0) of the DCM; nullopt without a nominal state
   * (VisualizationModel::nominalState). */
  std::optional<scalar_t> pendulumFrequency_;
  /** The robot's ACoM network; null for a robot without one
   * (AngularCenterOfMass::Create() found none). */
  std::unique_ptr<AngularCenterOfMass> acom_;

  humanoid_mpc_msgs::TelemetrySeries series_;
  std::vector<google::protobuf::RepeatedField<double>* absl_nonnull>
      groupValues_;

  // The groups, by what they plot. Panel groups:
  std::array<int, kPositionDim> basePosition_{};
  std::array<int, kEulerAnglesZyxDim> baseRollPitchYaw_{};
  std::array<int, kEulerAnglesZyxDim> baseRollPitchYawRates_{};
  std::array<int, kRotationTangentDim> baseAngularVelocityLocal_{};
  std::array<int, kPositionDim> baseLinearVelocity_{};
  std::array<int, kRotationTangentDim> baseAngularVelocity_{};
  feet_array_t<int> normalForce_{};
  feet_array_t<int> tangentialForce_{};
  std::array<int, kNumGeneralizedBasePanels> generalizedBaseCoordinate_{};
  std::array<int, kNumGeneralizedBasePanels> generalizedBaseVelocity_{};
  feet_array_t<int> footAcceleration_{};
  feet_array_t<int> footVelocity_{};
  std::array<int, kPositionDim> comPosition_{};
  std::array<int, kHorizontalPositionDim> dcm_{};
  std::array<int, kHorizontalPositionDim> zmp_{};
  std::array<int, kEulerAnglesZyxDim> acomAngles_{};
  // Complete groups:
  std::array<int, kNumJointGroups> joints_{};
  std::array<int, kNumSources> dofPositions_{};
  std::array<int, kNumSources> dofVelocities_{};
  std::array<int, kNumForceSources> dofForces_{};
  feet_array_t<int> contactWrenchMpc_{};
  feet_array_t<int> contactWrenchMeasured_{};
  int observationState_ = 0;
  int observationInput_ = 0;
  int observationMode_ = 0;
  int solutionPerformance_ = 0;
  int solutionCosts_ = 0;
  int solutionConstraints_ = 0;
  size_t solutionCostCount_ = 0;
  size_t solutionConstraintCount_ = 0;
  std::vector<FrameGroups> frames_;

  // The three sources: measured, reference, plan.
  std::array<Source, kNumSources> sources_;
  // The reference's base, as the base panels plot it; its orientation is the
  // reference source's.
  vector3_t referenceBasePosition_ = vector3_t::Zero();
  vector3_t referenceBaseLinearVelocity_ = vector3_t::Zero();

  bool hasPreviousSample_ = false;
  scalar_t previousTime_ = 0.0;
  vector_t previousVelocities_;
};

}  // namespace ocs2::humanoid::visualization
