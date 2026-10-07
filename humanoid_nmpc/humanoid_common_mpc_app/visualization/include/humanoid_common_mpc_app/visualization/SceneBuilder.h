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
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ocs2_mpc/SystemObservation.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact/ContactWrenchMapper.h"
#include "humanoid_common_mpc_app/visualization/PolicySnapshot.h"
#include "humanoid_common_mpc_app/visualization/VisualizationConfig.h"
#include "humanoid_common_mpc_app/visualization/VisualizationModel.h"
#include "humanoid_mpc_msgs/line_strips.pb.h"
#include "humanoid_mpc_msgs/robot_model_instance.pb.h"
#include "humanoid_mpc_msgs/robot_state_sample.nproto.h"
#include "humanoid_mpc_msgs/spheres.pb.h"
#include "humanoid_mpc_msgs/visualization_scene.pb.h"

namespace ocs2::humanoid::visualization {

/** What one scene is drawn from; any of them may be absent. */
struct SceneInputs {
  /** The latest MPC observation: the contact markers and the collision spheres are drawn at its state. */
  const SystemObservation* absl_nullable observation = nullptr;
  /** The latest MPC solution: the plan, the terminal robots, and the contact wrenches at the observation. */
  const PolicySnapshot* absl_nullable policy = nullptr;
  /** Changes with every new policy; the plan is computed again only when it does. */
  uint64_t policyVersion = 0;
  /** The latest robot/state sample: the measured robot, with every joint. Without one, the observation's state. */
  const msgs::RobotStateSample* absl_nullable robotState = nullptr;
};

/**
 * Computes viz/scene (humanoid_nmpc/humanoid_rerun_viewer/README.md, "The 3D scene") from the MPC's observation and
 * policy and the robot's measured state, with the MPC's robot model and Pinocchio:
 *
 *   - the robot instances as world poses of every URDF link: `measured` with every joint of the URDF (a Pinocchio
 *     model of the full URDF, from the robot/state sample), or the observation's state when there is no sample;
 *     `terminal_state`, the last node of the plan, and `terminal_target`, the reference at the plan's last time, with
 *     the MPC's model (its fixed joints at their URDF zero);
 *   - at the observation: the contact force of every stance foot, an arrow ending at the foot's center of pressure,
 *     their net center of pressure, the four statically equivalent corner forces of every stance foot, and the
 *     collision spheres. The contact wrenches are the policy's input at the observation's time, taken through the
 *     state-aware world-frame accessors, so that they are right for every input parameterization, and not the
 *     observation's input, which the MuJoCo simulations leave zero. A stance foot whose normal force
 *     is below kMinNormalForceForCop has its center of pressure at its contact frame rather than NaN;
 *   - the plan: the paths of the task file's rerun_plan_frames, of the base and of the CoM projected to the ground, and
 *     the footholds where a foot lands within the horizon.
 *
 * Every scene carries every marker path, empty when there is nothing to draw, so that what is no longer true
 * disappears. Not thread-safe: one thread builds the scenes (the visualization thread).
 */
class SceneBuilder {
 public:
  /** [N] Below this normal force a stance foot's center of pressure is its contact frame's origin. */
  static constexpr scalar_t kMinNormalForceForCop = 1.0;

  /**
   * @return InvalidArgument when the model is not the MPC's (checkVisualizationModel()), when the URDF does not parse,
   *         when a frame of config.planFrames or a contact frame is not in the MPC's model, or when the task file's
   *         contact polygons do not load or are not rectangles whose corner frames the model has. Collision spheres
   *         that do not load are not an error: none is drawn.
   */
  static absl::StatusOr<std::unique_ptr<SceneBuilder>> Create(const VisualizationModel& model, const VisualizationConfig& config);

  /**
   * Writes the scene of `inputs` into `scene`, replacing what it held. Its time is the robot/state sample's, or the
   * observation's without a sample. An observation or a policy of other dimensions than the model's is ignored.
   *
   * @return FailedPrecondition, with `scene` cleared, when there is neither an observation nor a sample to draw.
   */
  absl::Status build(const SceneInputs& inputs, humanoid_mpc_msgs::VisualizationScene* absl_nonnull scene);

  /** The links of the MPC's robot instance: the BODY frames of the MPC's model, in frame order. */
  const std::vector<std::string>& mpcLinkNames() const { return mpcLinkNames_; }

 private:
  struct CollisionSphere {
    pinocchio::FrameIndex frame;
    scalar_t radius;
  };

  /** A joint of the full URDF model and where the robot/state sample carries it. */
  struct FullModelJoint {
    std::string name;
    int idxQ;
    int nq;
    /** Index into the sample's joint arrays; -1 when the sample does not carry the joint. */
    int sampleIndex = -1;
  };

  SceneBuilder(const VisualizationModel& model, const VisualizationConfig& config);
  absl::Status initialize(const VisualizationModel& model, const VisualizationConfig& config);

  void writeMeasuredFromSample(const msgs::RobotStateSample& sample, humanoid_mpc_msgs::RobotModelInstance* absl_nonnull instance);
  void updateGroundHeight(size_t mode);
  void writeObservationMarkers(const SystemObservation& observation,
                               const PolicySnapshot* absl_nullable policy,
                               humanoid_mpc_msgs::VisualizationScene* absl_nonnull scene);
  void computePlan(const PolicySnapshot& policy);
  void clearPlan();

  PinocchioInterface pinocchioInterface_;
  std::unique_ptr<MpcRobotModelBase<scalar_t>> robotModel_;
  const ModelSettings& modelSettings_;

  PinocchioInterface::Model fullModel_;
  std::unique_ptr<PinocchioInterface::Data> fullData_;
  vector_t fullConfiguration_;
  std::vector<FullModelJoint> fullJoints_;
  std::vector<std::string> sampleJointNames_;
  bool hasSampleJointIndices_ = false;

  std::vector<pinocchio::FrameIndex> fullLinkFrames_;
  std::vector<std::string> fullLinkNames_;
  std::vector<pinocchio::FrameIndex> mpcLinkFrames_;
  std::vector<std::string> mpcLinkNames_;

  feet_array_t<pinocchio::FrameIndex> contactFrames_{};
  feet_array_t<std::array<pinocchio::FrameIndex, 4>> cornerFrames_{};
  std::vector<ContactWrenchMapper<4>> cornerForceMappers_;
  std::vector<CollisionSphere> collisionSpheres_;
  std::vector<pinocchio::FrameIndex> planFrames_;

  /** [m] The height the planned CoM is projected to: the stance feet's mean height at the last observation. */
  scalar_t groundHeight_ = 0.0;

  // The plan of the policy of planVersion_.
  bool hasPlan_ = false;
  uint64_t planVersion_ = 0;
  humanoid_mpc_msgs::LineStrips planEndEffectors_;
  humanoid_mpc_msgs::LineStrips planBase_;
  humanoid_mpc_msgs::LineStrips planCom_;
  humanoid_mpc_msgs::Spheres planFootholds_;
  bool hasTerminalState_ = false;
  humanoid_mpc_msgs::RobotModelInstance terminalState_;
  bool hasTerminalTarget_ = false;
  humanoid_mpc_msgs::RobotModelInstance terminalTarget_;
};

}  // namespace ocs2::humanoid::visualization
