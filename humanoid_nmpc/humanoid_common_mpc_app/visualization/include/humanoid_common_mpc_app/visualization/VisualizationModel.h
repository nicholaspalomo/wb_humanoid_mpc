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

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "ocs2_core/manifold/StateManifold.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"

namespace ocs2::humanoid::visualization {

/**
 * The robot as the visualization sees it: the MPC's model and the files it was built from. The Pinocchio model and the
 * robot model are borrowed only while a builder or the publisher is constructed, which copies the first and clones the
 * second, so construct them before the solver thread starts using `mpcRobotModel`. The state manifold is retained.
 */
struct VisualizationModel {
  /** The robot's task file: the visualization fields, the contact polygons and the collision spheres. */
  std::string taskFile;
  /** The robot's URDF: the measured robot is drawn with every joint of it. */
  std::string urdfFile;
  /**
   * The MPC's Pinocchio model, as loadCustomPinocchioInterface() builds it: the joints that are not MPC joints fixed,
   * the contact frames and the contact polygon corners added, and a Composite(Translation, Spherical) root, whose
   * configuration is the position and the unit quaternion (x, y, z, w) of the base and whose velocity is the world
   * linear and the body angular velocity.
   */
  const PinocchioInterface* absl_nullable pinocchioInterface = nullptr;
  /**
   * The MPC's robot model, the one its dynamics use (CentroidalMpcRobotModel, its BasisInputsModelDecorator, or
   * WBAccelMpcRobotModel); its ModelSettings must outlive the visualization.
   */
  const MpcRobotModelBase<scalar_t>* absl_nullable mpcRobotModel = nullptr;
  /**
   * The manifold the MPC's state lives on (its interface's getStateManifold(): the base quaternion block on SO(3)),
   * along which the scene and the telemetry interpolate the plan's and the target's states (samplePlan(),
   * sampleTarget()). Required: checkVisualizationModel() refuses a null one, as every humanoid state has the quaternion
   * block. Unlike the two models it is retained, by the builders and the publisher: it must outlive them, as the MPC
   * interface that owns it outlives the MPC node's publisher. Const and immutable, so the visualization thread reads it
   * while the solver uses it.
   */
  const StateManifold* absl_nullable stateManifold = nullptr;
};

/**
 * InvalidArgument when a model pointer or the state manifold is null, when the robot model's base is not the
 * Composite(Translation, Spherical) root (its configuration must have kBaseConfigurationDim and its velocity
 * kBaseVelocityDim entries before the getJointDim() joints: nq = 7 + nj, nv = 6 + nj), when the Pinocchio model is not
 * the MPC's (its configuration q and its velocity v must have the robot model's getConfigurationDim() and
 * getVelocityDim() entries), or when the state manifold is not one of the robot model's state (its ambient size must be
 * getStateDim()).
 */
absl::Status checkVisualizationModel(const VisualizationModel& model);

}  // namespace ocs2::humanoid::visualization
