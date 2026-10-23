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

// Pinocchio forward declarations must be included first.
#include "pinocchio/fwd.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "ocs2_robotic_tools/common/RotationTransforms.h"

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc_app/robot/InitialSimState.h"
#include "humanoid_nmpc/humanoid_common_mpc_app/visualization/test/VisualizationTestRobot.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotState.h"

namespace ocs2::humanoid {
namespace {

using visualization::test::Formulation;
using visualization::test::RobotFiles;
using visualization::test::TestRobot;

// The simulator starts in the MPC's initial state, in every formulation: the base's position and orientation read
// through the robot model's layout-agnostic accessors, the MPC joints by name, every other joint at zero.
TEST(InitialSimStateTest, TheSimulatorStartsInTheMpcState) {
  for (const std::pair<RobotFiles, Formulation>& configuration : visualization::test::shippedConfigurations()) {
    SCOPED_TRACE(configuration.first.taskFile);
    const std::unique_ptr<TestRobot> robot = TestRobot::load(configuration.first, configuration.second);
    const MpcRobotModelBase<scalar_t>& model = robot->robotModel();
    // A tilted base turned past +pi, its yaw unwrapped as a reference knot may hold it.
    const vector3_t eulerZyx(3.5, 0.2, -0.1);
    vector_t state = robot->nominalState();
    const vector3_t position = model.getBasePosition(state) + vector3_t(0.3, -0.2, 0.05);
    model.setBasePosition(state, position);
    model.setBaseOrientationEulerZyx(state, eulerZyx);
    vector_t joints = model.getJointAngles(state);
    for (Eigen::Index joint = 0; joint < joints.size(); ++joint) {
      joints[joint] += 0.01 * static_cast<scalar_t>(joint + 1);
    }
    model.setJointAngles(state, joints);

    const absl::StatusOr<robot::model::RobotDescription> description = robot::model::RobotDescription::Create(robot->urdfFile());
    ASSERT_TRUE(description.ok()) << description.status();
    const robot::model::RobotState simState = createInitialSimState(*description, robot->modelSettings(), model, state);

    EXPECT_EQ(simState.getRootPositionInWorldFrame(), position);
    // The orientation is the model's stored quaternion (getBaseOrientation), bit for bit, which is the rotation of the
    // Euler angles it was set from up to round-off: the quaternion layout stores the coefficients, not the angles.
    const quaternion_t rootOrientation = simState.getRootRotationLocalToWorldFrame();
    EXPECT_EQ(rootOrientation.coeffs(), model.getBaseOrientation(state));
    EXPECT_NEAR(rootOrientation.norm(), 1.0, 1.0e-15);
    EXPECT_TRUE(rootOrientation.toRotationMatrix().isApprox(getRotationMatrixFromZyxEulerAngles<scalar_t>(eulerZyx), 1.0e-14))
        << rootOrientation.toRotationMatrix() << "\n"
        << getRotationMatrixFromZyxEulerAngles<scalar_t>(eulerZyx);

    const std::vector<std::string>& mpcJoints = robot->modelSettings().mpcModelJointNames;
    for (size_t joint = 0; joint < mpcJoints.size(); ++joint) {
      EXPECT_EQ(simState.getCheckedJointPosition(description->getJointIndex(mpcJoints[joint])), joints[static_cast<Eigen::Index>(joint)])
          << mpcJoints[joint];
    }
    for (const std::string& name : description->getJointNames()) {
      if (!absl::c_linear_search(mpcJoints, name)) {
        EXPECT_EQ(simState.getCheckedJointPosition(description->getJointIndex(name)), 0.0) << name;
      }
    }
  }
}

}  // namespace
}  // namespace ocs2::humanoid
