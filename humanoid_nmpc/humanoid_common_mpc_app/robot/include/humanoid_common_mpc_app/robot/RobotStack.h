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

#include "ocs2_robotic_tools/common/RobotInterface.h"

#include "humanoid_common_mpc_app/robot/RobotBackend.h"
#include "humanoid_common_mpc_app/robot/RobotController.h"
#include "humanoid_common_mpc_app/robot/RobotProcess.h"
#include "robot_ipc/Bus.h"
#include "robot_model/ContactEstimatorRegistry.h"
#include "robot_model/RobotDescription.h"

namespace ocs2::humanoid {

/**
 * Everything a robot binary builds before it starts (the setUpRobot() of humanoid_centroidal_mpc_robot and
 * humanoid_wb_mpc_robot, run by runRobot()): the formulation's controller models, the robot description, the bus, the
 * backend and its contact estimators, the controller and the process. Passive data; the members are declared in the
 * order they may be built, so that they are destroyed in the reverse one: the process first (it stops the realtime loop,
 * the configuration store's writer and the bus), then the controller (whose MPC link is registered on the bus), the
 * estimators, the backend, the bus, and last the models and the description the controller refers to. Every object is on
 * the heap, so moving the stack moves no object a pointer refers to.
 */
struct RobotStack {
  std::unique_ptr<RobotInterface> interface;
  std::unique_ptr<robot::model::RobotDescription> robotDescription;
  std::unique_ptr<robot::ipc::Bus> bus;
  std::unique_ptr<RobotBackend> backend;
  std::unique_ptr<robot::model::ContactEstimatorRegistry> contactEstimators;
  std::unique_ptr<RobotController> controller;
  std::unique_ptr<RobotProcess> process;
};

}  // namespace ocs2::humanoid
