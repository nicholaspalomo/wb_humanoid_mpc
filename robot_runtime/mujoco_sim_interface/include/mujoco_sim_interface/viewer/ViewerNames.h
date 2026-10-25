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

namespace robot::mujoco_sim_interface {

/*
 * The names a MuJoCo viewer is selected by: MujocoSimConfig::viewer
 * (ViewerSettings.h), the robot app's --viewer and the launch files' `viewer`
 * variable. A list of them is a preference order: initSim() starts the first
 * one that starts. A new viewer gets its name here, its description in
 * availableViewers() and its factory in createViewer()
 * (MujocoViewerRegistry.h).
 */
// LINT.IfChange(viewer_names)
inline constexpr char kGlfwViewerName[] =
    "glfw";  // a GLFW window on DISPLAY (GlfwMujocoViewer)
inline constexpr char kWebViewerName[] =
    "web";  // offscreen EGL frames that a browser pulls over HTTP
            // (WebMujocoViewer)
inline constexpr char kNoViewerName[] = "none";  // no viewer (NoMujocoViewer)
// clang-format off
// LINT.ThenChange(//robot_runtime/mujoco_sim_interface/src/viewer/ViewerNames.cpp:viewer_descriptions, //robot_runtime/mujoco_sim_interface/src/viewer/MujocoViewerRegistry.cpp:viewer_factories, //robot_runtime/mujoco_sim_interface/README.md:viewer_names, //humanoid_nmpc/humanoid_common_mpc_app/robot/src/RobotAppFlags.cpp:robot_flags, //humanoid_nmpc/humanoid_common_mpc_app/robot/README.md:robot_flags, //robot_models/drc_atlas/drc_atlas_centroidal_mpc/launch/robot.textproto:robot_variables, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/launch/robot.textproto:robot_variables, //robot_models/unitree_g1/g1_centroidal_mpc/launch/robot.textproto:robot_variables, //robot_models/unitree_g1/g1_wb_mpc/launch/robot.textproto:robot_variables, //robot_models/unitree_r1/unitree_r1_centroidal_mpc/launch/robot.textproto:robot_variables, //tools/deploy/README.md:robot_environment, //Makefile:viewer_variable, //robot_models/tests/test_launch_files.py:viewer_names, //humanoid_nmpc/humanoid_rerun_viewer/python/humanoid_rerun_viewer/plot_config.py:mujoco_viewer_names)
// clang-format on

/** One viewer of the registry: its name and one line saying what it is. */
struct ViewerInfo {
  std::string name;
  std::string description;
};

/** Every viewer, in the order glfw, web, none. */
std::vector<ViewerInfo> availableViewers();

/** The names of availableViewers(), joined with ", ", for messages that list
 * the valid ones. */
std::string availableViewerNames();

}  // namespace robot::mujoco_sim_interface
