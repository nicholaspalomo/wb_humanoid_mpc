# OCS2 Toolbox
A fork of the Optimal Control for Switched Systems (OCS2) library, built with Bazel and free of ROS and Boost.

## Summary
OCS2 is a C++ toolbox tailored for Optimal Control for Switched Systems (OCS2). The toolbox provides an efficient implementation of the following algorith

* SLQ: Continuous-time domin DDP
* iLQR: Discrete-time domain DDP
* SQP: Multiple-shooting algorithm based on HPIPM
* PISOC: Path integral stochatic optimal control

![legged-robot](https://leggedrobotics.github.io/ocs2/_static/gif/legged_robot.gif)

OCS2 handles general path constraints through Augmented Lagrangian or relaxed barrier methods. To facilitate the application of OCS2 in robotic tasks, it provides the user with additional tools to set up the system dynamics (such as kinematic or dynamic models) and cost/constraints (such as self-collision avoidance and end-effector tracking) from a URDF model. The library also provides an automatic differentiation tool to calculate derivatives of the system dynamics, constraints, and cost. Upstream OCS2 also ships ROS interfaces for deployment on robots; this fork has none, and the humanoid NMPC talks over its own IPC bus instead (`humanoid_nmpc/docs/distributed_runtime/README.md`). The toolbox’s efficient and numerically stable implementations in conjunction with its user-friendly interface have paved the way for employing it on numerous robotic applications with limited onboard computation power.

For more information refer to the project's [Documentation Page](https://leggedrobotics.github.io/ocs2/) 

## What this fork leaves out

The fork carries only the packages this repository builds (`lib/ocs2/BUILD.bazel`), and it does not depend on Boost:
configuration files are YAML only, read into the native `ocs2::PropertyTree`, and the integrators are native ports of
the odeint loops OCS2 used. The upstream code that was never built here, or that nothing here uses, is removed rather
than ported:

- loopshaping, throughout `core`, `oc`, `mpc` and `robotic_tools`;
- every ROS package: the ROS 1 `msgs` and `ros_interfaces` and their ROS 2 ports (`MPC_ROS_Interface`,
  `MRT_ROS_Interface`, `MRT_ROS_Dummy_Loop`, `DummyObserver`, the visualization helpers and the messages), which the
  IPC bus replaces (`humanoid_nmpc/humanoid_mpc_ipc`), and every package of `robotic_examples`, `raisim`, `mpcnet` and
  `perceptive`, including their ROS launch files;
- the solvers `frank_wolfe`, `ipm` and `slp`, `pinocchio/self_collision`, `self_collision_visualization` and
  `sphere_approximation`, and `python_interface`;
- the Boost INFO configuration format, `misc/Log.h`, `ddp/unsupported` and `DDP_DataCollector`;
- the catkin and CMake scaffolding: the `ocs2` metapackage, `sqp/blasfeo_catkin`, `core/cmake`, `doc` and the Jenkins
  pipeline.
