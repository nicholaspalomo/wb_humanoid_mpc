Install the xpad driver: https://github.com/paroj/xpad

## The FSM state and the joysticks

The Base Controller tab follows the state the simulator's FSM bridge publishes on `/humanoid/fsm_state`
(`SimFsmBridge`, transient-local): `<mode>,<gantry>,<controller resets>`, for example `JOINT_PD,GANTRY_LOCKED,3`.
`remote_control/fsm_state.py` parses it; the mode selector and the gantry checkbox mirror it.

The virtual joysticks are re-centered (`fsm_state.should_recenter`) whenever the robot stops being walked by the MPC
without the stick having moved:

- on every transition into a passive mode (`ZERO_TORQUE`, `JOINT_PD`, `GRAVITY_COMP`, `SAFETY`);
- on every new gantry lock;
- on every controller reset the simulation loop counts: a fall caught on the gantry, a `LOCK_GANTRY`, or the simulator
  putting the robot back in its initial state - also while the gantry was already locked in `JOINT_PD`, which changes
  neither the mode nor the gantry and is covered only by the count.

A stick left forward would otherwise keep commanding a walk, and re-entering `WB_MPC` would execute it. The height
slider is not touched: it is the gantry height while the gantry is locked. A connected Xbox controller writes its own
stick positions every cycle, so the release applies to the on-screen sticks only.
