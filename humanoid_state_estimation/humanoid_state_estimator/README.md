# humanoid_state_estimator

The robot process's state estimator stage, selected by the task file's `state_estimator`
(`StateEstimatorRegistry`):

| Name | What the controller receives |
|---|---|
| `ground_truth` (default) | the backend's state as it is |
| `kalman` | the backend's state with the base position and linear velocity estimated from the IMU and the leg kinematics |

`kalman` is `KalmanBaseEstimator` behind `KalmanStateEstimator`: a linear Kalman filter (`KalmanFilter`) over the base
position and velocity and one world position per contact point, after Bloesch et al. (2013), with the orientation
taken from the IMU. Its noise model is the task file's `kalman_state_estimator` block
(`kalman_state_estimator_config.proto`).

## Contact points

`ContactKinematics` evaluates every contact point on a fixed-base Pinocchio model of the URDF. A point is a
`ContactPointSpec`: a URDF link and the point's offset in that link's frame. The robot process places them where the
MPC's model has its contacts (`stateEstimatorContacts()` in `humanoid_common_mpc_app/robot`): a 3-DoF contact at its
offset on its parent joint's child link, a 6-DoF contact at its sole or pad center, and a contact on the MPC's
`root_joint` on the URDF's root link.

## Measurements

Per contact in stance, three rows:

- the contact's position relative to the base, `c_i - p = R p_rel_i`;
- the base velocity the stance contact gives, `v = -R (omega x p_rel_i + v_rel_i)`;
- the contact's height, `e_z^T c_i = h_i`, with `h_i` the height of the surface the contact stands on.

The caller gives the heights every cycle in a `ContactEstimationInputs` (`StateEstimator::estimate()`): one entry per
contact, sized once, each a height or none. A contact without a height has no height row; its position and velocity
rows still apply, so the base height is then observed through the other contacts only. A freshly constructed
`ContactEstimationInputs` gives every contact the height 0, flat ground at z = 0.

The robot process fills the heights from a `ContactSurfaceHeightSource` (`humanoid_common_mpc_app/robot/README.md`):
the flat ground at `kalman_state_estimator.nom_pos_contact_rt_world_ewrt_world_z` while walking, or the patches of the
multi-contact plan the MPC tracks, looked up at the contact positions of the previous estimate
(`StateEstimator::estPosContactsRtWorldEwrtWorld()`, empty for `ground_truth`).

## Contact-trust ramp

A contact is least reliable just before it lifts off and just after it touches down: the foot may already be rolling
or still be settling. With `kalman_state_estimator.contact_trust_ramp_duration` (T, seconds) positive, the filter
trusts a stance contact less near its planned switches. The caller gives each contact's time since its last and until
its next planned switch in the `ContactEstimationInputs` (the robot process takes them from the mode schedule of the
MPC policy in use, `PlannedSwitchTimesSource`). With d the smaller of the two, the stance random walk and the kinematic
velocity and contact height noise standard deviations of that contact are multiplied by

    w = clamp(1 - d / T, 0, 1),   s = 1 + (S - 1) w^2 (3 - 2 w),

with S = `contact_trust_ramp_max_scale` (>= 1): S at the switch, 1 from T away on, continuously differentiable in d
(`contactTrustScale()`, `ContactTrustRamp.h`). An unknown switch (+infinity, the default of the inputs and what the
robot process writes while no policy is in use) gives 1. The relative position row keeps its noise: the leg
kinematics stay exact near a switch, the assumption that the contact does not move is what weakens.

The ramp is off by default (T = 0). Off, `update()` rewrites none of the noise for it and the filter is bit for bit the
one without the ramp.

## Realtime

`estimate()`, `reset()` and `KalmanBaseEstimator::update()` allocate nothing once the estimator is built; the contact
positions are copied into storage sized at construction. `:test_state_estimator_allocations` counts the heap
allocations of a walk with stance changes, a reset, contact heights cleared and set again and planned switch times set
and cleared, with the contact-trust ramp off and on.

## Tests

| Target | What it checks |
|---|---|
| `:test_kalman_filter`, `:test_kalman_filter_allocations` | the channel-based filter and its allocation-free step |
| `:test_contact_kinematics` | the contact points' positions and velocities against the URDF |
| `:test_contact_trust_ramp` | the ramp's scale: 1 outside the window, when off and with no known switch, the maximum at the switch, monotone and C1 |
| `:test_kalman_base_estimator` | the walk of the prismatic biped tracked with exact and noisy sensors; the base height observed through the contact heights: fresh inputs equal explicit zeros bit for bit, a walk on a 0.3 m box top converges to the box's height, and a contact without a height keeps its position and velocity rows; the ramp off ignores the switch times bit for bit, on it raises a contact's covariance near its switch; the refusals |
| `:test_state_estimator_registry` | the names, `ground_truth` passing its input through unchanged, `kalman` estimating the base and its contact positions |
| `:test_state_estimator_allocations` | `estimate()` and `reset()` allocate nothing |
