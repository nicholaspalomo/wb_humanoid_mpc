# Dodgeball: throwing a physical ball at the robot

A push test you can aim. The GUI's **Dodgeball** tab throws a dodgeball at the robot's base from an angle, a
distance and a launch speed you choose, at a mass you choose; the simulator compiles a real sphere into the MuJoCo
scene, flies it, and lets the impact, the bounce and the recovery all be physics rather than a model of physics.

The point is disturbance rejection testing that is *repeatable and directed*. A wrench applied to the base tells
you how the controller handles a step in momentum; a ball tells you how it handles one delivered somewhere
specific — off a shoulder, into a swing foot mid-flight, low into a shin — which is the case that actually breaks
walking controllers.

---

## 1. The pipeline

```text
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  remote_control (Python, Tkinter)                                                        │
  │                                                                                          │
  │   DodgeballTab                    dodgeball.py                                           │
  │   ├─ azimuth      slider  ──┐     ├─ spawn_offset(throw)       where it starts, in the   │
  │   ├─ elevation    slider  ──┤     │                            robot's own yaw frame     │
  │   ├─ distance     slider  ──┼──►  ├─ launch_speed(throw)       slider's, or the minimum  │
  │   ├─ speed        slider  ──┤     ├─ flight_time(throw)        direct root of |v| = s    │
  │   ├─ mass         slider  ──┤     ├─ launch_velocity(throw)    aimed, lifted by g t / 2  │
  │   ├─ randomize    checkbox──┘     ├─ impact_momentum(throw)    m |v_arrival|             │
  │   └─ THROW button ────────────►   └─ throw_message(throw)      the message below         │
  │   (every slider holds its range: SliderRow(clamp_to_range=True))                         │
  └───────────────────────────────────────────┬──────────────────────────────────────────────┘
                                              │  DodgeballThrow on operator/dodgeball_throw (IPC bus;
                                              │  every message is delivered - a throw is an event,
                                              │  not a stream; none of them may be dropped)
                                              ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  robot process  (humanoid_common_mpc_app/robot)                                          │
  │                                                                                          │
  │   IO thread: dodgeballThrowFromMessage()   required fields, finite, flight_time >= 0,    │
  │              mass > 0; the throw goes into an SpscQueue<DodgeballThrow>                  │
  │   realtime thread: SimFsmBridge            pops it first, then one FSM command           │
  │   SimFallRecovery::update                  catches the robot -> lockGantry()             │
  └───────────────────────────────────────────┬──────────────────────────────────────────────┘
                                              │  MujocoSimInterface::throwDodgeball(...)
                                              ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  mujoco_sim_interface :: MujocoSimInterface          (simulation thread owns mjData)     │
  │                                                                                          │
  │   ── at construction ─────────────────────────────────────────────────────────────────   │
  │   config.projectile == ""   ──► mj_loadXML(scene)           the scene exactly as on disk │
  │   config.projectile == name ──► mj_parseXML  ─► addProjectileToSpec(spec, ball)          │
  │                                              ─► mj_compile                               │
  │                                 resolve the ball's ids, parkProjectile()                 │
  │   every joint-damping write goes through setRobotJointDamping (skips the ball)           │
  │                                                                                          │
  │   ── every simulation step: applyDodgeball() ─────────────────────────────────────────   │
  │     ├─ cancel requested?   (lockGantry on another thread) park, drop the impulse         │
  │     ├─ a throw is staged?  mj_kinematics, then setProjectileMass(clamped mass),          │
  │     │                      clearProjectileLaunch(...), place, set qvel, arm              │
  │     │                      (no ball compiled in? schedule projectileArrivalMomentum)     │
  │     └─ armed?              ProjectileRestMonitor: slow for 0.25 s or 10 s in play: park  │
  │                                                                                          │
  │   groundTruthContactMask(..., ignoreBodyId)  and  ZMP marker's groundReaction(...)       │
  │        └─► the ball is never mistaken for the ground                                     │
  └───────────────────────────────────────────┬──────────────────────────────────────────────┘
                                              ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  mujoco_sim_interface :: Projectile                  (pure functions, unit-tested)       │
  │                                                                                          │
  │   projectileFromName / availableProjectiles     the registry sim_projectile names        │
  │   contactDampRatioForRestitution(e)             restitution -> solref damping ratio      │
  │   addProjectileToSpec(spec, ball, name)         appended LAST; gravcomp, priority,       │
  │                                                 condim 6, colliding - all compiled in    │
  │   setProjectileCollisionEnabled(m, b, on)       geom flags AND body-level aggregates     │
  │   setProjectileMass(m, b, mass)                 mass, inertia, derived constants         │
  │   setRobotJointDamping(m, b, damping)           every robot joint, never the ball        │
  │   clearProjectileLaunch(m, d, b, launch, g)     slide the start along its own path       │
  │   projectileArrivalMomentum(v, t, m, g)         the no-ball fallback impulse             │
  │   ProjectileRestMonitor                         when a thrown ball is finished with      │
  └──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Choosing the ball

```textproto
# robot_models/<robot>/<robot>_centroidal_mpc/config/mpc/task.textproto
sim_projectile: "dodgeball"
```

Selected **by name**, like every other pluggable component in this repository, so a second projectile is a new entry
in `availableProjectiles()` rather than a second boolean. Remove the field, or leave it empty, and the scene compiles
exactly as it is on disk — the code's own default, so a robot that has not opted in is unchanged.

| | `dodgeball` |
| --- | --- |
| diameter | 0.216 m — a regulation 8.5-inch ball |
| mass | 0.45 kg by default; the **Ball mass** slider sets 0.1 – 5.0 kg per throw |
| restitution | 0.80 — measured 0.797 at the simulator's 0.5 ms step, 0.842 at 1 ms |
| sliding friction | 0.60 |
| rolling and torsional friction | 5 mm — from 3 m/s a ball rolls to rest in about seven seconds |
| contact time constant | 20 ms |

At the default mass a 25 m/s throw carries about 11 N s. For scale, Atlas at 159.7 kg takes about 0.07 m/s of base
velocity from that, before the bounce adds more: the default ball is a shove rather than a demolition, and what
makes a throw interesting is *where* and *when* it lands. At the top of the mass slider, 5 kg arriving at 25 – 28 m/s
is 125 – 140 N s, roughly 0.8 m/s for Atlas — deliberately more than a balance controller should be expected to
absorb standing still.

### Restitution, and why MuJoCo does not have it

MuJoCo models a contact as a spring-damper and exposes `solref = (timeconst, dampratio)`. A damping ratio below one
is what makes a contact bounce, and the two are related by the logarithmic decrement of a damped oscillator:

$$e \;=\; \exp\!\left(\frac{-\pi \zeta}{\sqrt{1 - \zeta^{2}}}\right) \qquad \Longleftrightarrow \qquad \zeta \;=\; \frac{-\ln e}{\sqrt{\pi^{2} + \ln^{2} e}}.$$

So the registry names the number an engineer actually knows and `contactDampRatioForRestitution` converts it:
$e = 0.80$ gives $\zeta = 0.0708$. That relation only holds for the contact the ball actually gets, which takes two
more things:

- **Priority.** Two geoms of equal priority get the *average* of their solrefs. Every floor and robot geom in the
  shipped scenes carries `solref = (0.004, 1.0)`, so without priority the ball's contacts ran at a damping ratio of
  about 0.54, a restitution of about 0.25, whatever the ball was configured with. The ball's geom has `priority = 1`,
  so its solref, solimp, friction and condim govern every contact it makes.
- **A time constant the integrator can resolve.** A lightly damped contact rings at about $1/(t_c \zeta)$. At 2 ms
  and $\zeta = 0.07$ that is ~7000 rad/s, seven radians per 1 ms step, and the discrete contact then *adds* energy —
  rebounds of 4 to 50 times the arrival speed were measured. At 20 ms it is ~740 rad/s, resolved at both 0.5 ms and
  1 ms, and 20 ms is also about how long a foam ball is in contact with what it hits.

### Changing the mass after the scene is compiled

The size, the friction and the restitution are fixed when the scene is compiled; the mass is not, because it is the
one property worth sweeping. Mass and speed both scale the momentum, but they are not interchangeable: a heavy ball
carries its momentum *through* the contact while a light one gives it back and bounces away.

Assigning `body_mass` is not enough. MuJoCo normalizes each contact's reference acceleration by `body_invweight0`,
an inverse mass the compiler worked out once; left stale, the *contact* changes and the restitution starts depending
on the mass. `setProjectileMass` therefore writes, besides the mass and the sphere's $\tfrac{2}{5} m r^{2}$ inertia,
every constant the compiler derived from them — `body_invweight0`, `dof_invweight0`, `dof_M0` and `body_subtreemass`
— in closed form. For a centered sphere on its own free joint the joint-space inertia is
$\mathrm{diag}(m, m, m, I, I, I)$, so each is the diagonal or its inverse, and a test checks every one against a
fresh compile at the new mass.

It does *not* call `mj_setConst`, the function MuJoCo provides for this, for three reasons found the hard way: it
uses the `mjData` it is handed as scratch and leaves its `qpos` overwritten (handed the live state, it would reset
the walking robot); it recomputes `mjModel::stat` from the bounding box, discarding the scene's own
`<statistic extent center>` and moving the viewer's near clip plane from 1.5 cm to half a meter; and it rebuilds
fields by accumulating into them, so the render thread, which reads the same model, would see partial sums. The
closed form is a handful of single stores with no allocation, so the simulator retunes the mass on every throw.

---

## 3. The geometry of a throw

All of it is computed in `remote_control/tk_app/dodgeball.py`, in the robot's own yaw frame, so that "behind the
robot" means behind the robot however it is facing. The simulator only rotates the result into the world.

Given an azimuth $\psi$ about $z$, an elevation $\theta$ in the vertical plane and a distance $d$, the ball starts at

$$\mathbf{p}_{\text{spawn}} \;=\; \mathbf{p}_{\text{base}} \;+\; d \begin{pmatrix} \cos\theta \cos\psi \\ \cos\theta \sin\psi \\ \sin\theta \end{pmatrix}.$$

**The speed slider is the launch speed** $s$, as its label says. A ball that is to arrive at the base after $t$
seconds must leave with

$$\mathbf{v}_{\text{launch}} \;=\; \frac{\boldsymbol{\Delta}}{t} \;+\; \begin{pmatrix} 0 \\ 0 \\ \tfrac{1}{2} g t\end{pmatrix}, \qquad \boldsymbol{\Delta} = \mathbf{p}_{\text{base}} - \mathbf{p}_{\text{spawn}},$$

because the vertical term integrates to exactly the drop it cancels. Requiring $|\mathbf{v}_{\text{launch}}| = s$ gives
a quadratic in $t^{2}$,

$$\tfrac{1}{4} g^{2} t^{4} \;+\; \left(g \Delta_z - s^{2}\right) t^{2} \;+\; d^{2} \;=\; 0,$$

whose smaller root is the direct throw and the larger a lob; `flight_time` takes the direct one, in the form that
reduces to $d / s$ as $g \to 0$. It has a real root only for $s^{2} \ge g\,(d + \Delta_z)$: slower than that, no aim
reaches the base, so `launch_speed` raises the slider's speed to that minimum and the preview says so. From 3 m level
with the base the minimum is 5.4 m/s; from 8 m it is 8.9 m/s, with a flight of $\sqrt{2d/g} \approx 1.3$ s.

The ball arrives with $\mathbf{v}_{\text{launch}} - (0, 0, g t)$, whose speed follows from energy conservation,
$v_{\text{arrival}}^{2} = s^{2} + 2 g\,(z_{\text{spawn}} - z_{\text{base}})$, and the preview's momentum is
$m\,v_{\text{arrival}}$. A ball dropped from above arrives faster than it left.

The base pose is read at the moment of the throw, in the simulation thread — not when the button was pressed — so a
robot that turns during the flight does not drag the ball around with it.

The **Randomize** checkbox samples the azimuth and the elevation only. Distance, speed and mass stay where you put
them, because those are the magnitudes you sweep when you are looking for the limit; the direction is the part you
want unbiased. The mass has no effect on the flight itself, only on what arrives.

### A spawn point inside the robot or under the floor

The GUI aims without knowing the robot's shape or where the floor is. A short throw from the side can start inside an
outstretched arm, and a negative elevation from far away starts below the floor — and a plane in MuJoCo collides as a
half-space. MuJoCo resolves a ball that starts inside something as a stiff spring and fires it out at tens of meters
per second, whatever speed was asked for.

`clearProjectileLaunch` keeps the aim by sliding the start point along the ball's **own ballistic path**,

$$\mathbf{p}(\tau) = \mathbf{p}_0 + \mathbf{v}_0 \tau - \tfrac{1}{2} g \tau^{2} \hat{\mathbf{z}}, \qquad \mathbf{v}(\tau) = \mathbf{v}_0 - g \tau \hat{\mathbf{z}},$$

to the smallest $|\tau|$ at which the ball is 2 cm from every geom it could collide with (measured with
`mj_geomDistance`): backwards in time out of an arm along the approach, forwards out of the floor along the climb.
The flight time changes by $\tau$ and the simulator logs a warning. If nothing within a second before the throw or 80%
of the flight after it is clear, the throw is refused with a message naming the slider to change.

---

## 4. Things that fail silently

Each of these produces no error anywhere and a plausible-looking simulation, which is why each has a test.

**The ball must be the last body.** Several things here take the *first* body carrying a free joint to be the robot:
`robotCentroidalState`'s center of mass, the ZMP and DCM markers, the viewer's tracking camera, and every read of
`qpos[3..6]` as the base quaternion. A ball inserted ahead of the robot quietly *becomes* the robot.

**Some properties only work if they were compiled in.** MuJoCo's broadphase prunes whole bodies by
`body_contype` / `body_conaffinity`, which the compiler folds from the geoms once: a ball compiled non-colliding
stays pruned however its geom flags are later set, and sails through the robot. Gravity compensation is skipped
entirely when the compiler counted no body using it (`mjModel::ngravcomp == 0`): a ball compiled without it and
parked by setting `body_gravcomp` at runtime free-falls from its parking spot for hours, until its position
overflows and MuJoCo resets the whole simulation, teleporting the walking robot. So the ball is compiled colliding
and gravity-compensated, and parked and armed at runtime.

**A ball touching a foot is not that foot being planted.** `groundTruthContactMask` feeds the `RobotState` contact
flags and the `cheater_sim` estimator, and a ball against a *swing* foot would tell the MPC that foot is down. The
ZMP marker, "the zero moment point of the ground reaction", would be thrown meters from the feet by a 25 m/s impact.
Both take an `ignoreBodyId`, and the simulator and the marker pass the ball's.

**Joint damping must never reach the ball.** Twenty N s/m on its free joint — the zero-torque ragdoll damping the
simulator starts in — brings a thrown ball to a 0.2 m/s terminal speed within centimeters of where it was thrown.
All three writers (the start-up default, the ragdoll boost and `disableTorques`) go through `setRobotJointDamping`,
so the skip cannot drift apart between them again.

**A thrown ball must be put away.** With the usual condim-3 contact a ball rolling on a plane keeps 5/7 of its speed
forever. The ball has rolling friction, and `ProjectileRestMonitor` parks it once it has been slow for 0.25 s or in
play for ten seconds, whichever comes first. A throw is also canceled when the robot is caught: `lockGantry()`,
which both the fall catch and the operator's LOCK_GANTRY use, parks the ball and drops a scheduled impulse, and so
does `reset()`.

**The parking spot is above the scene, not below the floor.** The viewer renders the live model against a state
snapshot up to a frame old. A ball parked 50 m below a half-space floor, in the frame after a throw arms the model,
would be a contact of millions of newtons drawn as a kilometers-long force arrow. Parked 100 m up, nothing is in
reach, and it is beyond the far clipping plane of every camera on the robot.

---

## 5. No ball compiled in

If `sim_projectile` is empty the Throw button still works. The flight is ballistic and known in closed form, so the
simulator applies the momentum the ball would have *arrived* with, $m\,(\mathbf{v}_{\text{launch}} - g t \hat{\mathbf{z}})$
at the slider's (clamped) mass, as a one-step force on the base when it would have arrived. That is a lower bound on
the push: a real ball that bounces off hands over up to $(1 + e)$ times its momentum. The force lasts one physics
step and the viewer samples the state about every thirty, so it is usually not drawn.

---

## 6. Tests

| Test | What it pins |
| --- | --- |
| `remote_control/test/test_dodgeball.py` | the geometry: spawn offset; the launch speed equals the slider's, or the minimum that reaches; the path passes through the base; the direct root, not the lob; arrival speed by energy conservation; momentum = mass × arrival speed; NaN and out-of-range clamping of all five parameters; randomization leaving distance, speed and mass alone; the payload keys against the golden file the C++ test reads; the topic against the one the bridge subscribes to; and the tab — all five sliders holding their ranges on screen and on the wire, a non-number rejected, the status line actually showing, the preview saying when a speed was raised |
| `humanoid_common_mpc_app/robot/test/config/robot/testDodgeballThrowFromMessage.cpp` | the GUI's own throw (the golden `test/data/dodgeball_throw.textproto`) converts; every field the simulator uses is required and named when missing; non-finite values, a negative flight and a non-positive mass are rejected; an over-range mass is left for the simulator to clamp |
| `mujoco_sim_interface/test/testProjectile.cpp` | registry; the logarithmic-decrement round trip; the ball compiled last, colliding, gravity-compensated, with priority, condim 6 and rolling friction; a parked ball staying put; **the restitution against a floor with the shipped contact parameters at 0.5 and 1 ms**; no energy gain at three timesteps; a rolling ball coming to rest before its lifetime; the broadphase park-then-throw cycle; `setProjectileMass` matching a fresh compile field by field, leaving `<statistic>` alone, keeping the restitution mass-invariant, clamping, and refusing anything that is not a centered sphere on its own free joint; the damping helper; path-sliding out of the robot and out of the floor, staying on the path, and refusing the impossible; the fallback impulse; the rest monitor; the contact mask and the ground reaction ignoring the ball, each with a positive control |
| `mujoco_sim_interface/test/testMujocoSimInterfaceDodgeball.cpp` | the real simulator, headless, on the shipped Atlas scene: the ball found, parked and undamped at start-up and through torque toggles; a throw flying the real ball at the slider's mass to the base; a hand-published mass clamped; an under-floor spawn lifted; catching the robot canceling the throw while re-locking an already-locked gantry does not; a reset parking it; a finished ball parked again |

Every fix above was also checked the other way round: the bug was put back, and at least one of these tests failed.
