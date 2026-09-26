# Dodgeball: throwing a physical ball at the robot

A push test you can aim. The GUI's **Dodgeball** tab throws a dodgeball at the robot's base from an angle, a
distance and a speed you choose, at a mass you choose; the simulator compiles a real sphere into the MuJoCo scene,
flies it, and lets the impact, the bounce and the recovery all be physics rather than a model of physics.

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
  │   ├─ azimuth      slider  ──┐     ├─ spawn_offset(throw)      -> where it starts, in the │
  │   ├─ elevation    slider  ──┤     │                              robot's own yaw frame   │
  │   ├─ distance     slider  ──┼──►  ├─ flight_time(throw)       -> distance / speed        │
  │   ├─ speed        slider  ──┤     ├─ launch_velocity(throw)   -> aim, lifted by g t / 2  │
  │   ├─ mass         slider  ──┤     ├─ impact_momentum(throw)   -> what it will deliver    │
  │   ├─ randomize    checkbox──┘     │                                                      │
  │   └─ THROW button ────────────►   └─ throw_payload(throw)     -> the YAML below          │
  └───────────────────────────────────────────┬──────────────────────────────────────────────┘
                                              │  std_msgs/String, /humanoid/dodgeball_throw
                                              │  (RELIABLE, depth 10 - a throw is an event,
                                              │   not a stream; none of them may be dropped)
                                              ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  humanoid_common_mpc_ros2 :: SimFsmBridge                                                │
  │                                                                                          │
  │   dodgeballCallback()  parses the YAML, stashes it under dodgeballMutex_                 │
  │   processCommands()    drains pendingDodgeball_ FIRST, then the FSM commands             │
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
  │                                 then resolve dodgeballBodyId_ / QposAdr_ / DofAdr_       │
  │                                 and parkProjectile()                                     │
  │                                                                                          │
  │   ── every simulation step ───────────────────────────────────────────────────────────   │
  │   applyDodgeball()                                                                       │
  │     ├─ a throw is staged?  mass changed? setProjectileMass()  (then, every throw:)       │
  │     │                      place at base + R(yaw) offset, set qvel, setProjectileArmed() │
  │     │                      (no ball compiled in? schedule the equivalent base impulse)   │
  │     └─ armed and at rest?  parkProjectile()                                              │
  │                                                                                          │
  │   groundTruthContactMask(..., ignoreBodyId = dodgeballBodyId_)                           │
  │        └─► the ball is never reported to the controller as ground contact                │
  └───────────────────────────────────────────┬──────────────────────────────────────────────┘
                                              ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  mujoco_sim_interface :: Projectile          (the ball itself - a value type, no state)  │
  │                                                                                          │
  │   availableProjectiles()                 the names simProjectile accepts                 │
  │   projectileFromName(name)               -> Projectile, or an error listing the valid    │
  │   contactDampRatioForRestitution(e)      restitution -> MuJoCo's solref damping ratio    │
  │   addProjectileToSpec(spec, ball, name)  appends the body, LAST, colliding, parked       │
  │   setProjectileCollisionEnabled(m, b, on)  geom flags AND the body-level aggregates      │
  │   setProjectileMass(m, b, mass, r)       mass, inertia, mj_setConst on a scratch mjData  │
  └──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Choosing the ball

```yaml
# robot_models/<robot>/<robot>_centroidal_mpc/config/mpc/task.yaml
simProjectile: dodgeball
```

Selected **by name**, like every other pluggable component in this repository, so a second projectile is a new entry
in `availableProjectiles()` rather than a second boolean. Remove the key, or leave it empty, and the scene compiles
exactly as it is on disk — the code's own default, so a robot that has not opted in is bit-for-bit unchanged.

| | `dodgeball` |
| --- | --- |
| diameter | 0.216 m — a regulation 8.5-inch ball |
| mass | 0.45 kg by default; the **Ball mass** slider sets 0.1 – 5.0 kg per throw |
| restitution | 0.80 |
| sliding friction | 0.60 |
| contact time constant | 0.002 s — a foam ball flattening, not a steel bearing |

At the default mass and the GUI's top speed of 25 m/s that is 11.3 kg·m/s of momentum, delivered in roughly the
contact time constant.
For scale, Atlas at 159.7 kg takes about 0.07 m/s of base velocity from a square hit, so the top of the speed
slider is a shove rather than a demolition; what makes a throw interesting is *where* and *when* it lands. At the
top of the mass slider, 5 kg at 25 m/s is 125 kg·m/s and about 0.8 m/s for Atlas: a medicine ball, and deliberately
more than a balance controller should be expected to absorb standing still.

### Changing the mass after the scene is compiled

The size, the friction and the restitution are fixed when the scene is compiled; the mass is not, because it is the
one property worth sweeping. Mass and speed both scale the momentum, but they are not interchangeable: a heavy ball
carries its momentum *through* the contact while a light one gives it back and bounces away, so the two produce
different pushes for the same $m v$.

MuJoCo has no API for changing a body's mass, and assigning `body_mass` is not enough. `setProjectileMass` does
three things, and the third is the one that is easy to miss:

1. `body_mass`, the mass itself;
2. `body_inertia`, a solid sphere's $\tfrac{2}{5} m r^{2}$ about every axis;
3. `mj_setConst`, which recomputes `body_invweight0` — an inverse mass the compiler worked out once, and by which
   MuJoCo normalises every contact's reference acceleration.

Skip the third and the ball does not behave like its old mass, nor like its new one: the *contact* changes, and the
restitution starts depending on the mass. Measured on the same drop, a refreshed ball rebounds to the same height
to four decimal places at 0.45, 2.0 and 5.0 kg — correct, since restitution belongs to the contact — while a stale
one loses more than half its rebound and wanders non-monotonically with the mass.

And `mj_setConst` has a trap of its own: it uses the `mjData` it is handed as scratch space and leaves its `qpos`
overwritten — by thousands of radians, on the Atlas scene. Handed the live simulation state, it would reset the
walking robot mid-stride. So `setProjectileMass` takes no `mjData` at all; it builds a throwaway one, uses it and
frees it, which makes the mistake impossible for a later caller rather than merely documented. That costs about
0.4 ms on Atlas — some twenty simulation steps — so the simulator only calls it when the requested mass differs from
the one the ball already has.

### Restitution, and why MuJoCo does not have it

MuJoCo models a contact as a spring-damper and exposes `solref = (timeconst, dampratio)`. A damping ratio below one
is what makes a contact bounce, and the two are related by the logarithmic decrement of a damped oscillator:

$$e \;=\; \exp\!\left(\frac{-\pi \zeta}{\sqrt{1 - \zeta^{2}}}\right) \qquad \Longleftrightarrow \qquad \zeta \;=\; \frac{-\ln e}{\sqrt{\pi^{2} + \ln^{2} e}}.$$

So the registry names the number an engineer actually knows — a rubber-skinned ball comes back at about four fifths
of the speed it arrived at — and `contactDampRatioForRestitution` converts it: $e = 0.80$ gives $\zeta = 0.0708$.
It is clamped into $(0, 1)$, because $e = 1$ is a contact that never settles and $e = 0$ takes the log of zero.

---

## 3. The geometry of a throw

All of it is computed in `remote_control/tk_app/dodgeball.py`, in the robot's own yaw frame, so that "behind the
robot" means behind the robot however it is facing. The simulator only rotates the result into the world.

Given an azimuth $\psi$ about $z$, an elevation $\theta$ in the vertical plane, a distance $d$ and a speed $v$, the
ball starts at

$$\mathbf{p}_{\text{spawn}} \;=\; \mathbf{p}_{\text{base}} \;+\; d \begin{pmatrix} \cos\theta \cos\psi \\ \cos\theta \sin\psi \\ \sin\theta \end{pmatrix},$$

flies for $t = d / v$, and is launched with

$$\mathbf{v}_{\text{launch}} \;=\; \frac{\mathbf{p}_{\text{base}} - \mathbf{p}_{\text{spawn}}}{t} \;+\; \begin{pmatrix} 0 \\ 0 \\ \tfrac{1}{2} g t\end{pmatrix}.$$

The second term is the gravity compensation that makes the aim mean something: a ball fired straight at the base
would arrive $\tfrac{1}{2} g t^{2}$ low, so the launch is lifted by exactly the average of the velocity gravity will
steal. **The ball hits where you aimed it**, and its arrival velocity is $\mathbf{v}_{\text{launch}} - (0, 0, g t)$.

The base pose is read at the moment of the throw, in the simulation thread — not when the button was pressed — so a
robot that turns during the flight does not drag the ball around with it.

The **Randomize** checkbox samples the azimuth and the elevation only. Distance, speed and mass stay where you put
them, because those are the magnitudes you sweep when you are looking for the limit; the direction is the part you
want unbiased. The mass has no effect on the flight itself — the launch is computed to put the ball on the base, and a
ballistic path does not depend on what is flying along it — only on what arrives.

---

## 4. Four things that would have been silent bugs

Worth stating, because each one fails in a way that produces no error anywhere and a plausible-looking simulation.

**The ball must be the last body.** Several things here take the *first* body carrying a free joint to be the robot:
`robotCentroidalState`'s centre of mass, the ZMP and DCM markers, the viewer's tracking camera, and every read of
`qpos[3..6]` as the base quaternion. A ball inserted ahead of the robot quietly *becomes* the robot.
`addProjectileToSpec` appends, and `testProjectile.cpp` pins it.

**The ball must be compiled colliding.** MuJoCo's broadphase prunes whole bodies by `mjModel::body_contype` and
`body_conaffinity`, which the compiler folds from the geoms *once*, at compile time. A ball compiled with
`contype = 0` therefore has a zero body aggregate for the rest of the session: restoring the geom flags on the throw
does not bring it back, the broadphase goes on skipping it, and the ball sails straight through the robot. So the
ball is compiled colliding and parked at runtime through `setProjectileCollisionEnabled`, which sets both levels.

**The mass-derived constants have to be refreshed, and not on the live state.** Section 2 has the detail: a stale
`body_invweight0` makes restitution depend on mass, and the function that refreshes it overwrites the `qpos` of the
`mjData` it is given. `setProjectileMass` owns both halves, and a restitution test drops three different masses and
requires the same rebound.

**A ball touching a foot is not that foot being planted.** `groundTruthContactMask` feeds the `RobotState` contact
flags and the `cheater_sim` estimator. A ball resting against a *swing* foot would otherwise tell the MPC that foot
is down — the disturbance would corrupt the state estimate as well as the dynamics, and the two failures would be
indistinguishable in the logs. The mask takes an `ignoreBodyId` and the simulator passes the ball's.

Alongside those: the ball's six dofs are skipped by all three joint-damping loops (20 N·s/m of ragdoll damping on a
thrown ball makes it fall through syrup), it is parked again once it comes to rest so the robot cannot trip over it,
and `reset()` puts it back in its box — a reset is how the gantry catches a fallen robot, and a ball still bouncing
around the freshly caught robot is a disturbance nobody threw.

---

## 5. No ball compiled in

If `simProjectile` is empty the Throw button still works. The flight is ballistic and known in closed form, so the
simulator schedules the momentum the ball *would* have delivered — at the slider's mass — as a one-step force on the
base at the moment of arrival. Nothing to look at, but the same disturbance — which is what the feature did before it had a body, and what
it falls back to if a scene somehow compiles without one.

---

## 6. Tests

| Test | What it pins |
| --- | --- |
| `remote_control/test/test_dodgeball.py` | the geometry: spawn offset, flight time, the gravity-compensated launch, that the ball arrives at the base, impact momentum, the mass leaving the flight untouched, clamping of all five parameters, randomization leaving distance, speed and mass alone, the payload round trip, and the tab's widgets — including the mass slider's range and default, its value reaching the published payload, and staying live while the direction is randomised |
| `mujoco_sim_interface/test/testProjectile.cpp` — `ProjectileRegistry` | a regulation ball; an unknown name erroring and listing the valid ones |
| — `ContactDampRatio` | the round trip $\zeta \rightarrow e$ against the logarithmic decrement itself, monotonicity, and the clamped ends |
| — `AddProjectileToSpec` | mass and radius as configured (not MuJoCo's 1000 kg/m³ default, which would make this a 5.3 kg medicine ball), the ball appended **last**, the robot still the first free body with its `qpos` address unmoved |
| — `SetProjectileCollisionEnabled` | both levels flipped, the robot untouched, and the park-then-throw cycle **still generating contacts** — the broadphase regression |
| — `SetProjectileMass` | mass and $\tfrac{2}{5} m r^{2}$ inertia; `body_invweight0` refreshed with the robot's untouched; **the same rebound height at three masses**; clamping to the slider range; and rejecting any body that is not one free joint and one sphere — the world, the robot's base, a link — without changing it. Both the refresh tests were checked to fail with the `mj_setConst` call removed, and the rejection test to fail with the identity check weakened |
| — `GroundTruthContactMask` | a ball striking a foot setting no contact bit, *with a positive control* proving the contact was really there; and a foot on the floor still reading as planted while a ball is armed |
