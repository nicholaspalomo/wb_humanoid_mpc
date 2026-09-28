"""****************************************************************************
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
****************************************************************************"""

"""The geometry of throwing a dodgeball at the robot, with no Tk and no ROS in sight.

Separated from the tab widget on purpose. The GUI tests in this package are headless and test the helpers rather than
the widgets, and everything here that could be wrong - an angle convention, a sign, the gravity compensation, the
wire format - is a pure function of its arguments and is covered by test/test_dodgeball.py.

It is also the ONLY place the throw is computed. The simulator receives the spawn offset and the launch velocity
already worked out, in the robot's own yaw frame, and does nothing but rotate them by the measured base yaw and add
the base position. Recomputing the trigonometry in C++ would be the same derivation written twice, in two languages,
with two chances to get the sign wrong.

ANGLE CONVENTIONS, which are the thing most easily misread:

* `azimuth_deg` is the angle about the world z axis, measured in the robot's OWN yaw frame, giving the direction the
  ball is thrown FROM. 0 deg is straight ahead of the robot, +90 deg is to its left, 180 deg is behind it. Measuring
  it in the robot's frame rather than the world's is what makes "hit it from the front" mean the same thing wherever
  the robot happens to be facing.
* `elevation_deg` is the angle out of the horizontal plane, again of the spawn point rather than of the velocity.
  0 deg spawns level with the base, +45 deg spawns above it and throws downwards, -20 deg spawns below it and throws
  upwards. The name is `elevation` rather than "the angle about the x-y axes" because a single angle out of the
  horizontal plane is what that describes once the azimuth has fixed which vertical plane we are in.
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

#: [kg] Mass of a regulation foam dodgeball, and where the mass slider starts.
#:
#: Mass is the property the impact scales with most directly: what the robot feels is the momentum `mass * speed`
#: arriving over the contact, so this and the speed slider are two ways of dialing the same thing - except that mass
#: also changes how the ball behaves AFTER the hit, because a heavy ball carries its momentum through the contact
#: while a light one gives it all back and bounces away.
#:
#: This must stay equal to the mass the C++ registry gives the ball it compiles into the scene. It is the default
#: rather than the only value - the slider overrides it per throw, and the simulator retunes the real ball's mass and
#: inertia to match - but a default that disagreed with the registry would make an untouched slider quietly change
#: the ball.
# LINT.IfChange(dodgeball_mass)
DEFAULT_BALL_MASS_KG = 0.45
# LINT.ThenChange(//robot_runtime/mujoco_sim_interface/src/Projectile.cpp:dodgeball_properties)

#: [m/s^2] Gravity the flight is computed under. The launch velocity is lifted so that the ball's BALLISTIC path
#: passes through the base, rather than pointing straight at it and landing low - see `launch_velocity`. The simulator
#: flies the ball, and times the fallback impulse, under the same value.
# LINT.IfChange(dodgeball_gravity)
GRAVITY = 9.81
# LINT.ThenChange(//robot_runtime/mujoco_sim_interface/src/MujocoSimInterface.cpp:dodgeball_gravity)

#: Inclusive slider ranges. They are here rather than in the tab so that the sampler and the tests agree with the
#: widgets by construction.
AZIMUTH_RANGE_DEG = (-180.0, 180.0)
ELEVATION_RANGE_DEG = (-30.0, 60.0)
DISTANCE_RANGE_M = (0.5, 8.0)
SPEED_RANGE_MPS = (1.0, 25.0)

#: [kg] The mass slider's range, from a beach ball to a medicine ball, with the regulation 0.45 kg near the bottom of
#: it. The top end is a deliberately unfair throw: 5 kg arriving at 25-28 m/s is 125-140 N s, roughly 0.8 m/s of base
#: velocity for the 160 kg Atlas before the bounce adds more - a shove that no balance controller should be expected to
#: absorb standing still, and therefore the interesting end of the sweep. The bottom end stays clear of zero because a
#: sphere's inertia is proportional to its mass, and MuJoCo cannot integrate a free body with zero inertia.
#:
#: The simulator clamps to the same range, because a topic can be published by hand.
# LINT.IfChange(dodgeball_mass_range)
MASS_RANGE_KG = (0.1, 5.0)
# LINT.ThenChange(//robot_runtime/mujoco_sim_interface/include/mujoco_sim_interface/Projectile.h:projectile_mass_range)


@dataclass(frozen=True)
class DodgeballThrow:
    """One throw, as the operator specifies it with the five sliders."""

    azimuth_deg: float
    elevation_deg: float
    distance_m: float
    speed_mps: float
    mass_kg: float = DEFAULT_BALL_MASS_KG

    def clamped(self) -> "DodgeballThrow":
        """The same throw with every parameter inside its slider range.

        The sliders cannot leave their range but the numeric entry boxes beside them can, and a distance or a speed of
        zero would divide by zero in the flight time below. Clamping here rather than validating means a typo produces
        a sensible throw instead of a traceback in a Tk callback, where it would be swallowed.
        """
        return DodgeballThrow(
            azimuth_deg=_clamp(self.azimuth_deg, *AZIMUTH_RANGE_DEG),
            elevation_deg=_clamp(self.elevation_deg, *ELEVATION_RANGE_DEG),
            distance_m=_clamp(self.distance_m, *DISTANCE_RANGE_M),
            speed_mps=_clamp(self.speed_mps, *SPEED_RANGE_MPS),
            mass_kg=_clamp(self.mass_kg, *MASS_RANGE_KG),
        )


def _clamp(value: float, low: float, high: float) -> float:
    # NaN fails every comparison and would otherwise pass straight through; it takes the low end, as -inf does.
    if math.isnan(value):
        return low
    return low if value < low else (high if value > high else value)


def spawn_offset(throw: DodgeballThrow) -> Tuple[float, float, float]:
    """Where the ball appears, as an offset from the robot's base in the base's yaw frame, in meters.

    Spherical: the azimuth sweeps the horizontal plane and the elevation lifts out of it, both of the SPAWN POINT.
    The distance is the straight-line distance to the base, so `norm(spawn_offset) == distance` exactly, whatever the
    two angles are - which is the property that makes the slider mean what it says.
    """
    throw = throw.clamped()
    azimuth = math.radians(throw.azimuth_deg)
    elevation = math.radians(throw.elevation_deg)
    horizontal = throw.distance_m * math.cos(elevation)
    return (
        horizontal * math.cos(azimuth),
        horizontal * math.sin(azimuth),
        throw.distance_m * math.sin(elevation),
    )


def minimum_launch_speed(throw: DodgeballThrow, gravity: float = GRAVITY) -> float:
    """[m/s] The slowest launch that can reach the base at all: sqrt(g (d + dz)).

    Here d is the distance and dz the height of the base above the spawn point. A ball launched any slower falls short
    whichever way it is aimed, so `launch_speed` raises the slider's speed to this. It is 5.4 m/s from 3 m level with
    the base and 8.9 m/s from 8 m, which is why the bottom of the speed slider only works for short or high throws.
    """
    throw = throw.clamped()
    rise = -spawn_offset(throw)[2]
    return math.sqrt(max(gravity * (throw.distance_m + rise), 0.0))


def launch_speed(throw: DodgeballThrow, gravity: float = GRAVITY) -> float:
    """[m/s] The speed the ball actually leaves at: the slider's, or the minimum that reaches the base if that is more."""
    throw = throw.clamped()
    return max(throw.speed_mps, minimum_launch_speed(throw, gravity))


def flight_time(throw: DodgeballThrow, gravity: float = GRAVITY) -> float:
    """[s] How long the ball is in the air on its way to the base, launched at `launch_speed`.

    The speed slider is the LAUNCH speed, as its label says. A ball launched at speed s from a point d away and dz
    below the base, aimed so that it arrives there after t seconds, leaves with v = (base - spawn) / t + (0, 0, g t/2)
    (see `launch_velocity`), and |v| = s gives a quadratic in t^2:

        (g^2 / 4) t^4 + (g dz - s^2) t^2 + d^2 = 0.

    Of its two roots the smaller is the direct throw and the larger a lob; this takes the direct one, written in the
    form that stays accurate as g goes to zero, where it reduces to d / s.
    """
    throw = throw.clamped()
    speed = launch_speed(throw, gravity)
    rise = -spawn_offset(throw)[2]
    distance = throw.distance_m
    b = speed * speed - gravity * rise
    discriminant = max(b * b - gravity * gravity * distance * distance, 0.0)
    return math.sqrt(2.0 * distance * distance / (b + math.sqrt(discriminant)))


def launch_velocity(
    throw: DodgeballThrow, gravity: float = GRAVITY
) -> Tuple[float, float, float]:
    """[m/s] The velocity the ball leaves at, in the base's yaw frame, so that it HITS the base.

    "Aimed at the robot's base" is taken to mean the ball arrives there, not merely that it sets off in that
    direction. Over a flight of `t` seconds gravity drops it by `g t^2 / 2`, so a velocity pointed straight down the
    line lands that far low. For the flight time `t` from `flight_time`,

        v = (base - spawn) / t + (0, 0, g t / 2)

    puts the ball at the base at time `t` under constant gravity, because the vertical term integrates to exactly the
    drop it cancels, and |v| is `launch_speed` by construction of `t`. Pass `gravity=0.0` to get the straight-line
    aim, which is what the tests use to check the geometry on its own.
    """
    throw = throw.clamped()
    offset_x, offset_y, offset_z = spawn_offset(throw)
    time_of_flight = flight_time(throw, gravity)
    return (
        -offset_x / time_of_flight,
        -offset_y / time_of_flight,
        -offset_z / time_of_flight + 0.5 * gravity * time_of_flight,
    )


def arrival_velocity(
    throw: DodgeballThrow, gravity: float = GRAVITY
) -> Tuple[float, float, float]:
    """[m/s] The velocity the ball reaches the base with: the launch velocity less what gravity took, `g t`."""
    velocity_x, velocity_y, velocity_z = launch_velocity(throw, gravity)
    return (
        velocity_x,
        velocity_y,
        velocity_z - gravity * flight_time(throw, gravity),
    )


def impact_momentum(throw: DodgeballThrow, gravity: float = GRAVITY) -> float:
    """[N s] The momentum the ball carries when it reaches the base: its mass times its ARRIVAL speed.

    Not mass times the slider's speed: a ball thrown from above arrives faster than it left and one thrown upwards
    slower, by energy conservation, v_arrival^2 = v_launch^2 + 2 g (spawn height - base height). This is what the ball
    brings to the collision. A ball that bounces off hands the robot more than this - up to (1 + e) times as much -
    and a real ball compiled into the scene does exactly that; the simulator applies precisely this momentum only when
    no ball is compiled in and the throw falls back to an impulse on the base.
    """
    throw = throw.clamped()
    return throw.mass_kg * math.hypot(*arrival_velocity(throw, gravity))


def sample_random_angles(rng: Optional[random.Random] = None) -> Tuple[float, float]:
    """A random azimuth and elevation, uniform over their slider ranges.

    Only the two ANGLES. The distance, the speed and the MASS are left to the operator deliberately: those three set
    how hard and how soon the hit lands, which is the part of the experiment being controlled, while the direction is
    the part worth randomizing to avoid unconsciously always throwing from the same side. Mass belongs with the speed
    rather than with the angles for exactly the reason the original request put the speed on the operator's side of
    that line: it is a magnitude being swept, not a condition being sampled.

    `rng` is injected so the tests can pin a seed; the tab passes nothing and gets the module-level generator.
    """
    generator = rng if rng is not None else random
    return (
        generator.uniform(*AZIMUTH_RANGE_DEG),
        generator.uniform(*ELEVATION_RANGE_DEG),
    )


def throw_payload(throw: DodgeballThrow, gravity: float = GRAVITY) -> Dict[str, object]:
    """The message the tab publishes, as the dict that is then dumped to YAML.

    It carries BOTH the operator's five parameters and what was derived from them. The simulator reads the mass, the
    spawn offset, the launch velocity and the flight time (humanoid_common_mpc_ros2's parseDodgeballThrow); the rest
    is there so that a recorded bag, or someone watching the topic with `ros2 topic echo`, says what was asked for as
    well as what was computed - including `launchSpeed`, which differs from `speed` when the slider's speed was too
    slow to reach the base and was raised.
    """
    throw = throw.clamped()
    offset = spawn_offset(throw)
    velocity = launch_velocity(throw, gravity)
    # LINT.IfChange(dodgeball_payload_keys)
    return {
        "dodgeball": {
            # What the operator set.
            "azimuthDeg": round(throw.azimuth_deg, 4),
            "elevationDeg": round(throw.elevation_deg, 4),
            "distance": round(throw.distance_m, 4),
            "speed": round(throw.speed_mps, 4),
            "mass": round(throw.mass_kg, 4),
            # What the simulator acts on, in the robot's own yaw frame: it rotates these by the measured base yaw and
            # adds the base position. With a ball compiled into the scene it flies the ball from there; without one,
            # it applies the arrival momentum to the base after `flightTime`.
            "spawnOffset": [round(value, 6) for value in offset],
            "launchVelocity": [round(value, 6) for value in velocity],
            "flightTime": round(flight_time(throw, gravity), 6),
            # Documentation only.
            "launchSpeed": round(launch_speed(throw, gravity), 6),
            "impactMomentum": round(impact_momentum(throw, gravity), 6),
        }
    }
    # LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_ros2/src/fsm/DodgeballThrowParser.cpp:dodgeball_payload_keys)
