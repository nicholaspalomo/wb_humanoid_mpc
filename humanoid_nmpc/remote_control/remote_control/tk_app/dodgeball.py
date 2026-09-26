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
#: arriving over the contact, so this and the speed slider are two ways of dialling the same thing - except that mass
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
#: passes through the base, rather than pointing straight at it and landing low - see `launch_velocity`.
GRAVITY = 9.81

#: Inclusive slider ranges. They are here rather than in the tab so that the sampler and the tests agree with the
#: widgets by construction.
AZIMUTH_RANGE_DEG = (-180.0, 180.0)
ELEVATION_RANGE_DEG = (-30.0, 60.0)
DISTANCE_RANGE_M = (0.5, 8.0)
SPEED_RANGE_MPS = (1.0, 25.0)

#: [kg] The mass slider's range, from a beach ball to a medicine ball, with the regulation 0.45 kg near the bottom of
#: it. The top end is a deliberately unfair throw: 5 kg at the top speed is 125 kg m/s, which is about 0.8 m/s of base
#: velocity for a robot the mass of Atlas - a shove that no balance controller should be expected to absorb standing
#: still, and therefore the interesting end of the sweep. The bottom end stays clear of zero because the simulator
#: divides by the mass to retune the ball's inertia.
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
    return low if value < low else (high if value > high else value)


def spawn_offset(throw: DodgeballThrow) -> Tuple[float, float, float]:
    """Where the ball appears, as an offset from the robot's base in the base's yaw frame, in metres.

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


def flight_time(throw: DodgeballThrow) -> float:
    """[s] How long the ball is in the air: the straight-line distance over the commanded speed.

    The commanded speed is therefore the CLOSING speed along the line to the base, not quite the launch speed - the
    lift that `launch_velocity` adds to beat gravity makes the launch marginally faster. Defining it this way is what
    lets the flight time be exactly `distance / speed`, so the operator can predict when the ball lands from the two
    sliders alone.
    """
    throw = throw.clamped()
    return throw.distance_m / throw.speed_mps


def launch_velocity(
    throw: DodgeballThrow, gravity: float = GRAVITY
) -> Tuple[float, float, float]:
    """[m/s] The velocity the ball leaves at, in the base's yaw frame, so that it HITS the base.

    "Aimed at the robot's base" is taken to mean the ball arrives there, not merely that it sets off in that
    direction. Over a flight of `t` seconds gravity drops it by `g t^2 / 2`, so a velocity pointed straight down the
    line lands that far low - half a metre at 3 m and 5 m/s, which on these robots is the difference between the
    chest and the floor. The correction is exact rather than iterative: for a straight-line flight time `t`,

        v = (base - spawn) / t + (0, 0, g t / 2)

    puts the ball at the base at time `t` under constant gravity, because the vertical term integrates to exactly the
    drop it cancels. Pass `gravity=0.0` to get the straight-line aim, which is what the tests use to check the
    geometry on its own.
    """
    throw = throw.clamped()
    offset_x, offset_y, offset_z = spawn_offset(throw)
    time_of_flight = flight_time(throw)
    return (
        -offset_x / time_of_flight,
        -offset_y / time_of_flight,
        -offset_z / time_of_flight + 0.5 * gravity * time_of_flight,
    )


def impact_momentum(throw: DodgeballThrow) -> float:
    """[N s] The momentum the ball carries along its line of travel, which is what the robot feels.

    The simulator delivers this as an impulse on the base, so it is the one number that sets how hard the hit is. A
    0.45 kg ball at 15 m/s carries 6.75 N s - enough to shift a 33 kg SA01 by 0.2 m/s and barely to nudge a 160 kg
    Atlas, which is the honest asymmetry between the two robots rather than anything the GUI should hide.
    """
    throw = throw.clamped()
    return throw.mass_kg * throw.speed_mps


def sample_random_angles(rng: Optional[random.Random] = None) -> Tuple[float, float]:
    """A random azimuth and elevation, uniform over their slider ranges.

    Only the two ANGLES. The distance, the speed and the MASS are left to the operator deliberately: those three set
    how hard and how soon the hit lands, which is the part of the experiment being controlled, while the direction is
    the part worth randomising to avoid unconsciously always throwing from the same side. Mass belongs with the speed
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

    It carries BOTH the operator's five parameters and the derived spawn offset, launch velocity and flight time. The
    derived values are what the simulator acts on; the parameters are there so that a recorded bag, or someone
    watching the topic with `ros2 topic echo`, says what was asked for rather than only what was computed.
    """
    throw = throw.clamped()
    offset = spawn_offset(throw)
    velocity = launch_velocity(throw, gravity)
    return {
        "dodgeball": {
            # What the operator set.
            "azimuthDeg": round(throw.azimuth_deg, 4),
            "elevationDeg": round(throw.elevation_deg, 4),
            "distance": round(throw.distance_m, 4),
            "speed": round(throw.speed_mps, 4),
            "mass": round(throw.mass_kg, 4),
            # What the simulator acts on, in the robot's own yaw frame: it rotates these by the measured base yaw and
            # adds the base position, and applies the impulse after `flightTime`.
            "spawnOffset": [round(value, 6) for value in offset],
            "launchVelocity": [round(value, 6) for value in velocity],
            "flightTime": round(flight_time(throw), 6),
            "impactMomentum": round(impact_momentum(throw), 6),
        }
    }
