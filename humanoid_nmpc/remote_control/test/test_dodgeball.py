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

"""Unit tests for the dodgeball throw geometry and for the tab that publishes it.

The geometry tests are headless by construction - `remote_control.tk_app.dodgeball` imports neither Tk nor ROS - and
are where the angle conventions, the gravity compensation and the wire format are actually pinned. The widget tests
build a real `Tk` root and withdraw it, which is what test_base_controller_contact_estimator.py already does, and
skip themselves where no display is available so the file is still runnable over ssh.
"""

import math
import random
import unittest

import yaml

from remote_control.tk_app.dodgeball import (
    AZIMUTH_RANGE_DEG,
    DEFAULT_BALL_MASS_KG,
    DISTANCE_RANGE_M,
    ELEVATION_RANGE_DEG,
    GRAVITY,
    MASS_RANGE_KG,
    SPEED_RANGE_MPS,
    DodgeballThrow,
    flight_time,
    impact_momentum,
    launch_velocity,
    sample_random_angles,
    spawn_offset,
    throw_payload,
)


class MockPublisher:
    """Stand-in for an rclpy Publisher that records what was published."""

    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)

    @property
    def last_data(self):
        return self.messages[-1].data if self.messages else None

    @property
    def publish_count(self):
        return len(self.messages)


def _norm(vector):
    return math.sqrt(sum(component * component for component in vector))


class TestSpawnOffset(unittest.TestCase):
    """Where the ball appears, relative to the base, in the robot's own yaw frame."""

    def test_distance_is_the_distance_whatever_the_angles(self):
        # The property that makes the slider mean what it says: the spawn point is on a sphere of exactly that
        # radius, so changing an angle never changes how far away the ball starts.
        for azimuth in (-180.0, -90.0, 0.0, 37.0, 180.0):
            for elevation in (-30.0, 0.0, 15.0, 60.0):
                for distance in (0.5, 3.0, 8.0):
                    throw = DodgeballThrow(azimuth, elevation, distance, 10.0)
                    self.assertAlmostEqual(
                        _norm(spawn_offset(throw)), distance, places=9
                    )

    def test_azimuth_zero_is_in_front_of_the_robot(self):
        # 0 deg must be the robot's +x, or "throw it at the front" means something different on every robot.
        offset = spawn_offset(DodgeballThrow(0.0, 0.0, 2.0, 10.0))
        self.assertAlmostEqual(offset[0], 2.0, places=9)
        self.assertAlmostEqual(offset[1], 0.0, places=9)
        self.assertAlmostEqual(offset[2], 0.0, places=9)

    def test_azimuth_ninety_is_to_the_robots_left(self):
        offset = spawn_offset(DodgeballThrow(90.0, 0.0, 2.0, 10.0))
        self.assertAlmostEqual(offset[0], 0.0, places=9)
        self.assertAlmostEqual(offset[1], 2.0, places=9)

    def test_azimuth_one_eighty_is_behind(self):
        offset = spawn_offset(DodgeballThrow(180.0, 0.0, 2.0, 10.0))
        self.assertAlmostEqual(offset[0], -2.0, places=9)

    def test_positive_elevation_spawns_above_the_base(self):
        # And therefore throws DOWNWARDS, which is the half of the convention most easily inverted.
        above = spawn_offset(DodgeballThrow(0.0, 30.0, 2.0, 10.0))
        below = spawn_offset(DodgeballThrow(0.0, -30.0, 2.0, 10.0))
        self.assertGreater(above[2], 0.0)
        self.assertLess(below[2], 0.0)
        self.assertAlmostEqual(above[2], -below[2], places=9)


class TestLaunchVelocity(unittest.TestCase):
    """The velocity the ball leaves at, which is the thing that has to actually hit."""

    def test_without_gravity_it_points_straight_back_at_the_base(self):
        throw = DodgeballThrow(53.0, 21.0, 4.0, 9.0)
        offset = spawn_offset(throw)
        velocity = launch_velocity(throw, gravity=0.0)
        # Anti-parallel to the offset: the ball travels from the spawn point to the base along the straight line.
        for axis in range(3):
            self.assertAlmostEqual(
                velocity[axis], -offset[axis] / flight_time(throw), places=9
            )
        self.assertAlmostEqual(_norm(velocity), throw.speed_mps, places=9)

    def test_the_ballistic_path_passes_through_the_base(self):
        # The whole point of the gravity term: integrate the flight and land on the base, not below it.
        for throw in (
            DodgeballThrow(0.0, 0.0, 3.0, 5.0),
            DodgeballThrow(-140.0, 45.0, 6.0, 12.0),
            DodgeballThrow(90.0, -20.0, 1.5, 3.0),
        ):
            spawn = spawn_offset(throw)
            velocity = launch_velocity(throw)
            time_of_flight = flight_time(throw)
            arrival = [
                spawn[0] + velocity[0] * time_of_flight,
                spawn[1] + velocity[1] * time_of_flight,
                spawn[2]
                + velocity[2] * time_of_flight
                - 0.5 * GRAVITY * time_of_flight * time_of_flight,
            ]
            for axis in range(3):
                self.assertAlmostEqual(
                    arrival[axis], 0.0, places=9, msg=f"{throw} missed on axis {axis}"
                )

    def test_gravity_compensation_only_lifts_it(self):
        # It must not steer: the horizontal components are the straight-line ones, so the ball still comes from the
        # direction the azimuth slider says.
        throw = DodgeballThrow(37.0, 10.0, 4.0, 8.0)
        straight = launch_velocity(throw, gravity=0.0)
        lifted = launch_velocity(throw)
        self.assertAlmostEqual(lifted[0], straight[0], places=9)
        self.assertAlmostEqual(lifted[1], straight[1], places=9)
        self.assertGreater(lifted[2], straight[2])

    def test_a_slow_long_throw_needs_more_lift_than_a_fast_short_one(self):
        slow = launch_velocity(DodgeballThrow(0.0, 0.0, 6.0, 2.0))
        fast = launch_velocity(DodgeballThrow(0.0, 0.0, 6.0, 20.0))
        self.assertGreater(slow[2], fast[2])


class TestFlightAndImpact(unittest.TestCase):
    def test_flight_time_is_distance_over_speed(self):
        self.assertAlmostEqual(
            flight_time(DodgeballThrow(0.0, 0.0, 6.0, 3.0)), 2.0, places=9
        )

    def test_impact_momentum_scales_with_mass_and_speed(self):
        self.assertAlmostEqual(
            impact_momentum(DodgeballThrow(0.0, 0.0, 3.0, 10.0, mass_kg=0.5)),
            5.0,
            places=9,
        )
        self.assertAlmostEqual(
            impact_momentum(DodgeballThrow(0.0, 0.0, 3.0, 20.0, mass_kg=0.5)),
            10.0,
            places=9,
        )


class TestClamping(unittest.TestCase):
    """The numeric entry boxes beside the sliders can be typed into, so out-of-range values do arrive."""

    def test_every_parameter_is_clamped_into_its_slider_range(self):
        throw = DodgeballThrow(1e6, -1e6, 1e6, -1e6, mass_kg=1e6).clamped()
        self.assertEqual(throw.azimuth_deg, AZIMUTH_RANGE_DEG[1])
        self.assertEqual(throw.elevation_deg, ELEVATION_RANGE_DEG[0])
        self.assertEqual(throw.distance_m, DISTANCE_RANGE_M[1])
        self.assertEqual(throw.speed_mps, SPEED_RANGE_MPS[0])
        self.assertEqual(throw.mass_kg, MASS_RANGE_KG[1])
        self.assertEqual(
            DodgeballThrow(0.0, 0.0, 3.0, 5.0, mass_kg=-2.0).clamped().mass_kg,
            MASS_RANGE_KG[0],
        )

    def test_a_zero_speed_cannot_divide_by_zero(self):
        # Without the clamp this is where a typed 0 would raise inside a Tk callback, where it would be swallowed.
        throw = DodgeballThrow(0.0, 0.0, 3.0, 0.0)
        self.assertGreater(flight_time(throw), 0.0)
        self.assertTrue(
            all(math.isfinite(component) for component in launch_velocity(throw))
        )

    def test_a_zero_mass_cannot_divide_by_zero(self):
        # The simulator divides by the mass to retune the ball's inertia, so the floor is a real kilogram figure and
        # not merely "positive".
        self.assertEqual(
            DodgeballThrow(0.0, 0.0, 3.0, 5.0, mass_kg=0.0).clamped().mass_kg,
            MASS_RANGE_KG[0],
        )
        self.assertGreater(MASS_RANGE_KG[0], 0.0)

    def test_the_regulation_ball_is_inside_the_mass_range(self):
        # Otherwise the slider's own default would be clamped away the first time it was thrown.
        self.assertGreaterEqual(DEFAULT_BALL_MASS_KG, MASS_RANGE_KG[0])
        self.assertLessEqual(DEFAULT_BALL_MASS_KG, MASS_RANGE_KG[1])
        self.assertLess(MASS_RANGE_KG[0], MASS_RANGE_KG[1])


class TestRandomSampling(unittest.TestCase):
    def test_only_the_two_angles_are_sampled_and_both_stay_in_range(self):
        rng = random.Random(20260921)
        for _ in range(200):
            azimuth, elevation = sample_random_angles(rng)
            self.assertGreaterEqual(azimuth, AZIMUTH_RANGE_DEG[0])
            self.assertLessEqual(azimuth, AZIMUTH_RANGE_DEG[1])
            self.assertGreaterEqual(elevation, ELEVATION_RANGE_DEG[0])
            self.assertLessEqual(elevation, ELEVATION_RANGE_DEG[1])

    def test_it_is_seedable_so_a_sweep_can_be_repeated(self):
        first = [sample_random_angles(random.Random(7)) for _ in range(3)]
        second = [sample_random_angles(random.Random(7)) for _ in range(3)]
        self.assertEqual(first, second)

    def test_it_actually_spreads_over_the_circle(self):
        # A sampler that always returned the same quadrant would pass every range check above and be useless.
        rng = random.Random(11)
        quadrants = {
            int((azimuth + 180.0) // 90.0)
            for azimuth, _ in (sample_random_angles(rng) for _ in range(300))
        }
        self.assertEqual(len(quadrants), 4)


class TestThrowPayload(unittest.TestCase):
    """The wire format. The C++ side (SimFsmBridge::dodgeballCallback) reads exactly these keys."""

    def test_it_carries_both_the_parameters_and_the_derived_vectors(self):
        throw = DodgeballThrow(45.0, 15.0, 3.0, 8.0)
        ball = throw_payload(throw)["dodgeball"]
        for key in (
            "azimuthDeg",
            "elevationDeg",
            "distance",
            "speed",
            "mass",
            "spawnOffset",
            "launchVelocity",
            "flightTime",
            "impactMomentum",
        ):
            self.assertIn(key, ball)
        self.assertEqual(len(ball["spawnOffset"]), 3)
        self.assertEqual(len(ball["launchVelocity"]), 3)

    def test_it_round_trips_through_yaml(self):
        # This is how it actually travels: yaml.dump in the tab, YAML::Load in SimFsmBridge.
        payload = throw_payload(DodgeballThrow(-73.0, 22.0, 5.5, 14.0))
        restored = yaml.safe_load(
            yaml.dump(payload, default_flow_style=False, sort_keys=False)
        )
        self.assertEqual(restored, payload)

    def test_the_derived_vectors_agree_with_the_functions(self):
        throw = DodgeballThrow(120.0, -10.0, 2.5, 6.0)
        ball = throw_payload(throw)["dodgeball"]
        for axis, value in enumerate(spawn_offset(throw)):
            self.assertAlmostEqual(ball["spawnOffset"][axis], value, places=5)
        for axis, value in enumerate(launch_velocity(throw)):
            self.assertAlmostEqual(ball["launchVelocity"][axis], value, places=5)

    def test_out_of_range_parameters_are_clamped_before_they_are_published(self):
        ball = throw_payload(DodgeballThrow(1e9, 1e9, 1e9, 1e9, mass_kg=1e9))[
            "dodgeball"
        ]
        self.assertEqual(ball["azimuthDeg"], AZIMUTH_RANGE_DEG[1])
        self.assertEqual(ball["distance"], DISTANCE_RANGE_M[1])
        self.assertEqual(ball["mass"], MASS_RANGE_KG[1])
        # And the momentum is computed from the CLAMPED mass, so the preview never promises a throw that is not sent.
        self.assertAlmostEqual(
            ball["impactMomentum"], MASS_RANGE_KG[1] * SPEED_RANGE_MPS[1], places=4
        )

    def test_the_mass_does_not_change_where_the_ball_goes(self):
        # Mass is the one parameter with no say in the flight: the launch is computed to put the ball on the base,
        # and a ballistic path does not depend on what is flying along it. Only the momentum it arrives with scales.
        light = throw_payload(DodgeballThrow(30.0, 20.0, 4.0, 9.0, mass_kg=0.2))[
            "dodgeball"
        ]
        heavy = throw_payload(DodgeballThrow(30.0, 20.0, 4.0, 9.0, mass_kg=4.0))[
            "dodgeball"
        ]
        for key in ("spawnOffset", "launchVelocity", "flightTime"):
            self.assertEqual(light[key], heavy[key], key)
        self.assertAlmostEqual(
            heavy["impactMomentum"] / light["impactMomentum"], 20.0, places=6
        )


def _tk_available():
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        root.destroy()
        return True
    except Exception:  # noqa: BLE001 - any failure here means no usable display
        return False


@unittest.skipUnless(_tk_available(), "no usable Tk display")
class TestDodgeballTab(unittest.TestCase):
    """The widget itself, built against a withdrawn Tk root and a mock publisher."""

    def setUp(self):
        import tkinter as tk

        from remote_control.tk_app.dodgeball_tab import DodgeballTab

        self.root = tk.Tk()
        self.root.withdraw()
        self.publisher = MockPublisher()
        self.tab = DodgeballTab(self.root, throw_publisher=self.publisher)

    def tearDown(self):
        self.root.destroy()

    def test_the_tab_has_the_five_sliders_the_checkbox_and_the_button(self):
        self.assertIsNotNone(self.tab.azimuth_row)
        self.assertIsNotNone(self.tab.elevation_row)
        self.assertIsNotNone(self.tab.distance_row)
        self.assertIsNotNone(self.tab.speed_row)
        self.assertIsNotNone(self.tab.mass_row)
        self.assertIsNotNone(self.tab.randomize_check)
        self.assertIsNotNone(self.tab.throw_button)

    def test_throwing_publishes_one_yaml_message_the_sim_can_read(self):
        self.tab.azimuth_row.set_value(90.0)
        self.tab.elevation_row.set_value(0.0)
        self.tab.distance_row.set_value(4.0)
        self.tab.speed_row.set_value(8.0)
        self.tab.throw()

        self.assertEqual(self.publisher.publish_count, 1)
        ball = yaml.safe_load(self.publisher.last_data)["dodgeball"]
        self.assertAlmostEqual(ball["azimuthDeg"], 90.0, places=3)
        self.assertAlmostEqual(ball["distance"], 4.0, places=3)
        self.assertAlmostEqual(ball["flightTime"], 0.5, places=3)
        # 90 deg is the robot's left, so the ball spawns at +y and flies in -y.
        self.assertGreater(ball["spawnOffset"][1], 3.9)
        self.assertLess(ball["launchVelocity"][1], 0.0)

    def test_the_button_publishes_once_per_press(self):
        self.tab.throw()
        self.tab.throw()
        self.assertEqual(self.publisher.publish_count, 2)

    def test_the_checkbox_randomises_the_angles_but_not_distance_speed_or_mass(self):
        self.tab.distance_row.set_value(5.0)
        self.tab.speed_row.set_value(12.0)
        self.tab.mass_row.set_value(1.5)
        self.tab.randomize_var.set(True)
        self.tab._on_randomize_toggle()

        azimuths = set()
        for _ in range(25):
            ball = self.tab.throw()["dodgeball"]
            azimuths.add(round(ball["azimuthDeg"], 6))
            # The three the operator owns must survive every throw untouched.
            self.assertAlmostEqual(ball["distance"], 5.0, places=6)
            self.assertAlmostEqual(ball["speed"], 12.0, places=6)
            self.assertAlmostEqual(ball["mass"], 1.5, places=6)
        self.assertGreater(
            len(azimuths), 1, "the checkbox did not randomise the azimuth"
        )

    def test_randomising_writes_the_sampled_angles_back_onto_the_sliders(self):
        # So that the screen records what was thrown, rather than whatever the sliders were last left at.
        self.tab.randomize_var.set(True)
        self.tab._on_randomize_toggle()
        ball = self.tab.throw()["dodgeball"]
        self.assertAlmostEqual(
            self.tab.azimuth_row.get_value(), ball["azimuthDeg"], places=3
        )
        self.assertAlmostEqual(
            self.tab.elevation_row.get_value(), ball["elevationDeg"], places=3
        )

    def test_the_angle_sliders_are_disabled_while_randomising(self):
        self.tab.randomize_var.set(True)
        self.tab._on_randomize_toggle()
        self.assertEqual(str(self.tab.azimuth_row.scale.cget("state")), "disabled")
        self.tab.randomize_var.set(False)
        self.tab._on_randomize_toggle()
        self.assertNotEqual(str(self.tab.azimuth_row.scale.cget("state")), "disabled")

    def test_with_no_publisher_it_still_computes_the_throw_and_does_not_raise(self):
        from remote_control.tk_app.dodgeball_tab import DodgeballTab

        tab = DodgeballTab(self.root, throw_publisher=None)
        payload = tab.throw()
        self.assertIn("dodgeball", payload)

    def test_the_topic_matches_the_one_the_gui_publishes_on(self):
        from remote_control.tk_app.dodgeball_tab import DodgeballTab

        self.assertEqual(DodgeballTab.TOPIC_NAME, "/humanoid/dodgeball_throw")

    def test_the_default_ball_is_a_dodgeball_and_not_a_cannonball(self):
        self.assertAlmostEqual(
            self.tab.current_throw().mass_kg, DEFAULT_BALL_MASS_KG, places=6
        )
        self.assertAlmostEqual(
            self.tab.mass_row.get_value(), DEFAULT_BALL_MASS_KG, places=6
        )

    def test_the_mass_slider_spans_the_mass_range(self):
        self.assertAlmostEqual(
            float(self.tab.mass_row.scale.cget("from")), MASS_RANGE_KG[0], places=6
        )
        self.assertAlmostEqual(
            float(self.tab.mass_row.scale.cget("to")), MASS_RANGE_KG[1], places=6
        )

    def test_the_mass_slider_reaches_the_published_payload(self):
        # The whole point of the slider: the number on screen is the number the simulator is told to throw.
        self.tab.mass_row.set_value(2.25)
        self.tab.speed_row.set_value(10.0)
        self.tab.throw()
        ball = yaml.safe_load(self.publisher.last_data)["dodgeball"]
        self.assertAlmostEqual(ball["mass"], 2.25, places=4)
        self.assertAlmostEqual(ball["impactMomentum"], 22.5, places=4)

    def test_the_mass_slider_stays_live_while_the_direction_is_randomised(self):
        # The checkbox greys out the two angles it drives, and nothing else: a disabled mass slider would look like
        # the mass was being randomised too.
        self.tab.randomize_var.set(True)
        self.tab._on_randomize_toggle()
        self.assertNotEqual(str(self.tab.mass_row.scale.cget("state")), "disabled")
        self.assertNotEqual(str(self.tab.speed_row.scale.cget("state")), "disabled")

    def test_the_preview_shows_the_mass_and_the_momentum_it_implies(self):
        self.tab.mass_row.set_value(3.0)
        self.tab.speed_row.set_value(8.0)
        self.tab._update_preview()
        preview = self.tab.preview_var.get()
        self.assertIn("3.00 kg", preview)
        self.assertIn("24.00 N s", preview)


if __name__ == "__main__":
    unittest.main()
