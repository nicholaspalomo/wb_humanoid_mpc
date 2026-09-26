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

"""The Dodgeball tab: throw a ball at the robot and watch what the controller does about it."""

import tkinter as tk
from tkinter import ttk
from typing import Optional

import yaml

from remote_control.tk_app.dodgeball import (
    AZIMUTH_RANGE_DEG,
    DEFAULT_BALL_MASS_KG,
    DISTANCE_RANGE_M,
    ELEVATION_RANGE_DEG,
    MASS_RANGE_KG,
    SPEED_RANGE_MPS,
    DodgeballThrow,
    flight_time,
    impact_momentum,
    sample_random_angles,
    throw_payload,
)
from remote_control.tk_app.slider_row import SliderRow


class DodgeballTab(ttk.Frame):
    """Throws a dodgeball at the robot's base, for testing push recovery without leaving the GUI.

    A disturbance is the one thing a locomotion controller is hardest to evaluate without, and until now the only way
    to push this robot in simulation was to reach into MuJoCo by hand. The five sliders say where the ball comes from,
    how fast and how heavy it is, the checkbox randomises the direction so that a sweep is not unconsciously always
    thrown from the same side, and the button throws it. The ball is always aimed at the base - there is no aiming error to tune,
    because a disturbance the operator can miss with is a disturbance that cannot be repeated.

    ALL OF THE GEOMETRY IS IN `dodgeball.py`, not here. This class reads the widgets, hands the five numbers to that
    module and publishes what comes back; everything that could be wrong about an angle convention or the gravity
    compensation is a pure function over there, and is unit-tested in test/test_dodgeball.py. What is left here is
    Tk, which the tests in this package do not exercise.
    """

    # LINT.IfChange(dodgeball_topic_name)
    TOPIC_NAME = "/humanoid/dodgeball_throw"
    # LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_ros2/include/humanoid_common_mpc_ros2/fsm/SimFsmBridge.h:dodgeball_topic_name)

    def __init__(
        self,
        parent,
        throw_publisher=None,
        *args,
        **kwargs,
    ):
        super().__init__(parent, *args, **kwargs)
        self.configure(style="TFrame")

        self.throw_publisher = throw_publisher
        self.status_var = tk.StringVar(value="")

        header = ttk.Frame(self)
        header.pack(fill="x", padx=10, pady=(10, 4))
        ttk.Label(
            header,
            text="🏐 Dodgeball",
            font=("Helvetica", 12, "bold"),
        ).pack(side="left")
        ttk.Label(
            header,
            text="Throws a ball at the robot's base. Simulator only.",
            font=("Helvetica", 9),
        ).pack(side="left", padx=(10, 0))

        body = ttk.Frame(self)
        body.pack(fill="x", padx=10, pady=(4, 4))

        # The two ANGLES, which the randomise checkbox drives, and the three magnitudes the operator always owns. Kept
        # in that order so that the pair the checkbox disables is contiguous on screen.
        self.azimuth_row = SliderRow(
            body,
            name="Azimuth (about z)",
            initial_value=0.0,
            min_val=AZIMUTH_RANGE_DEG[0],
            max_val=AZIMUTH_RANGE_DEG[1],
            unit="deg",
            on_change=self._on_slider_change,
            label_width=22,
        )
        self.azimuth_row.pack(fill="x", pady=2)

        self.elevation_row = SliderRow(
            body,
            name="Elevation (about x-y)",
            initial_value=10.0,
            min_val=ELEVATION_RANGE_DEG[0],
            max_val=ELEVATION_RANGE_DEG[1],
            unit="deg",
            on_change=self._on_slider_change,
            label_width=22,
        )
        self.elevation_row.pack(fill="x", pady=2)

        self.distance_row = SliderRow(
            body,
            name="Spawn distance",
            initial_value=3.0,
            min_val=DISTANCE_RANGE_M[0],
            max_val=DISTANCE_RANGE_M[1],
            unit="m",
            on_change=self._on_slider_change,
            label_width=22,
        )
        self.distance_row.pack(fill="x", pady=2)

        self.speed_row = SliderRow(
            body,
            name="Start velocity",
            initial_value=8.0,
            min_val=SPEED_RANGE_MPS[0],
            max_val=SPEED_RANGE_MPS[1],
            unit="m/s",
            on_change=self._on_slider_change,
            label_width=22,
        )
        self.speed_row.pack(fill="x", pady=2)

        self.mass_row = SliderRow(
            body,
            name="Ball mass",
            initial_value=DEFAULT_BALL_MASS_KG,
            min_val=MASS_RANGE_KG[0],
            max_val=MASS_RANGE_KG[1],
            unit="kg",
            on_change=self._on_slider_change,
            label_width=22,
        )
        self.mass_row.pack(fill="x", pady=2)

        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=10, pady=(8, 4))

        # Randomises the two ANGLES only. The distance, the speed and the mass stay where the operator put them
        # because those three are what makes one throw comparable with the next: randomising them as well would turn
        # a sweep into anecdotes.
        self.randomize_var = tk.BooleanVar(value=False)
        self.randomize_check = ttk.Checkbutton(
            controls,
            text="Randomise direction on every throw (azimuth and elevation)",
            variable=self.randomize_var,
            command=self._on_randomize_toggle,
        )
        self.randomize_check.pack(side="left")

        self.throw_button = ttk.Button(
            controls,
            text="🏐 Throw dodgeball",
            command=self.throw,
        )
        self.throw_button.pack(side="right")

        # What the current sliders would actually do, updated as they move: the flight time and the momentum are the
        # two numbers that decide whether a throw is a nudge or a knockdown, and neither is obvious from the sliders.
        self.preview_var = tk.StringVar(value="")
        ttk.Label(
            self,
            textvariable=self.preview_var,
            font=("Courier", 9),
        ).pack(fill="x", padx=10, pady=(0, 2))

        self.status_label = ttk.Label(
            self,
            textvariable=self.status_var,
            font=("Helvetica", 9),
        )
        self.status_label.pack(fill="x", padx=10, pady=(0, 10))

        self._update_preview()

    # ---------------------------------------------------------------------------------------------------------
    # Reading the widgets
    # ---------------------------------------------------------------------------------------------------------

    def current_throw(self) -> DodgeballThrow:
        """The throw the five sliders currently describe, clamped into range."""
        return DodgeballThrow(
            azimuth_deg=self.azimuth_row.get_value(),
            elevation_deg=self.elevation_row.get_value(),
            distance_m=self.distance_row.get_value(),
            speed_mps=self.speed_row.get_value(),
            mass_kg=self.mass_row.get_value(),
        ).clamped()

    def _on_slider_change(self, name: str, value: float) -> None:
        self._update_preview()

    def _on_randomize_toggle(self) -> None:
        """Greys the two angle sliders out while they are being randomised, so the display cannot lie.

        Without this the sliders would keep showing whatever they were last left at while every throw came from
        somewhere else, which is the kind of small dishonesty that costs an afternoon.
        """
        state = "disabled" if self.randomize_var.get() else "normal"
        self.azimuth_row.set_state(state)
        self.elevation_row.set_state(state)
        self._update_preview()

    def _update_preview(self) -> None:
        throw = self.current_throw()
        direction = (
            "randomised each throw"
            if self.randomize_var.get()
            else "azimuth %+.0f deg, elevation %+.0f deg"
            % (throw.azimuth_deg, throw.elevation_deg)
        )
        self.preview_var.set(
            "%s  |  %.2f s of flight  |  %.2f kg x %.1f m/s = %.2f N s of momentum"
            % (
                direction,
                flight_time(throw),
                throw.mass_kg,
                throw.speed_mps,
                impact_momentum(throw),
            )
        )

    # ---------------------------------------------------------------------------------------------------------
    # Throwing
    # ---------------------------------------------------------------------------------------------------------

    def throw(self) -> Optional[dict]:
        """Throws one ball, and returns the payload that was published (or would have been).

        Returning it is what lets a test drive the button and assert on the result without a ROS graph; the GUI
        itself ignores the return value.
        """
        if self.randomize_var.get():
            azimuth, elevation = sample_random_angles()
            # Written back to the sliders even while they are disabled, so that the record of what was thrown is on
            # screen rather than only in the topic.
            self.azimuth_row.set_value(azimuth)
            self.elevation_row.set_value(elevation)

        throw = self.current_throw()
        payload = throw_payload(throw)

        if self.throw_publisher is None:
            self._show_status(
                "No publisher: the throw was computed but not sent.", error=True
            )
            return payload

        try:
            from std_msgs.msg import String

            msg = String()
            msg.data = yaml.dump(payload, default_flow_style=False, sort_keys=False)
            self.throw_publisher.publish(msg)
        except Exception as error:  # noqa: BLE001 - a Tk callback must not raise
            print(f"[DodgeballTab] ERROR publishing a throw: {error}")
            self._show_status(f"Error publishing: {error}", error=True)
            return payload

        self._show_status(
            "Thrown from %+.0f deg / %+.0f deg at %.1f m, %.1f m/s - impact in %.2f s."
            % (
                throw.azimuth_deg,
                throw.elevation_deg,
                throw.distance_m,
                throw.speed_mps,
                flight_time(throw),
            )
        )
        self._update_preview()
        return payload

    def _show_status(self, msg: str, error: bool = False) -> None:
        color = "#e74c3c" if error else "#27ae60"
        self.status_label.configure(text=msg, foreground=color)
        self.after(5000, lambda: self.status_label.configure(text=""))
