"""****************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

import unittest
from humanoid_nmpc.remote_control.remote_control.dashboard_backend import (
    SimProcessManager,
    VirtualJoystickROS2,
)


class TestDashboardBackend(unittest.TestCase):
    def setUp(self):
        self.sim_mgr = SimProcessManager()
        self.sim_mgr.stop()
        self.joy = VirtualJoystickROS2()

    def tearDown(self):
        self.sim_mgr.stop()
        self.joy.shutdown()

    def test_sim_manager_targets(self):
        """Verifies simulation target registry and initial state."""
        self.assertIn("g1_centroidal_dummy", self.sim_mgr.TARGETS)
        self.assertIn("g1_centroidal_sim", self.sim_mgr.TARGETS)
        self.assertIn("g1_wb_dummy", self.sim_mgr.TARGETS)
        self.assertIn("g1_wb_sim", self.sim_mgr.TARGETS)
        self.assertIn("atlas_centroidal_dummy", self.sim_mgr.TARGETS)
        self.assertIn("atlas_centroidal_sim", self.sim_mgr.TARGETS)

        status = self.sim_mgr.get_status()
        self.assertEqual(status["status"], "STOPPED")

    def test_virtual_joystick_set_and_step(self):
        """Verifies virtual joystick command setting, directional stepping, and limits."""
        self.joy.set_velocity(
            linear_x=0.5, linear_y=-0.2, angular_z=0.3, desired_height=0.7
        )
        self.assertAlmostEqual(self.joy.v_x, 0.5)
        self.assertAlmostEqual(self.joy.v_y, -0.2)
        self.assertAlmostEqual(self.joy.v_yaw, 0.3)
        self.assertAlmostEqual(self.joy.desired_height, 0.7)

        # Step forward
        self.joy.step("forward", delta_v=0.2)
        self.assertAlmostEqual(self.joy.v_x, 0.7)

        # Emergency stop
        self.joy.step("stop")
        self.assertEqual(self.joy.v_x, 0.0)
        self.assertEqual(self.joy.v_y, 0.0)
        self.assertEqual(self.joy.v_yaw, 0.0)


if __name__ == "__main__":
    unittest.main()
