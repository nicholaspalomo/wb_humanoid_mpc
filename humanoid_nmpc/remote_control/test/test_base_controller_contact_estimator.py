# Copyright (c) 2026, Nicholas Palomo. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Tests of the contact estimator drop-down on the Base Controller tab of the joystick GUI.

The drop-down offers the names of the contact_estimator registry (config_registries.textproto), mirrors the selection
of the MPC Parameters tab (task file `contact_estimator`) and selects through it, so a choice publishes the name on
operator/mpc_parameters like a slider. Needs a display.
"""

import os
import shutil
import tempfile
import unittest

from humanoid_mpc_config import mpc_parameter_update_pb2

from humanoid_mpc_ipc import topics
import operator_test_support
from remote_control import base_velocity_controller_gui


@operator_test_support.requires_display
class TestBaseControllerContactEstimatorDropDown(unittest.TestCase):
    atlas_config: str

    @classmethod
    def setUpClass(cls):
        cls.atlas_config = operator_test_support.ATLAS_CONFIG

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        config = operator_test_support.copy_config(self.atlas_config, self.tmpdir)
        self.task_file = os.path.join(config, "mpc", "task.textproto")
        self.publisher = operator_test_support.RecordingPublisher(
            topics.OPERATOR_MPC_PARAMETERS
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_app(self, enable_online_tuning=True):
        return base_velocity_controller_gui.App(
            pd_gains_file=operator_test_support.ATLAS_PD_GAINS_FILE,
            task_file=self.task_file,
            reference_file=operator_test_support.ATLAS_REFERENCE_FILE,
            enable_online_tuning=enable_online_tuning,
            param_publisher=self.publisher,
            pd_gains_publisher=operator_test_support.RecordingPublisher(
                topics.OPERATOR_PD_GAINS
            ),
            joint_targets_publisher=operator_test_support.RecordingPublisher(
                topics.OPERATOR_JOINT_TARGETS
            ),
        )

    def test_the_drop_down_offers_the_registry_and_selects_the_contact_estimator(self):
        app = self._create_app()
        try:
            app.withdraw()
            self.assertEqual(
                app.contact_estimator_var.get(),
                "cheater_sim",
                "the Atlas task file selects cheater_sim",
            )
            self.assertNotIn("disabled", app.contact_estimator_combobox.state())
            offered = tuple(app.contact_estimator_combobox.cget("values"))
            self.assertIn("cheater_sim", offered)
            self.assertIn("always_in_contact", offered)

            # Choosing always_in_contact selects it through the MPC Parameters tab and publishes it.
            app.contact_estimator_var.set("always_in_contact")
            app._on_contact_estimator_selected()
            tab = app.mpc_params_tab
            self.assertEqual(tab.selected_contact_estimator(), "always_in_contact")
            tab.after_cancel(tab._debounce_publish_id)
            tab._debounce_publish_id = None
            tab._publish_to_topic()
            published = self.publisher.last_message
            assert isinstance(published, mpc_parameter_update_pb2.MpcParameterUpdate)
            self.assertEqual(published.task.contact_estimator, "always_in_contact")

            # The tab's Reset All restores the file's selection and the drop-down follows.
            tab.reset_all_defaults()
            self.assertEqual(app.contact_estimator_var.get(), "cheater_sim")
            reset = self.publisher.last_message
            assert isinstance(reset, mpc_parameter_update_pb2.MpcParameterUpdate)
            self.assertEqual(reset.task.contact_estimator, "cheater_sim")
        finally:
            app.destroy()

    def test_the_drop_down_is_disabled_without_online_tuning(self):
        app = self._create_app(enable_online_tuning=False)
        try:
            app.withdraw()
            self.assertIn("disabled", app.contact_estimator_combobox.state())
            self.assertEqual(app.contact_estimator_var.get(), "cheater_sim")
        finally:
            app.destroy()


if __name__ == "__main__":
    unittest.main()
