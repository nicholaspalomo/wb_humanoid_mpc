"""
Tests of the cheater contact estimator checkbox on the Base Controller tab of the joystick GUI.

The checkbox mirrors the contact estimator selection of the MPC Parameters tab (task file `contactEstimator`) and
selects through it, so a toggle publishes the name on operator/mpc_parameters like a slider. Needs a display.
"""

import os
import shutil
import tempfile
import unittest

import yaml

from humanoid_mpc_ipc import topics
from humanoid_mpc_msgs import yaml_document_pb2
from operator_test_support import ATLAS_CONFIG, RecordingPublisher, requires_display


@requires_display
class TestBaseControllerContactEstimatorCheckbox(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.atlas_config = ATLAS_CONFIG

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.task_file = os.path.join(self.tmpdir, "task.yaml")
        shutil.copy2(os.path.join(self.atlas_config, "mpc/task.yaml"), self.task_file)
        shutil.copy2(
            os.path.join(self.atlas_config, "mpc/contact_planning.yaml"),
            os.path.join(self.tmpdir, "contact_planning.yaml"),
        )
        self.publisher = RecordingPublisher(topics.OPERATOR_MPC_PARAMETERS)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_app(self, enable_online_tuning=True):
        from remote_control.base_velocity_controller_gui import App

        return App(
            pd_gains_file=os.path.join(
                self.atlas_config, "controller/joint_pd_gains.yaml"
            ),
            task_file=self.task_file,
            reference_file=os.path.join(self.atlas_config, "command/reference.yaml"),
            enable_online_tuning=enable_online_tuning,
            param_publisher=self.publisher,
            pd_gains_publisher=RecordingPublisher(topics.OPERATOR_PD_GAINS),
            joint_targets_publisher=RecordingPublisher(topics.OPERATOR_JOINT_TARGETS),
        )

    def test_checkbox_mirrors_and_selects_the_contact_estimator(self):
        app = self._create_app()
        try:
            app.withdraw()
            self.assertTrue(
                app.cheater_contacts_var.get(),
                "the Atlas task file selects cheater_sim",
            )
            self.assertNotIn("disabled", app.cheater_contacts_checkbox.state())

            # Unchecking selects always_in_contact through the MPC Parameters tab and publishes it.
            app.cheater_contacts_var.set(False)
            app._on_cheater_contacts_toggle()
            tab = app.mpc_params_tab
            self.assertFalse(tab.is_cheater_contact_estimator_selected())
            tab.after_cancel(tab._debounce_publish_id)
            tab._debounce_publish_id = None
            tab._publish_to_topic()
            self.assertIsInstance(
                self.publisher.last_message, yaml_document_pb2.YamlDocument
            )
            self.assertEqual(
                yaml.safe_load(self.publisher.last_yaml)["contactEstimator"],
                "always_in_contact",
            )

            # The tab's Reset All restores the file's selection and the checkbox follows.
            tab.reset_all_defaults()
            self.assertTrue(app.cheater_contacts_var.get())
            self.assertEqual(
                yaml.safe_load(self.publisher.last_yaml)["contactEstimator"],
                "cheater_sim",
            )
        finally:
            app.destroy()

    def test_checkbox_is_disabled_without_online_tuning(self):
        app = self._create_app(enable_online_tuning=False)
        try:
            app.withdraw()
            self.assertIn("disabled", app.cheater_contacts_checkbox.state())
            self.assertTrue(app.cheater_contacts_var.get())
        finally:
            app.destroy()


if __name__ == "__main__":
    unittest.main()
