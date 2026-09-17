"""
Tests of the cheater contact estimator checkbox on the Base Controller tab of the joystick GUI.

The checkbox mirrors the contact estimator selection of the MPC Parameters tab (task file `contactEstimator`) and
selects through it, so a toggle publishes the name on the parameter topic like a slider. Needs the ROS 2 Python
packages the GUI module imports and a display (run under xvfb-run).
"""

import os
import shutil
import tempfile
import unittest

import yaml


class MockPublisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)

    @property
    def last_data(self):
        return self.messages[-1].data if self.messages else None


class TestBaseControllerContactEstimatorCheckbox(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../..")
        )
        cls.atlas_config = os.path.join(
            cls.repo_root, "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config"
        )

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.task_file = os.path.join(self.tmpdir, "task.yaml")
        shutil.copy2(os.path.join(self.atlas_config, "mpc/task.yaml"), self.task_file)
        shutil.copy2(
            os.path.join(self.atlas_config, "mpc/contact_planning.yaml"),
            os.path.join(self.tmpdir, "contact_planning.yaml"),
        )
        self.publisher = MockPublisher()

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
            enable_telemetry=False,
            param_publisher=self.publisher,
            pd_gains_publisher=MockPublisher(),
            joint_targets_publisher=MockPublisher(),
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
            self.assertEqual(
                yaml.safe_load(self.publisher.last_data)["contactEstimator"],
                "always_in_contact",
            )

            # The tab's Reset All restores the file's selection and the checkbox follows.
            tab.reset_all_defaults()
            self.assertTrue(app.cheater_contacts_var.get())
            self.assertEqual(
                yaml.safe_load(self.publisher.last_data)["contactEstimator"],
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
