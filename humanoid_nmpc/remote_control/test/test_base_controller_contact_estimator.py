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

    @staticmethod
    def _set_the_hop_rule(app, listed):
        """Lists or unlists hop_on_request in the loaded planner block, so the test covers a planner that hops and one
        that only walks whichever way the shipped file happens to be configured."""
        planning = app.mpc_params_tab.raw_data.get("contact_planning", {})
        rules = [r for r in planning.get("logic_rules", []) if r != "hop_on_request"]
        planning["logic_rules"] = rules + (["hop_on_request"] if listed else [])
        app._sync_hop_label()
        return planning

    def test_height_slider_shows_the_hop_trigger_only_when_the_planner_hops(self):
        """The root height slider says where hops begin while the planner lists the hop_on_request rule. A planner that
        only walks says so instead, and a press explains itself rather than leaving a dead button.
        """
        app = self._create_app()
        try:
            app.withdraw()
            planning = self._set_the_hop_rule(app, listed=True)
            self.assertEqual(planning["hop_on_request"]["triggerBaseHeight"], 1.0)
            self.assertEqual(app.hop_trigger_height(), 1.0)
            self.assertEqual(app.hop_label.cget("text"), "hop above 1.00 m")

            self._set_the_hop_rule(app, listed=False)
            self.assertIsNone(app.hop_trigger_height())
            self.assertEqual(
                app.hop_label.cget("text"), "no hop in contact_planning.yaml"
            )
            app.jump()
            self.assertIsNone(app._jump_after_id)
            self.assertIn("hop_on_request", app.hop_label.cget("text"))
            app.after_cancel(app._hop_message_after_id)

            # A block without the parameters gives no label rather than a wrong one.
            planning = self._set_the_hop_rule(app, listed=True)
            del planning["hop_on_request"]
            app._sync_hop_label()
            self.assertIsNone(app.hop_trigger_height())
            self.assertEqual(
                app.hop_label.cget("text"), "no hop in contact_planning.yaml"
            )
        finally:
            app.destroy()

    def test_jump_button_commands_a_height_above_the_trigger_and_restores_it(self):
        """The button is the whole jump command: it raises the commanded root height past the planner's hop trigger,
        which is what hop_on_request watches, and puts the previous height back when the pulse is over.
        """
        app = self._create_app()
        try:
            app.withdraw()
            self._set_the_hop_rule(app, listed=True)
            standing = app.get_walking_command_msg().desired_pelvis_height
            self.assertLess(standing, 1.0, "the robot stands below the hop trigger")

            app.jump()
            commanded = app.get_walking_command_msg().desired_pelvis_height
            self.assertGreater(commanded, 1.0, "the planner hops above the trigger")
            self.assertAlmostEqual(commanded, 1.0 + app.JUMP_TRIGGER_MARGIN, places=3)
            self.assertIn(
                "disabled", app.jump_button.state(), "no second jump while one runs"
            )
            self.assertEqual(app.jump_button.cget("text"), "\u2912 Jumping\u2026")

            # A second press while the jump runs changes nothing.
            pending = app._jump_after_id
            app.jump()
            self.assertEqual(app._jump_after_id, pending)

            app.after_cancel(pending)
            app._jump_after_id = pending  # end_jump clears it
            app.end_jump()
            self.assertAlmostEqual(
                app.get_walking_command_msg().desired_pelvis_height, standing, places=6
            )
            self.assertEqual(app.jump_button.cget("text"), "\u2912 Jump")
            self.assertNotIn("disabled", app.jump_button.state())
        finally:
            app.destroy()

    def test_jump_button_stays_dead_when_the_trigger_is_out_of_the_sliders_range(self):
        """A trigger above the slider's maximum cannot be commanded: the label says so and the button stays disabled,
        instead of sending a height that never reaches the planner's threshold.
        """
        app = self._create_app()
        try:
            app.withdraw()
            planning = self._set_the_hop_rule(app, listed=True)
            planning["hop_on_request"]["triggerBaseHeight"] = app.max_height + 0.5
            app._sync_hop_label()
            self.assertIn("past the slider", app.hop_label.cget("text"))
            before = app.get_walking_command_msg().desired_pelvis_height
            app.jump()
            self.assertIsNone(app._jump_after_id)
            self.assertAlmostEqual(
                app.get_walking_command_msg().desired_pelvis_height, before, places=6
            )
            self.assertIn(
                "below the",
                app.hop_label.cget("text"),
                "the press says why it could not act",
            )
            app.after_cancel(app._hop_message_after_id)
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
