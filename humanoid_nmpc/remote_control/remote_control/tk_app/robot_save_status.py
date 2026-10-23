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

"""A tuning tab's robot copy after Save: sends the saved text and shows the robot's answer on the tab's status line.

The MPC Parameters, Joint PD Gains and Command Limits tabs each own one RobotSaveStatus. Their Save writes the laptop's
file first and, only when that succeeded, calls send() with the exact text written; send() hands it to the GUI's
robot_config_save.RobotConfigSaver and shows "Saved on the laptop · saving on the robot…", and Tk's after() polls the
saver every 200 ms while the robot's answer may still change the line (robot_config_save.describe() words it).
"""

from collections.abc import Callable
import tkinter as tk

from remote_control import robot_config_save

# How often the tab polls the saver while it waits for the robot [ms].
POLL_PERIOD_MS = 200


class RobotSaveStatus:
    """Follows a tab's last save to the robot with Tk's timer and shows each change of it.

    Called from Tk's thread only.

    Args:
        widget: The tab, whose after() runs the polls.
        saver: The GUI's saver; None: the tab has no bus, and send() shows that the file was not sent.
        show: Shows a status line: the text, and whether it reports a problem (the robot's copy may differ).
    """

    def __init__(
        self,
        widget: tk.Misc,
        saver: robot_config_save.RobotConfigSaver | None,
        show: Callable[[str, bool], None],
    ) -> None:
        self._widget = widget
        self._tracker = robot_config_save.SaveTracker(saver)
        self._show = show
        self._poll_id: str | None = None

    def outcome(self) -> robot_config_save.SaveOutcome | None:
        """The outcome the status line shows; None before the first send()."""
        return self._tracker.outcome()

    def send(self, kind: int, path: str, text: str) -> robot_config_save.SaveOutcome:
        """Sends a file the tab has just saved on the laptop to the robot, and shows and follows the outcome.

        Args:
            kind: The file's kind (robot_config_save.KIND_*).
            path: The laptop's file.
            text: Exactly the text written into it (tuned_file.TunedFile.save()).

        Returns:
            The outcome now: SAVING, or NOT_SENT with the reason.
        """
        if self._poll_id is not None:
            self._widget.after_cancel(self._poll_id)
            self._poll_id = None
        outcome = self._tracker.send(kind, path, text)
        self._display(outcome)
        self._schedule()
        return outcome

    def poll(self) -> None:
        """Takes the robot's answers, shows the outcome when it changed, and polls again while it may still change."""
        self._poll_id = None
        changed = self._tracker.poll()
        if changed is not None:
            self._display(changed)
        self._schedule()

    def _schedule(self) -> None:
        outcome = self._tracker.outcome()
        if outcome is not None and outcome.waiting:
            self._poll_id = self._widget.after(POLL_PERIOD_MS, self.poll)

    def _display(self, outcome: robot_config_save.SaveOutcome) -> None:
        self._show(
            robot_config_save.describe(outcome, self._tracker.timeout), outcome.problem
        )
