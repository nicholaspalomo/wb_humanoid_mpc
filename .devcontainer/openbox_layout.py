#!/usr/bin/env python3
"""Write the Openbox <application> placement rules for the VNC desktop.

Three windows share the display: the native Rerun viewer (when the Rerun bridge runs it inside the container rather than
on the host), the MuJoCo viewer, and the Tkinter "Robot Base Controller & Tuning" GUI from the operator tools. Each gets
a pane of its own, so that no window ever covers another and none of them ever needs raising.

That matters for the GUI in particular. It raises itself with `-topmost` for only 1.5 s at start-up, and a 3D viewer
that is still opening under software GL beats that every time. When the GUI had no rule, the 3D viewer was placed
across the whole left pane on top of it, leaving a ~10 px strip of the GUI and nothing to click to bring it forward. So the left pane is split between the Rerun viewer (top) and the GUI (bottom), and MuJoCo keeps the right pane.

Called by start_vnc.sh with the target resolution in $RESOLUTION. It lives in its own file rather than inside a
`python3 -c` string so that the XML quoting stays readable.
"""

import os
import sys
from typing import Dict, Tuple

# The controller GUI's own tkinter minsize(), which Openbox cannot shrink a window below.
GUI_MIN_W = 800
GUI_MIN_H = 520

# The fallback when $RESOLUTION is unset or malformed.
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080

# Openbox 3.6 <application> rules accept position, size, maximized, focus, layer, desktop and friends; `force="yes"`
# on the position is what makes it override a window that sets its own geometry, which all three of these do. Every
# attribute of a rule has to match, so each one matches on the window title alone.
RULE_TEMPLATE = """
  <application title="*Rerun*" name="*" class="*">
    <position force="yes"><x>0</x><y>0</y></position>
    <size><width>{left_w}</width><height>{viewer_h}</height></size>
    <maximized>no</maximized>
  </application>
  <application title="*Robot Base Controller*" name="*" class="*">
    <position force="yes"><x>0</x><y>{gui_y}</y></position>
    <size><width>{left_w}</width><height>{gui_h}</height></size>
    <maximized>no</maximized>
  </application>
  <application title="*Mujoco*" name="*" class="*">
    <position force="yes"><x>{half_w}</x><y>0</y></position>
    <size><width>{left_w}</width><height>{pane_h}</height></size>
    <maximized>no</maximized>
    <focus>yes</focus>
  </application>
"""


def layout(width: int, height: int) -> Dict[str, int]:
    """Geometry of the three panes for a display of the given size."""
    half_w = width // 2
    pane_h = height - 40  # leave room for the window bar
    gap = 10
    left_w = half_w - gap

    # The GUI declares minsize(800, 520). Openbox cannot size a window below the minimum the application itself asks
    # for through WM_NORMAL_HINTS, so asking for less than that would silently leave it taller than its slot and
    # overlapping the viewer again. Give it at least its own minimum, anchor it to the bottom of the pane, and hand the
    # viewer everything above it. Anchoring the GUI rather than stacking upwards from the viewer keeps the two inside
    # the pane at every resolution: on a short display the viewer is squeezed (and main() says so) instead of the GUI
    # hanging off-screen.
    gui_h = min(max(GUI_MIN_H, (pane_h - gap) // 2), pane_h)
    gui_y = pane_h - gui_h
    viewer_h = max(1, gui_y - gap)
    return {
        "half_w": half_w,
        "pane_h": pane_h,
        "left_w": left_w,
        "gui_h": gui_h,
        "viewer_h": viewer_h,
        "gui_y": gui_y,
    }


def parse_resolution(value: str) -> Tuple[int, int]:
    """WIDTHxHEIGHT as two integers, or the default display when the value is not of that form."""
    try:
        width, height = (int(x) for x in value.split("x"))
    except ValueError:
        return DEFAULT_WIDTH, DEFAULT_HEIGHT
    return width, height


def main() -> int:
    width, height = parse_resolution(os.environ.get("RESOLUTION", ""))

    geom = layout(width, height)
    rc_path = os.path.expanduser("~/.config/openbox/rc.xml")

    with open(rc_path, "r") as handle:
        content = handle.read()

    if "</applications>" not in content:
        print(
            f"WARNING: {rc_path} has no <applications> section; layout rules not applied.",
            file=sys.stderr,
        )
        return 1

    content = content.replace(
        "</applications>", RULE_TEMPLATE.format(**geom) + "\n</applications>"
    )
    with open(rc_path, "w") as handle:
        handle.write(content)

    # Say so when the display cannot really hold the tiled layout, rather than letting a window silently overlap.
    if geom["left_w"] < GUI_MIN_W:
        print(
            f"NOTE: the left pane is {geom['left_w']} px wide but the controller GUI cannot go below its own "
            f"{GUI_MIN_W} px minimum, so it will overlap the MuJoCo pane at this resolution.",
            file=sys.stderr,
        )
    if geom["viewer_h"] < 300:
        print(
            f"NOTE: this display is short, so the Rerun viewer gets only {geom['viewer_h']} px of height above the "
            "controller GUI.",
            file=sys.stderr,
        )

    print(
        f"Openbox layout for {width}x{height}: "
        f"Rerun {geom['left_w']}x{geom['viewer_h']}+0+0, "
        f"Controller GUI {geom['left_w']}x{geom['gui_h']}+0+{geom['gui_y']}, "
        f"MuJoCo {geom['left_w']}x{geom['pane_h']}+{geom['half_w']}+0"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
