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

"""The colors of the viewer: OCS2's MATLAB-like marker palette and matplotlib's "tab10" curve colors.

The marker palette is OCS2's (`ocs2::Color`), which its visualization helpers drew the markers with before the Rerun
viewer, so a marker keeps the color it always had. The curve colors keep those of the ROS-era plots: measured blue
against a red reference for positions, measured green against an orange reference for angles and rates.
"""

from collections.abc import Sequence

# An RGB or RGBA color with channels in [0, 1], as humanoid_mpc_msgs.Color carries it.
Rgb = tuple[float, float, float]
Rgba = tuple[float, float, float, float]
# An RGBA color with channels in [0, 255], the form the bridge hands to Rerun.
Rgba8 = tuple[int, int, int, int]

# The MATLAB-like palette (ocs2::Color). The visualization publisher colors the footholds with it.
# LINT.IfChange(marker_palette)
BLUE: Rgb = (0.0, 0.4470, 0.7410)
ORANGE: Rgb = (0.8500, 0.3250, 0.0980)
YELLOW: Rgb = (0.9290, 0.6940, 0.1250)
PURPLE: Rgb = (0.4940, 0.1840, 0.5560)
GREEN: Rgb = (0.4660, 0.6740, 0.1880)
RED: Rgb = (0.6350, 0.0780, 0.1840)
BLACK: Rgb = (0.25, 0.25, 0.25)

# The colors of the contact points and their trajectories, by contact index.
CONTACT_COLORS: tuple[Rgb, ...] = (PURPLE, ORANGE, BLUE, GREEN, YELLOW)
# LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/visualization/include/humanoid_common_mpc_app/visualization/SceneContract.h:marker_palette)

# The color of a URDF visual that names no material.
DEFAULT_MESH_COLOR: Rgb = (0.7, 0.7, 0.7)

# The ground grid: light gray, alpha 0.9.
GRID_COLOR: Rgba8 = (160, 160, 164, 230)

# The curve colors (matplotlib tab10).
PLOT_BLUE = "#1f77b4"
PLOT_ORANGE = "#ff7f0e"
PLOT_GREEN = "#2ca02c"
PLOT_RED = "#d62728"
PLOT_PURPLE = "#9467bd"
PLOT_BROWN = "#8c564b"


def with_alpha(rgb: Sequence[float], alpha: float) -> Rgba:
    """The RGB color `rgb` (or the RGB channels of an RGBA one) with the opacity `alpha`."""
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), float(alpha))


def to_rgba8(color: Sequence[float]) -> Rgba8:
    """An RGB or RGBA color in [0, 1] as RGBA in [0, 255], clamped; RGB is opaque."""
    channels = list(color) + ([1.0] if len(color) == 3 else [])
    if len(channels) != 4:
        raise ValueError(f"a color has 3 or 4 channels, not {len(color)}")
    return tuple(int(round(min(max(float(c), 0.0), 1.0) * 255.0)) for c in channels)  # type: ignore[return-value]  # Four channels, checked above.


def hex_to_rgba8(color: str) -> Rgba8:
    """'#rrggbb' as opaque RGBA in [0, 255]."""
    text = color.lstrip("#")
    if len(text) != 6:
        raise ValueError(f"'{color}' is not a #rrggbb color")
    return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16), 255)
