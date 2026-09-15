/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.

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
******************************************************************************/

#include "mujoco_sim_interface/visualization/ContactTimelineVisualization.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#include "mujoco_sim_interface/MujocoSimInterface.h"

namespace robot::mujoco_sim_interface {

namespace {
struct Rgba {
  float r, g, b, a;
};
constexpr Rgba kPanelBackground{0.0f, 0.0f, 0.0f, 0.55f};
constexpr Rgba kPlanContact{0.35f, 0.70f, 1.0f, 1.0f};  // the plan has the point in contact
constexpr Rgba kSwing{0.16f, 0.16f, 0.20f, 1.0f};       // in the air (plan or physics, in agreement)
constexpr Rgba kUnknown{0.40f, 0.40f, 0.40f, 1.0f};     // no plan yet, or no MuJoCo body for the contact point
constexpr Rgba kSimContact{0.35f, 0.85f, 0.35f, 1.0f};  // physics agrees: touching
constexpr Rgba kSimEarly{0.95f, 0.25f, 0.25f, 1.0f};    // touching while the plan says swing (early touch-down, scuff)
constexpr Rgba kSimLate{1.0f, 0.62f, 0.10f, 1.0f};      // in the air while the plan says contact (late touch-down, slip)
constexpr Rgba kTick{1.0f, 1.0f, 1.0f, 0.25f};
constexpr Rgba kText{0.92f, 0.92f, 0.92f, 1.0f};

void fillRect(int left, int bottom, int width, int height, const Rgba& color) {
  if (width <= 0 || height <= 0) return;
  mjr_rectangle(mjrRect{left, bottom, width, height}, color.r, color.g, color.b, color.a);
}

/// Width in pixels of `text` in the normal font of `con`. A label box narrower or shorter than its text leaves the
/// string's raster position outside the box, and OpenGL then drops the whole string, so boxes are sized from this.
int textWidth(const char* text, const mjrContext* con) {
  int width = 0;
  for (; *text != '\0'; ++text) width += con->charWidth[static_cast<unsigned char>(*text) & 127];
  return width;
}

void drawLabel(int left, int bottom, int width, int height, const char* text, const mjrContext* con) {
  if (width <= 0 || height <= 0) return;
  mjr_label(mjrRect{left, bottom, width, height}, mjFONT_NORMAL, text, 0.0f, 0.0f, 0.0f, 0.0f, kText.r, kText.g, kText.b, con);
}

/// Combined state of one contact point in one sample: bit 0 physics touching, bit 1 plan contact, bit 2 plan known.
int contactStateCode(const ContactTimelineSample& sample, int contact) {
  return static_cast<int>((sample.actual >> contact) & 1u) | (static_cast<int>((sample.target >> contact) & 1u) << 1) |
         (sample.targetKnown ? 4 : 0);
}
}  // namespace

void ContactTimelineVisualization::renderOverlay(const VisualizationFrame& frame) {
  if (frame.sim == nullptr || frame.context == nullptr || !frame.sim->hasContactDetection()) return;
  const MujocoSimInterface& sim = *frame.sim;
  sim.copyContactTimeline(scratch_);
  const std::vector<ContactTimelineSample>& samples = scratch_;
  if (samples.empty()) return;

  const std::vector<std::string>& names = sim.getContactNames();
  const int numContacts = static_cast<int>(names.size());
  const double window = sim.getContactTimelineWindow();
  const double now = samples.back().time;
  const double start = now - window;
  const uint32_t unresolved = sim.getUnresolvedContactMask();
  const mjrRect& viewport = frame.viewport;

  // Layout in framebuffer pixels (origin bottom-left): a translucent panel along the bottom edge, the label columns on
  // the left, the time axis running left (oldest) to right (now). Every box that holds text is sized from the font.
  const mjrContext* con = frame.context;
  const int fontH = std::max(12, con->charHeight);
  constexpr int margin = 12, pad = 8, stripGap = 2, groupGap = 8;
  const int stripH = fontH + 4;  // tall enough for its "plan" / "sim" tag
  const int headerH = fontH + 6;
  const int axisH = fontH + 6;
  const int tagW = std::max(textWidth("plan", con), textWidth("sim", con)) + 12;
  int nameTextW = 0;
  for (const std::string& name : names) nameTextW = std::max(nameTextW, textWidth(name.c_str(), con));
  const int nameW = std::clamp(nameTextW + 12, 60, 400);
  const int groupH = 2 * stripH + stripGap;
  const int panelH = 2 * pad + headerH + numContacts * (groupH + groupGap) + axisH;
  const int panelW = viewport.width - 2 * margin;
  const int panelL = margin;
  const int panelB = margin;
  const int x0 = panelL + pad + nameW + tagW;
  const int x1 = panelL + panelW - pad;
  const int stripW = x1 - x0;
  if (stripW < 120 || panelH > viewport.height / 2) return;
  fillRect(panelL, panelB, panelW, panelH, kPanelBackground);

  const auto xOf = [&](double time) {
    const double fraction = std::clamp((time - start) / window, 0.0, 1.0);
    return x0 + static_cast<int>(std::lround(fraction * stripW));
  };

  // Header: title and legend.
  const int headerB = panelB + panelH - pad - headerH;
  const char* title = "contact timeline [b]";
  drawLabel(panelL + pad, headerB, std::max(nameW + tagW, textWidth(title, con) + 8), headerH, title, con);
  {
    int lx = x0;
    const auto legend = [&](const Rgba& color, const char* text) {
      const int textW = textWidth(text, con) + 8;
      if (lx + 18 + textW > x1) return;  // no room: the rest of the legend is dropped
      fillRect(lx, headerB + 4, fontH - 4, headerH - 8, color);
      lx += fontH;
      drawLabel(lx, headerB, textW, headerH, text, con);
      lx += textW + 14;
    };
    legend(kPlanContact, "plan: contact");
    legend(kSimContact, "sim: contact");
    legend(kSimEarly, "sim touching, plan swing");
    legend(kSimLate, "sim in air, plan contact");
  }

  // One group of two strips per contact point: the plan on top, the physics below.
  const int stripsTop = headerB - groupGap;
  int groupTop = stripsTop;
  for (int contact = 0; contact < numContacts; ++contact) {
    const int targetB = groupTop - stripH;
    const int actualB = targetB - stripGap - stripH;
    drawLabel(panelL + pad, actualB, nameW, groupH, names[contact].c_str(), con);
    drawLabel(panelL + pad + nameW, targetB, tagW, stripH, "plan", con);
    drawLabel(panelL + pad + nameW, actualB, tagW, stripH, "sim", con);

    if (((unresolved >> contact) & 1u) != 0u) {
      fillRect(x0, actualB, stripW, groupH, kUnknown);
      drawLabel(x0, actualB, stripW, groupH, "no MuJoCo body resolved for this contact frame", con);
      groupTop = actualB - groupGap;
      continue;
    }

    // Runs of equal state are drawn as single bars.
    size_t runStart = 0;
    for (size_t k = 1; k <= samples.size(); ++k) {
      const int code = contactStateCode(samples[runStart], contact);
      if (k < samples.size() && contactStateCode(samples[k], contact) == code) continue;
      const double runEnd = (k < samples.size()) ? samples[k].time : now;
      if (runEnd >= start) {
        const int xa = xOf(samples[runStart].time);
        const int xb = std::max(xOf(runEnd), xa + 1);
        const bool touching = (code & 1) != 0;
        const bool planContact = (code & 2) != 0;
        const bool planKnown = (code & 4) != 0;
        fillRect(xa, targetB, xb - xa, stripH, !planKnown ? kUnknown : (planContact ? kPlanContact : kSwing));
        const Rgba& simColor = !planKnown ? (touching ? kSimContact : kSwing)
                               : touching ? (planContact ? kSimContact : kSimEarly)
                                          : (planContact ? kSimLate : kSwing);
        fillRect(xa, actualB, xb - xa, stripH, simColor);
      }
      runStart = k;
    }
    groupTop = actualB - groupGap;
  }

  // Time axis: one tick per second, relative to now, at fixed positions.
  const int axisB = panelB + pad;
  const int ticksH = stripsTop - (axisB + axisH);
  for (int secondsAgo = 0; secondsAgo <= static_cast<int>(window); ++secondsAgo) {
    const int x = xOf(now - secondsAgo);
    fillRect(x, axisB + axisH, 1, ticksH, kTick);
    char text[16];
    if (secondsAgo == 0) {
      std::snprintf(text, sizeof(text), "now");
    } else {
      std::snprintf(text, sizeof(text), "-%d s", secondsAgo);
    }
    const int labelW = textWidth(text, con) + 8;
    drawLabel(std::clamp(x - labelW / 2, panelL, x1 - labelW), axisB, labelW, axisH, text, con);
  }
}

}  // namespace robot::mujoco_sim_interface
