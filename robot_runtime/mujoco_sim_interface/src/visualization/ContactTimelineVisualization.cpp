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
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/strings/str_format.h"

#include "mujoco_sim_interface/MujocoSimInterface.h"

namespace robot::mujoco_sim_interface {

namespace {
struct Rgba {
  float r, g, b, a;
};
constexpr Rgba kPanelBackground{.r = 0.0f, .g = 0.0f, .b = 0.0f, .a = 0.55f};
constexpr Rgba kPlanContact{.r = 0.35f,
                            .g = 0.70f,
                            .b = 1.0f,
                            .a = 1.0f};  // the plan has the point in contact
constexpr Rgba kSwing{.r = 0.16f,
                      .g = 0.16f,
                      .b = 0.20f,
                      .a = 1.0f};  // in the air (plan or physics, in agreement)
constexpr Rgba kUnknown{
    .r = 0.40f,
    .g = 0.40f,
    .b = 0.40f,
    .a = 1.0f};  // no plan yet, or no MuJoCo body for the contact point
constexpr Rgba kSimContact{
    .r = 0.35f, .g = 0.85f, .b = 0.35f, .a = 1.0f};  // physics agrees: touching
constexpr Rgba kSimEarly{
    .r = 0.95f,
    .g = 0.25f,
    .b = 0.25f,
    .a = 1.0f};  // touching while the plan says swing (early touch-down, scuff)
constexpr Rgba kSimLate{.r = 1.0f,
                        .g = 0.62f,
                        .b = 0.10f,
                        .a = 1.0f};  // in the air while the plan says contact
                                     // (late touch-down, slip)
constexpr Rgba kTick{.r = 1.0f, .g = 1.0f, .b = 1.0f, .a = 0.25f};
constexpr Rgba kText{.r = 0.92f, .g = 0.92f, .b = 0.92f, .a = 1.0f};

// mjrContext::charWidth covers the 7-bit ASCII codes.
constexpr int kAsciiCodeMask = 127;

// The bits of contactStateCode().
constexpr int kTouchingBit = 1;     // physics touching
constexpr int kPlanContactBit = 2;  // plan contact
constexpr int kPlanKnownBit = 4;    // plan known

// The layout of the panel, in framebuffer pixels.
constexpr int kMinFontHeight = 12;
constexpr int kPanelMargin = 12;  // between the panel and the viewport's edges
constexpr int kPanelPadding = 8;  // between the panel's edges and its contents
constexpr int kPanelSidesPerAxis =
    2;  // the margin and the padding are on both sides of each axis
constexpr int kStripsPerGroup = 2;  // a contact point has a group of two
                                    // strips, the plan's and the physics'
constexpr int kStripGap = 2;        // between the two strips of a contact point
constexpr int kGroupGap = 8;        // between the groups of two contact points
// The heights of the rows beyond the font's, and the widths of the columns and
// labels beyond their text.
constexpr int kStripTextClearance = 4;  // a strip, for its "plan" / "sim" tag
constexpr int kRowTextClearance = 6;    // the header and the time axis
constexpr int kColumnTextPadding = 12;  // the tag and the name columns
constexpr int kLabelTextPadding = 8;    // a label
constexpr int kMinNameColumnWidth = 60;
constexpr int kMaxNameColumnWidth = 400;
// The panel is drawn only when its strips are at least kMinStripWidth wide and
// it takes at most a 1 / kMinViewportToPanelHeightRatio of the viewport's
// height.
constexpr int kMinStripWidth = 120;
constexpr int kMinViewportToPanelHeightRatio = 2;
// A legend entry: a swatch inset by kLegendSwatchInset in the header row and as
// much narrower than the font is high, then its label and kLegendEntryGap. An
// entry is drawn while kLegendSwatchRoom and its label fit.
constexpr int kLegendSwatchRoom = 18;
constexpr int kLegendSwatchInset = 4;
constexpr int kLegendSwatchHeightInset =
    2 * kLegendSwatchInset;  // above and below
constexpr int kLegendEntryGap = 14;
constexpr int kTickLabelCenteringDivisor =
    2;  // a tick's label is centered on the tick

void fillRect(int left, int bottom, int width, int height, const Rgba& color) {
  if (width <= 0 || height <= 0) return;
  mjr_rectangle(mjrRect{left, bottom, width, height}, color.r, color.g, color.b,
                color.a);
}

/// Width in pixels of `text` in the normal font of `con`. A label box narrower
/// or shorter than its text leaves the string's raster position outside the
/// box, and OpenGL then drops the whole string, so boxes are sized from this.
int textWidth(const char* absl_nonnull text,
              const mjrContext* absl_nonnull con) {
  int width = 0;
  for (; *text != '\0'; ++text)
    width += con->charWidth[static_cast<unsigned char>(*text) & kAsciiCodeMask];
  return width;
}

void drawLabel(int left,
               int bottom,
               int width,
               int height,
               const char* absl_nonnull text,
               const mjrContext* absl_nonnull con) {
  if (width <= 0 || height <= 0) return;
  mjr_label(mjrRect{left, bottom, width, height}, mjFONT_NORMAL, text,
            /*r=*/0.0f, /*g=*/0.0f, /*b=*/0.0f, /*a=*/0.0f, kText.r, kText.g,
            kText.b, con);
}

/// Combined state of one contact point in one sample: bit 0 physics touching,
/// bit 1 plan contact, bit 2 plan known.
int contactStateCode(const ContactTimelineSample& sample, int contact) {
  return static_cast<int>((sample.actual >> contact) & 1u) |
         (static_cast<int>((sample.target >> contact) & 1u) << 1) |
         (sample.targetKnown ? kPlanKnownBit : 0);
}
}  // namespace

std::string ContactTimelineVisualization::tickLabel(int secondsAgo) {
  return secondsAgo == 0 ? std::string("now")
                         : absl::StrFormat("-%d s", secondsAgo);
}

void ContactTimelineVisualization::renderOverlay(
    const VisualizationFrame& frame) {
  if (frame.sim == nullptr || frame.context == nullptr ||
      !frame.sim->hasContactDetection())
    return;
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

  // Layout in framebuffer pixels (origin bottom-left): a translucent panel
  // along the bottom edge, the label columns on the left, the time axis running
  // left (oldest) to right (now). Every box that holds text is sized from the
  // font.
  const mjrContext* absl_nonnull con = frame.context;
  const int fontH = std::max(kMinFontHeight, con->charHeight);
  const int stripH =
      fontH + kStripTextClearance;  // tall enough for its "plan" / "sim" tag
  const int headerH = fontH + kRowTextClearance;
  const int axisH = fontH + kRowTextClearance;
  const int tagW = std::max(textWidth("plan", con), textWidth("sim", con)) +
                   kColumnTextPadding;
  int nameTextW = 0;
  for (const std::string& name : names)
    nameTextW = std::max(nameTextW, textWidth(name.c_str(), con));
  const int nameW = std::clamp(nameTextW + kColumnTextPadding,
                               kMinNameColumnWidth, kMaxNameColumnWidth);
  const int groupH = kStripsPerGroup * stripH + kStripGap;
  const int panelH = kPanelSidesPerAxis * kPanelPadding + headerH +
                     numContacts * (groupH + kGroupGap) + axisH;
  const int panelW = viewport.width - kPanelSidesPerAxis * kPanelMargin;
  const int panelL = kPanelMargin;
  const int panelB = kPanelMargin;
  const int x0 = panelL + kPanelPadding + nameW + tagW;
  const int x1 = panelL + panelW - kPanelPadding;
  const int stripW = x1 - x0;
  if (stripW < kMinStripWidth ||
      panelH > viewport.height / kMinViewportToPanelHeightRatio)
    return;
  fillRect(panelL, panelB, panelW, panelH, kPanelBackground);

  const std::function<int(double)> xOf = [&](double time) {
    const double fraction = std::clamp((time - start) / window, 0.0, 1.0);
    return x0 + static_cast<int>(std::lround(fraction * stripW));
  };

  // Header: title and legend.
  const int headerB = panelB + panelH - kPanelPadding - headerH;
  const char* absl_nonnull title = "contact timeline [b]";
  drawLabel(panelL + kPanelPadding, headerB,
            std::max(nameW + tagW, textWidth(title, con) + kLabelTextPadding),
            headerH, title, con);
  {
    int lx = x0;
    const std::function<void(const Rgba&, const char* absl_nonnull)> legend =
        [&](const Rgba& color, const char* absl_nonnull text) {
          const int textW = textWidth(text, con) + kLabelTextPadding;
          if (lx + kLegendSwatchRoom + textW > x1)
            return;  // no room: the rest of the legend is dropped
          fillRect(lx, headerB + kLegendSwatchInset, fontH - kLegendSwatchInset,
                   headerH - kLegendSwatchHeightInset, color);
          lx += fontH;
          drawLabel(lx, headerB, textW, headerH, text, con);
          lx += textW + kLegendEntryGap;
        };
    legend(kPlanContact, "plan: contact");
    legend(kSimContact, "sim: contact");
    legend(kSimEarly, "sim touching, plan swing");
    legend(kSimLate, "sim in air, plan contact");
  }

  // One group of two strips per contact point: the plan on top, the physics
  // below.
  const int stripsTop = headerB - kGroupGap;
  int groupTop = stripsTop;
  for (int contact = 0; contact < numContacts; ++contact) {
    const int targetB = groupTop - stripH;
    const int actualB = targetB - kStripGap - stripH;
    drawLabel(panelL + kPanelPadding, actualB, nameW, groupH,
              names[contact].c_str(), con);
    drawLabel(panelL + kPanelPadding + nameW, targetB, tagW, stripH, "plan",
              con);
    drawLabel(panelL + kPanelPadding + nameW, actualB, tagW, stripH, "sim",
              con);

    if (((unresolved >> contact) & 1u) != 0u) {
      fillRect(x0, actualB, stripW, groupH, kUnknown);
      drawLabel(x0, actualB, stripW, groupH,
                "no MuJoCo body resolved for this contact frame", con);
      groupTop = actualB - kGroupGap;
      continue;
    }

    // Runs of equal state are drawn as single bars.
    size_t runStart = 0;
    for (size_t k = 1; k <= samples.size(); ++k) {
      const int code = contactStateCode(samples[runStart], contact);
      if (k < samples.size() && contactStateCode(samples[k], contact) == code)
        continue;
      const double runEnd = (k < samples.size()) ? samples[k].time : now;
      if (runEnd >= start) {
        const int xa = xOf(samples[runStart].time);
        const int xb = std::max(xOf(runEnd), xa + 1);
        const bool touching = (code & kTouchingBit) != 0;
        const bool planContact = (code & kPlanContactBit) != 0;
        const bool planKnown = (code & kPlanKnownBit) != 0;
        fillRect(xa, targetB, xb - xa, stripH,
                 !planKnown ? kUnknown : (planContact ? kPlanContact : kSwing));
        const Rgba& simColor = !planKnown ? (touching ? kSimContact : kSwing)
                               : touching
                                   ? (planContact ? kSimContact : kSimEarly)
                                   : (planContact ? kSimLate : kSwing);
        fillRect(xa, actualB, xb - xa, stripH, simColor);
      }
      runStart = k;
    }
    groupTop = actualB - kGroupGap;
  }

  // Time axis: one tick per second, relative to now, at fixed positions.
  const int axisB = panelB + kPanelPadding;
  const int ticksH = stripsTop - (axisB + axisH);
  for (int secondsAgo = 0; secondsAgo <= static_cast<int>(window);
       ++secondsAgo) {
    const int x = xOf(now - secondsAgo);
    fillRect(x, axisB + axisH, /*width=*/1, ticksH, kTick);
    const std::string text = tickLabel(secondsAgo);
    const int labelW = textWidth(text.c_str(), con) + kLabelTextPadding;
    drawLabel(std::clamp(x - labelW / kTickLabelCenteringDivisor, panelL,
                         x1 - labelW),
              axisB, labelW, axisH, text.c_str(), con);
  }
}

}  // namespace robot::mujoco_sim_interface
