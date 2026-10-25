/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

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

#include <limits>
#include <string>

#include "gtest/gtest.h"

#include "web_viewer/ViewerCameraMode.h"
#include "web_viewer/ViewerStatus.h"

namespace robot::web_viewer {
namespace {

TEST(ViewerStatusTest, EncodesEveryField) {
  ViewerStatus status;
  status.camera = {.mode = ViewerCameraMode::kFree,
                   .azimuth = 90.0,
                   .elevation = -5.5,
                   .distance = 3.0,
                   .lookat = {0.0, 0.25, 0.5}};
  status.render = {.width = 1280,
                   .height = 720,
                   .targetFps = 30,
                   .measuredFps = 16.5,
                   .quality = 70};
  status.geomGroups = {true, false, true};
  status.visualizations.push_back({.name = "contact_timeline",
                                   .hotkey = 'c',
                                   .enabled = true,
                                   .description = "contacts over time"});
  status.visualizations.push_back({.name = "metrics",
                                   .hotkey = 0,
                                   .enabled = false,
                                   .description = "rates"});
  EXPECT_EQ(
      viewerStatusJson(status),
      R"({"camera":{"mode":"free","azimuth":90,"elevation":-5.5,"distance":3,"lookat":[0,0.25,0.5]},)"
      R"("render":{"width":1280,"height":720,"targetFps":30,"measuredFps":16.5,"quality":70},)"
      R"("geomGroups":[true,false,true],)"
      R"("visualizations":[{"name":"contact_timeline","hotkey":"c","enabled":true,"description":"contacts over time"},)"
      R"({"name":"metrics","hotkey":null,"enabled":false,"description":"rates"}]})");
}

TEST(ViewerStatusTest, NumbersJsonCannotSpellAreNull) {
  ViewerStatus status;
  status.camera.azimuth = std::numeric_limits<double>::quiet_NaN();
  status.render.measuredFps = std::numeric_limits<double>::infinity();
  const std::string json = viewerStatusJson(status);
  EXPECT_NE(json.find(R"("azimuth":null)"), std::string::npos) << json;
  EXPECT_NE(json.find(R"("measuredFps":null)"), std::string::npos) << json;
  EXPECT_NE(json.find(R"("mode":"tracking")"), std::string::npos) << json;
}

TEST(ViewerStatusTest, EscapesStrings) {
  EXPECT_EQ(jsonString("plain"), R"("plain")");
  EXPECT_EQ(jsonString(R"(a "quoted" \ path)"), R"("a \"quoted\" \\ path")");
  EXPECT_EQ(jsonString("line\nbreak\ttab\x01"), R"("line\nbreak\ttab\u0001")");
}

}  // namespace
}  // namespace robot::web_viewer
