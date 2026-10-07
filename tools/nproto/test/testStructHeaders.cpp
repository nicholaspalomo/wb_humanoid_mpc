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

// The struct headers nproto generates hold plain C++ only: this file includes them alone, without protobuf or Abseil,
// and compiles with -Wall -Wextra -Wpedantic -Werror.

#include "gtest/gtest.h"

#include "nproto_test/other/point.nproto.h"
#include "nproto_test/other/unit.nproto.h"
#include "tools/nproto/test/defaults.nproto.h"
#include "tools/nproto/test/editions.nproto.h"
#include "tools/nproto/test/empty.nproto.h"
#include "tools/nproto/test/imports.nproto.h"
#include "tools/nproto/test/maps.nproto.h"
#include "tools/nproto/test/oneofs.nproto.h"
#include "tools/nproto/test/optionals.nproto.h"
#include "tools/nproto/test/outer.nproto.h"
#include "tools/nproto/test/repeated_fields.nproto.h"
#include "tools/nproto/test/scalars.nproto.h"
#include "tools/nproto/test/severity.nproto.h"
#include "tools/nproto/test/shadowing.nproto.h"

#ifdef GOOGLE_PROTOBUF_VERSION
#error "a struct header included protobuf"
#endif
#ifdef ABSL_BASE_CONFIG_H_
#error "a struct header included Abseil"
#endif

namespace nproto::test {
namespace {

TEST(StructHeaderTest, StructsAreUsableWithoutProtobuf) {
  Imports imports;
  imports.point.x = 1.0;
  imports.points.push_back(imports.point);
  imports.point_by_name["origin"] = other::Point();
  imports.unit = other::Unit::kFoot;
  EXPECT_TRUE(imports != Imports());
  EXPECT_TRUE(Empty() == Empty());
  EXPECT_FALSE(Empty() != Empty());
  EXPECT_TRUE(Defaults() == Defaults());
  EXPECT_TRUE(Maps() == Maps());
  EXPECT_TRUE(Oneofs() == Oneofs());
  EXPECT_TRUE(Optionals() == Optionals());
  EXPECT_TRUE(Editions() == Editions());
  EXPECT_TRUE(RepeatedFields() == RepeatedFields());
  EXPECT_TRUE(Shadowing() == Shadowing());
  EXPECT_EQ(static_cast<int>(Severity::kHigh), 2);
}

}  // namespace
}  // namespace nproto::test
