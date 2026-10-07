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

#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/StateLayout.h"

namespace ocs2::humanoid {
namespace {

bool isLowercaseLetterDigitOrUnderscore(char character) {
  return (character >= 'a' && character <= 'z') || (character >= '0' && character <= '9') || character == '_';
}

TEST(StateLayout, LayoutTagIsOneVersionedFolderName) {
  // The tag is appended to the CppAD library folder as one path component: no separator, no dot, no upper case that a
  // case-insensitive file system would fold.
  ASSERT_FALSE(kStateLayoutTag.empty());
  for (const char character : kStateLayoutTag) {
    EXPECT_TRUE(isLowercaseLetterDigitOrUnderscore(character)) << "'" << character << "' in " << kStateLayoutTag;
  }
  // It ends in a version, "_v<N>", which a change of the layout increments.
  const size_t versionStart = kStateLayoutTag.rfind("_v");
  ASSERT_NE(versionStart, absl::string_view::npos) << kStateLayoutTag;
  const absl::string_view version = kStateLayoutTag.substr(versionStart + 2);
  ASSERT_FALSE(version.empty()) << kStateLayoutTag;
  for (const char character : version) {
    EXPECT_TRUE(character >= '0' && character <= '9') << kStateLayoutTag;
  }
}

}  // namespace
}  // namespace ocs2::humanoid
