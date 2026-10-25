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

// The FNV-1a 64-bit constants of nproto/Fnv1a64Constants.h give the hashes of
// the reference test vectors of the algorithm.

#include <cstdint>

#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "nproto/Fnv1a64Constants.h"

namespace nproto {
namespace {

uint64_t fnv1a64(absl::string_view text) {
  uint64_t hash = kFnvOffsetBasis;
  for (const char character : text) {
    hash ^= static_cast<uint64_t>(static_cast<unsigned char>(character));
    hash *= kFnvPrime;
  }
  return hash;
}

TEST(Fnv1a64Constants, HashTheReferenceVectors) {
  EXPECT_EQ(fnv1a64(""), 0xcbf29ce484222325ULL);
  EXPECT_EQ(fnv1a64("a"), 0xaf63dc4c8601ec8cULL);
  EXPECT_EQ(fnv1a64("foobar"), 0x85944171f73967e8ULL);
}

TEST(Fnv1a64Constants, FoldAWordOctetByOctet) {
  EXPECT_EQ(kFnvWordOctets, 8);
  EXPECT_EQ(kFnvOctetMask, 0xffU);
}

}  // namespace
}  // namespace nproto
