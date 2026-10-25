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

#pragma once

#include <cstdint>

namespace nproto {

/*
 * FNV-1a, 64 bit (https://datatracker.ietf.org/doc/html/draft-eastlake-fnv):
 * the hash of the values that must be the same across processes and builds,
 * which absl::Hash deliberately is not - the schema fingerprints (Schema.h),
 * the fingerprints of the configuration dumps (tools/config_dump) and the
 * humanoid MPC's footprint fingerprints and basis-library keys. A hash starts
 * at kFnvOffsetBasis; each octet is XORed into it, and the hash is then
 * multiplied by kFnvPrime. A 64-bit word is folded in as its kFnvWordOctets
 * octets, the least significant first, each the word shifted down and masked
 * with kFnvOctetMask. The constants are here, without the rest of nproto,
 * because nproto is the lowest of the libraries that hash
 * (//tools/nproto:fnv1a64_constants has no dependencies).
 */
// LINT.IfChange(fnv1a)
inline constexpr uint64_t kFnvOffsetBasis = 0xcbf29ce484222325ULL;
inline constexpr uint64_t kFnvPrime = 0x100000001b3ULL;
// LINT.ThenChange(//tools/nproto/nproto_schema.py:fnv1a)
inline constexpr int kFnvWordOctets = static_cast<int>(sizeof(uint64_t));
inline constexpr uint64_t kFnvOctetMask = 0xffU;

}  // namespace nproto
