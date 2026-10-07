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

#include "humanoid_mpc_validation/io/Sha256.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace ocs2::humanoid::validation {
namespace {

constexpr std::array<uint32_t, 64> kRoundConstants = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be,
    0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa,
    0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85,
    0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
    0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

constexpr std::array<uint32_t, 8> kInitialHash = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                                  0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

template <int kBits>
uint32_t rotateRight(uint32_t value) {
  return (value >> kBits) | (value << (32 - kBits));
}

/** One 64-byte block into the running hash. */
void compressBlock(const uint8_t* absl_nonnull block, std::array<uint32_t, 8>& hash) {
  std::array<uint32_t, 64> schedule{};
  for (size_t i = 0; i < 16; ++i) {
    schedule[i] = (static_cast<uint32_t>(block[4 * i]) << 24) | (static_cast<uint32_t>(block[4 * i + 1]) << 16) |
                  (static_cast<uint32_t>(block[4 * i + 2]) << 8) | static_cast<uint32_t>(block[4 * i + 3]);
  }
  for (size_t i = 16; i < 64; ++i) {
    const uint32_t s0 = rotateRight<7>(schedule[i - 15]) ^ rotateRight<18>(schedule[i - 15]) ^ (schedule[i - 15] >> 3);
    const uint32_t s1 = rotateRight<17>(schedule[i - 2]) ^ rotateRight<19>(schedule[i - 2]) ^ (schedule[i - 2] >> 10);
    schedule[i] = schedule[i - 16] + s0 + schedule[i - 7] + s1;
  }
  std::array<uint32_t, 8> v = hash;
  for (size_t i = 0; i < 64; ++i) {
    const uint32_t s1 = rotateRight<6>(v[4]) ^ rotateRight<11>(v[4]) ^ rotateRight<25>(v[4]);
    const uint32_t choice = (v[4] & v[5]) ^ (~v[4] & v[6]);
    const uint32_t temp1 = v[7] + s1 + choice + kRoundConstants[i] + schedule[i];
    const uint32_t s0 = rotateRight<2>(v[0]) ^ rotateRight<13>(v[0]) ^ rotateRight<22>(v[0]);
    const uint32_t majority = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
    const uint32_t temp2 = s0 + majority;
    v[7] = v[6];
    v[6] = v[5];
    v[5] = v[4];
    v[4] = v[3] + temp1;
    v[3] = v[2];
    v[2] = v[1];
    v[1] = v[0];
    v[0] = temp1 + temp2;
  }
  for (size_t i = 0; i < 8; ++i) hash[i] += v[i];
}

}  // namespace

std::string sha256Hex(absl::string_view data) {
  // The message, a 1 bit, zeros up to 56 bytes modulo 64, and the message length in bits as a big-endian 64-bit number.
  std::vector<uint8_t> message(data.begin(), data.end());
  const uint64_t bitLength = static_cast<uint64_t>(data.size()) * 8u;
  message.push_back(0x80);
  while (message.size() % 64 != 56) message.push_back(0x00);
  for (int shift = 56; shift >= 0; shift -= 8) message.push_back(static_cast<uint8_t>(bitLength >> shift));

  std::array<uint32_t, 8> hash = kInitialHash;
  for (size_t offset = 0; offset < message.size(); offset += 64) compressBlock(message.data() + offset, hash);

  static constexpr char kHexDigits[] = "0123456789abcdef";
  std::string hex;
  hex.reserve(64);
  for (const uint32_t word : hash) {
    for (int shift = 28; shift >= 0; shift -= 4) hex.push_back(kHexDigits[(word >> shift) & 0xfu]);
  }
  return hex;
}

absl::StatusOr<std::string> sha256HexOfFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) return absl::NotFoundError(absl::StrCat("[sha256HexOfFile] cannot read '", path, "'"));
  const std::string contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  return sha256Hex(contents);
}

}  // namespace ocs2::humanoid::validation
