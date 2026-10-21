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
#include <climits>
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

// FIPS 180-4, sections 1, 5 and 6.2: a padded message is hashed block by block, a block being sixteen 32-bit words, in
// 64 rounds each, into a state of eight words; the padding ends with the message length in bits, a 64-bit number.
constexpr int kWordBits = 32;
constexpr int kWordBytes = 4;
// A word's bytes are big-endian (FIPS 180-4, section 3.1): the indices of its third and fourth byte in a block's word, and
// the left shift that moves each of its first three bytes into place.
constexpr int kWordThirdByte = 2;
constexpr int kWordFourthByte = 3;
constexpr int kWordFirstByteShift = 24;
constexpr int kWordSecondByteShift = 16;
constexpr int kWordThirdByteShift = 8;
constexpr int kBlockBytes = 64;
constexpr int kBlockWords = 16;
constexpr int kRounds = 64;
constexpr int kHashWords = 8;
constexpr int kLengthFieldBytes = 8;
constexpr int kLengthFieldBits = kLengthFieldBytes * CHAR_BIT;
/** The byte of a block the length field starts at, which the padding's zeros fill up to. */
constexpr int kLengthFieldOffset = kBlockBytes - kLengthFieldBytes;
/** The first byte of the padding: a 1 bit and seven zeros. */
constexpr int kPaddingLeadByte = 0x80;

// The rotations and shifts of the functions of FIPS 180-4, section 4.1.2, (4.4) to (4.7).
constexpr int kBigSigma0FirstRotation = 2;
constexpr int kBigSigma0SecondRotation = 13;
constexpr int kBigSigma0ThirdRotation = 22;
constexpr int kBigSigma1FirstRotation = 6;
constexpr int kBigSigma1SecondRotation = 11;
constexpr int kBigSigma1ThirdRotation = 25;
constexpr int kSmallSigma0FirstRotation = 7;
constexpr int kSmallSigma0SecondRotation = 18;
constexpr int kSmallSigma0Shift = 3;
constexpr int kSmallSigma1FirstRotation = 17;
constexpr int kSmallSigma1SecondRotation = 19;
constexpr int kSmallSigma1Shift = 10;

// How far back the message schedule reaches (FIPS 180-4, section 6.2.2, step 1):
// W_t = sigma_1(W_{t-2}) + W_{t-7} + sigma_0(W_{t-15}) + W_{t-16}.
constexpr int kSmallSigma1Lag = 2;
constexpr int kPlainTermLag = 7;
constexpr int kSmallSigma0Lag = 15;
constexpr int kOldestTermLag = 16;

// The indices of the working variables a to h (FIPS 180-4, section 6.2.2) in the state, the order of the hash's words.
constexpr int kA = 0;
constexpr int kB = 1;
constexpr int kC = 2;
constexpr int kD = 3;
constexpr int kE = 4;
constexpr int kF = 5;
constexpr int kG = 6;
constexpr int kH = 7;

// The digest in hexadecimal: 4 bits a digit, each word from its most significant bits on.
constexpr int kBitsPerHexDigit = 4;
constexpr unsigned int kHexDigitMask = 0xfu;
constexpr int kDigestHexDigits = kHashWords * kWordBits / kBitsPerHexDigit;

constexpr std::array<uint32_t, kRounds> kRoundConstants = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be,
    0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa,
    0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85,
    0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
    0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

constexpr std::array<uint32_t, kHashWords> kInitialHash = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                                           0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

template <int kBits>
uint32_t rotateRight(uint32_t value) {
  return (value >> kBits) | (value << (kWordBits - kBits));
}

/** One 64-byte block into the running hash. */
void compressBlock(const uint8_t* absl_nonnull block, std::array<uint32_t, kHashWords>& hash) {
  std::array<uint32_t, kRounds> schedule{};
  for (size_t i = 0; i < kBlockWords; ++i) {
    schedule[i] = (static_cast<uint32_t>(block[kWordBytes * i]) << kWordFirstByteShift) |
                  (static_cast<uint32_t>(block[kWordBytes * i + 1]) << kWordSecondByteShift) |
                  (static_cast<uint32_t>(block[kWordBytes * i + kWordThirdByte]) << kWordThirdByteShift) |
                  static_cast<uint32_t>(block[kWordBytes * i + kWordFourthByte]);
  }
  for (size_t i = kBlockWords; i < kRounds; ++i) {
    const uint32_t s0 = rotateRight<kSmallSigma0FirstRotation>(schedule[i - kSmallSigma0Lag]) ^
                        rotateRight<kSmallSigma0SecondRotation>(schedule[i - kSmallSigma0Lag]) ^
                        (schedule[i - kSmallSigma0Lag] >> kSmallSigma0Shift);
    const uint32_t s1 = rotateRight<kSmallSigma1FirstRotation>(schedule[i - kSmallSigma1Lag]) ^
                        rotateRight<kSmallSigma1SecondRotation>(schedule[i - kSmallSigma1Lag]) ^
                        (schedule[i - kSmallSigma1Lag] >> kSmallSigma1Shift);
    schedule[i] = schedule[i - kOldestTermLag] + s0 + schedule[i - kPlainTermLag] + s1;
  }
  std::array<uint32_t, kHashWords> v = hash;
  for (size_t i = 0; i < kRounds; ++i) {
    const uint32_t s1 = rotateRight<kBigSigma1FirstRotation>(v[kE]) ^ rotateRight<kBigSigma1SecondRotation>(v[kE]) ^
                        rotateRight<kBigSigma1ThirdRotation>(v[kE]);
    const uint32_t choice = (v[kE] & v[kF]) ^ (~v[kE] & v[kG]);
    const uint32_t temp1 = v[kH] + s1 + choice + kRoundConstants[i] + schedule[i];
    const uint32_t s0 = rotateRight<kBigSigma0FirstRotation>(v[kA]) ^ rotateRight<kBigSigma0SecondRotation>(v[kA]) ^
                        rotateRight<kBigSigma0ThirdRotation>(v[kA]);
    const uint32_t majority = (v[kA] & v[kB]) ^ (v[kA] & v[kC]) ^ (v[kB] & v[kC]);
    const uint32_t temp2 = s0 + majority;
    v[kH] = v[kG];
    v[kG] = v[kF];
    v[kF] = v[kE];
    v[kE] = v[kD] + temp1;
    v[kD] = v[kC];
    v[kC] = v[kB];
    v[kB] = v[kA];
    v[kA] = temp1 + temp2;
  }
  for (size_t i = 0; i < kHashWords; ++i) hash[i] += v[i];
}

}  // namespace

std::string sha256Hex(absl::string_view data) {
  // The message, a 1 bit, zeros up to 56 bytes modulo 64, and the message length in bits as a big-endian 64-bit number.
  std::vector<uint8_t> message(data.begin(), data.end());
  const uint64_t bitLength = static_cast<uint64_t>(data.size()) * CHAR_BIT;
  message.push_back(kPaddingLeadByte);
  while (message.size() % kBlockBytes != kLengthFieldOffset) message.push_back(0x00);
  for (int shift = kLengthFieldBits - CHAR_BIT; shift >= 0; shift -= CHAR_BIT) {
    message.push_back(static_cast<uint8_t>(bitLength >> shift));
  }

  std::array<uint32_t, kHashWords> hash = kInitialHash;
  for (size_t offset = 0; offset < message.size(); offset += kBlockBytes) compressBlock(message.data() + offset, hash);

  static constexpr char kHexDigits[] = "0123456789abcdef";
  std::string hex;
  hex.reserve(kDigestHexDigits);
  for (const uint32_t word : hash) {
    for (int shift = kWordBits - kBitsPerHexDigit; shift >= 0; shift -= kBitsPerHexDigit) {
      hex.push_back(kHexDigits[(word >> shift) & kHexDigitMask]);
    }
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
