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

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

#include "Eigen/Core"
#include "absl/strings/match.h"
#include "gtest/gtest.h"

#include "tools/config_dump/ValueDump.h"

namespace ocs2::humanoid::config_dump {
namespace {

/** The next double after `value` towards +infinity: one bit of difference. */
double nextUp(double value) {
  return std::nextafter(value, std::numeric_limits<double>::infinity());
}

TEST(ValueDumpTest, EqualBitsAreEqualLinesAndOneBitIsADifferentLine) {
  ValueDump a;
  ValueDump b;
  ValueDump c;
  a.addDouble("x", /*value=*/0.1);
  b.addDouble("x", /*value=*/0.1);
  c.addDouble("x", nextUp(0.1));
  EXPECT_EQ(a.text(), b.text());
  EXPECT_NE(a.text(), c.text());

  ValueDump zero;
  ValueDump negativeZero;
  zero.addDouble("x", /*value=*/0.0);
  negativeZero.addDouble("x", /*value=*/-0.0);
  EXPECT_NE(zero.text(), negativeZero.text()) << "-0.0 and 0.0 are different bits";
}

TEST(ValueDumpTest, ScalarsListsAndSectionsAreOneLineEach) {
  ValueDump dump;
  dump.addSection("model");
  dump.addInteger("count", /*value=*/-3);
  dump.addBool("flag", /*value=*/true);
  dump.addString("name", "a\"b");
  dump.addStrings("names", {"x", "y"});
  dump.addSizes("indices", {4, 2});
  EXPECT_EQ(dump.text(),
            "[model]\ncount = -3\nflag = true\nname = \"a\\\"b\"\nnames.size = 2\nnames[0] = \"x\"\nnames[1] = \"y\"\n"
            "indices.size = 2\nindices[0] = 4\nindices[1] = 2\n");
}

TEST(ValueDumpTest, AMatrixIsDumpedElementByElementAfterItsSize) {
  ValueDump dump;
  Eigen::MatrixXd matrix(2, 2);
  matrix << 1.0, 2.0, 3.0, 4.0;
  dump.addMatrix("m", matrix);
  EXPECT_TRUE(absl::StartsWith(dump.text(), "m.size = 2x2\nm(0,0) = ")) << dump.text();
  EXPECT_TRUE(absl::StrContains(dump.text(), "m(1,0) = 0x1.8p+1 (3)")) << dump.text();

  ValueDump diagonal;
  diagonal.addDiagonal("d", matrix);
  EXPECT_EQ(diagonal.text(), "d.size = 2x2\nd[0] = 0x1p+0 (1)\nd[1] = 0x1p+2 (4)\n");
}

TEST(ValueDumpTest, AFingerprintChangesWithAnyBitAndWithTheShape) {
  Eigen::MatrixXd matrix = Eigen::MatrixXd::Constant(3, 4, 0.25);
  const uint64_t hash = matrixHash(matrix);
  EXPECT_EQ(matrixHash(matrix), hash) << "the hash is a function of the matrix";
  for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
    for (Eigen::Index col = 0; col < matrix.cols(); ++col) {
      Eigen::MatrixXd changed = matrix;
      changed(row, col) = nextUp(changed(row, col));
      EXPECT_NE(matrixHash(changed), hash) << "element (" << row << "," << col << ")";
    }
  }
  const Eigen::MatrixXd reshaped = Eigen::MatrixXd::Constant(4, 3, 0.25);
  EXPECT_NE(matrixHash(reshaped), hash) << "the same elements in another shape";

  ValueDump dump;
  dump.addFingerprint("m", matrix);
  EXPECT_TRUE(absl::StartsWith(dump.text(), "m = 3x4 nonzeros 12 sum 0x1.8p+1 hash ")) << dump.text();
}

}  // namespace
}  // namespace ocs2::humanoid::config_dump
