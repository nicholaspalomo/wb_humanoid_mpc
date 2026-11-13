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

#include <cstddef>
#include <string>
#include <utility>

#include "gtest/gtest.h"
#include "ocs2_core/Types.h"

#include "legged_common_mpc/common/ContactArray.h"
#include "legged_common_mpc/common/Types.h"

/**
 * ContactArray: one value per contact of a set sized at run time, at most
 * kMaxNumContacts, whose iteration and equality cover the set's contacts only.
 */
namespace ocs2::legged {
namespace {

TEST(ContactArray, ADefaultArrayHasNoContacts) {
  const ContactArray<scalar_t> array;
  EXPECT_EQ(array.size(), 0U);
  EXPECT_TRUE(array.empty());
  EXPECT_EQ(array.begin(), array.end());
}

TEST(ContactArray, TheSizedConstructorValueInitializesTheEntries) {
  const ContactArray<scalar_t> numbers(4);
  EXPECT_EQ(numbers.size(), 4U);
  EXPECT_FALSE(numbers.empty());
  for (const scalar_t value : numbers) EXPECT_EQ(value, 0.0);

  const ContactArray<std::string> names(3);
  for (const std::string& name : names) EXPECT_TRUE(name.empty());
}

TEST(ContactArray, TheFilledConstructorSetsEveryEntry) {
  const ContactArray<int> array(6, 7);
  ASSERT_EQ(array.size(), 6U);
  for (size_t i = 0; i < array.size(); ++i) EXPECT_EQ(array[i], 7);
}

TEST(ContactArray, FillSetsTheContactsAndAssignResizes) {
  ContactArray<int> array(2, 1);
  array.fill(5);
  EXPECT_EQ(array[0], 5);
  EXPECT_EQ(array[1], 5);

  array.assign(4, 9);
  ASSERT_EQ(array.size(), 4U);
  for (const int value : array) EXPECT_EQ(value, 9);

  array.assign(kMaxNumContacts, 3);
  EXPECT_EQ(array.size(), kMaxNumContacts);
  array.assign(0, 3);
  EXPECT_TRUE(array.empty());
}

TEST(ContactArray, RangeForVisitsTheContactsOnly) {
  ContactArray<int> array(kMaxNumContacts, 1);
  array.assign(3, 2);
  size_t visited = 0;
  int sum = 0;
  for (int& value : array) {
    ++visited;
    sum += value;
    value = 4;
  }
  EXPECT_EQ(visited, 3U);
  EXPECT_EQ(sum, 6);
  EXPECT_EQ(array.end() - array.begin(), 3);
  EXPECT_EQ(array.data(), array.begin());
  EXPECT_EQ(array[2], 4);
}

TEST(ContactArray, EqualityComparesTheSizeAndTheContactsOnly) {
  ContactArray<int> a(3, 1);
  ContactArray<int> b(kMaxNumContacts, /*value=*/2);
  b.assign(3, 1);
  // The storage past the contacts differs, and is not compared.
  EXPECT_EQ(a, b);
  b[1] = 0;
  EXPECT_NE(a, b);
  EXPECT_NE(a, ContactArray<int>(4, 1));

  const ContactArray<int> copy = a;
  EXPECT_EQ(copy, a);
  ContactArray<int> assigned;
  assigned = a;
  EXPECT_EQ(assigned, a);
  ContactArray<std::string> names(/*numContacts=*/2, /*value=*/"foot");
  const ContactArray<std::string> namesCopy = names;
  ContactArray<std::string> moved = std::move(names);
  EXPECT_EQ(moved, namesCopy);
}

TEST(ContactArray, HoldsEigenFixedSizeEntries) {
  ContactArray<vector6_t> sixVectors(/*numContacts=*/4, vector6_t::Zero());
  sixVectors[2] = vector6_t::Constant(1.5);
  EXPECT_TRUE(sixVectors[0].isZero());
  EXPECT_EQ(sixVectors[2], vector6_t::Constant(1.5));
  const ContactArray<vector6_t> copy = sixVectors;
  EXPECT_EQ(copy, sixVectors);

  ContactArray<matrix3_t> matrices(/*numContacts=*/2, matrix3_t::Identity());
  matrices.fill(matrix3_t::Zero());
  for (const matrix3_t& matrix : matrices) EXPECT_TRUE(matrix.isZero());
}

TEST(ContactArrayDeathTest, MoreContactsThanTheCapacityAreRefused) {
  EXPECT_DEATH(ContactArray<int>(kMaxNumContacts + 1), "kMaxNumContacts");
  EXPECT_DEATH(ContactArray<int>(kMaxNumContacts + 1, /*value=*/0),
               "kMaxNumContacts");
  ContactArray<int> array;
  EXPECT_DEATH(array.assign(kMaxNumContacts + 1, 0), "kMaxNumContacts");
}

}  // namespace
}  // namespace ocs2::legged
