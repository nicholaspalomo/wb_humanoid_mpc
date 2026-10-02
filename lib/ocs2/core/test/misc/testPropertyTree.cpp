/******************************************************************************
Copyright (c) 2026. All rights reserved.

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

/*
 * ocs2::PropertyTree replaces boost::property_tree::ptree. Before Boost was removed, a parity test read every shipped
 * task file (and the GUI's live copies of them) and a document of edge cases into both, and required every lookup the
 * code makes - get, get with a default, getOptional, the immediate-child find and count, loadPtreeValue with its
 * printout, loadEigenMatrix, loadStdVector, loadCppDataType - to give the same value or the same kind of error, for
 * double, float, long double, every integer type, bool, the char types and std::string. It passed. CONVERSIONS is
 * ptree's stream_translator output from that run; the other tests pin the tree and path semantics it relied on.
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <ocs2_core/misc/LoadData.h>
#include <ocs2_core/misc/PropertyTree.h>

namespace ocs2 {
namespace {

struct Conversion {
  std::string data;
  std::optional<double> asDouble;
  std::optional<float> asFloat;
  std::optional<int> asInt;
  std::optional<long> asLong;
  std::optional<size_t> asSize;
  std::optional<bool> asBool;
  std::optional<char> asChar;
  std::optional<std::string> asString;
};

// clang-format off
const std::vector<Conversion> CONVERSIONS = {
    {"1", 1.0, 1.0f, 1, 1l, 1ul, true, char(49), std::string("1")},
    {"-2", -2.0, -2.0f, -2, -2l, 18446744073709551614ul, std::nullopt, std::nullopt, std::string("-2")},
    {"+3", 3.0, 3.0f, 3, 3l, 3ul, std::nullopt, std::nullopt, std::string("+3")},
    {"007", 7.0, 7.0f, 7, 7l, 7ul, std::nullopt, std::nullopt, std::string("007")},
    {"0x1F", std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("0x1F")},
    {"1e3", 1000.0, 1000.0f, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("1e3")},
    {"1.5", 1.5, 1.5f, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("1.5")},
    {" 4 ", 4.0, 4.0f, 4, 4l, 4ul, std::nullopt, std::nullopt, std::string(" 4 ")},
    {"4 ", 4.0, 4.0f, 4, 4l, 4ul, std::nullopt, std::nullopt, std::string("4 ")},
    {" 4", 4.0, 4.0f, 4, 4l, 4ul, std::nullopt, std::nullopt, std::string(" 4")},
    {"4x", std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("4x")},
    {"", std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("")},
    {"\t5\n", 5.0, 5.0f, 5, 5l, 5ul, std::nullopt, std::nullopt, std::string("\t5\n")},
    {"2147483648", 2147483648.0, 2147483648.0f, std::nullopt, 2147483648l, 2147483648ul, std::nullopt, std::nullopt, std::string("2147483648")},
    {"-2147483649", -2147483649.0, -2147483648.0f, std::nullopt, -2147483649l, 18446744071562067967ul, std::nullopt, std::nullopt, std::string("-2147483649")},
    {"18446744073709551616", 1.8446744073709552e+19, 1.8446744073709552e+19f, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("18446744073709551616")},
    {"-1", -1.0, -1.0f, -1, -1l, 18446744073709551615ul, std::nullopt, std::nullopt, std::string("-1")},
    {"4294967296", 4294967296.0, 4294967296.0f, std::nullopt, 4294967296l, 4294967296ul, std::nullopt, std::nullopt, std::string("4294967296")},
    {"32768", 32768.0, 32768.0f, 32768, 32768l, 32768ul, std::nullopt, std::nullopt, std::string("32768")},
    {".5", 0.5, 0.5f, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string(".5")},
    {"5.", 5.0, 5.0f, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("5.")},
    {"-0.0", -0.0, -0.0f, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("-0.0")},
    {"-0", -0.0, -0.0f, 0, 0l, 0ul, false, std::nullopt, std::string("-0")},
    {"1e400", std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("1e400")},
    {"-1e400", std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("-1e400")},
    {"inf", std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("inf")},
    {"nan", std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("nan")},
    {"1_000", std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("1_000")},
    {"1,5", std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("1,5")},
    {"1e", std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("1e")},
    {"1e+", std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("1e+")},
    {"-.5e-3", -0.00050000000000000001, -0.00050000002374872565f, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("-.5e-3")},
    {"4.9e-324", 4.9406564584124654e-324, 0.0f, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("4.9e-324")},
    {"1.7976931348623159e308", std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("1.7976931348623159e308")},
    {"3.5e38", 3.5e+38, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("3.5e38")},
    {"1e-46", 1e-46, 0.0f, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("1e-46")},
    {"+.5", 0.5, 0.5f, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("+.5")},
    {"- 1", std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("- 1")},
    {"1 2", std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("1 2")},
    {"true", std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, true, std::nullopt, std::string("true")},
    {"false", std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, std::nullopt, std::string("false")},
    {"True", std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("True")},
    {"FALSE", std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("FALSE")},
    {"yes", std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("yes")},
    {"on", std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("on")},
    {"0", 0.0, 0.0f, 0, 0l, 0ul, false, char(48), std::string("0")},
    {"1 ", 1.0, 1.0f, 1, 1l, 1ul, true, std::nullopt, std::string("1 ")},
    {"2", 2.0, 2.0f, 2, 2l, 2ul, std::nullopt, char(50), std::string("2")},
    {" true", std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, true, std::nullopt, std::string(" true")},
    {"true ", std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, true, std::nullopt, std::string("true ")},
    {"\ttrue", std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, true, std::nullopt, std::string("\ttrue")},
    {"tru", std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("tru")},
    {"truex", std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("truex")},
    {"00", 0.0, 0.0f, 0, 0l, 0ul, false, std::nullopt, std::string("00")},
    {"01", 1.0, 1.0f, 1, 1l, 1ul, true, std::nullopt, std::string("01")},
    {"1.0", 1.0, 1.0f, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("1.0")},
    {"x", std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, char(120), std::string("x")},
    {" ", std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, char(32), std::string(" ")},
    {"xy", std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("xy")},
    {" x", std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string(" x")},
    {"x ", std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("x ")},
    {"65", 65.0, 65.0f, 65, 65l, 65ul, std::nullopt, std::nullopt, std::string("65")},
    {"hello world", std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("hello world")},
    {"  padded  ", std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::string("  padded  ")},
};
// clang-format on

/** Equal values, and for floating point equal signs too, so that "-0.0" stays negative. */
template <typename T>
void expectSameValue(const std::optional<T>& actual, const std::optional<T>& expected, const std::string& what) {
  ASSERT_EQ(actual.has_value(), expected.has_value()) << what;
  if (!expected.has_value()) {
    return;
  }
  EXPECT_EQ(*actual, *expected) << what;
  if constexpr (std::is_floating_point_v<T>) {
    EXPECT_EQ(std::signbit(*actual), std::signbit(*expected)) << what;
  }
}

/** The node at `path` holding `data`, through getValueOptional, getValue, getOptional, get and get with a default. */
template <typename T>
void expectConversion(
    const PropertyTree& tree, const std::string& path, const std::optional<T>& expected, const T& defaultValue, const std::string& what) {
  const PropertyTree& node = tree.getChild(path);
  expectSameValue(node.getValueOptional<T>(), expected, what + " getValueOptional");
  expectSameValue(tree.getOptional<T>(path), expected, what + " getOptional");
  if (expected.has_value()) {
    expectSameValue(std::optional<T>(node.getValue<T>()), expected, what + " getValue");
    expectSameValue(std::optional<T>(tree.get<T>(path)), expected, what + " get");
    expectSameValue(std::optional<T>(tree.get<T>(path, defaultValue)), expected, what + " get with default");
  } else {
    EXPECT_THROW(node.getValue<T>(), PropertyTreeBadData) << what;
    EXPECT_THROW(tree.get<T>(path), PropertyTreeBadData) << what;
    // ptree's get(path, default) also falls back on a value that does not convert
    expectSameValue(std::optional<T>(tree.get<T>(path, defaultValue)), std::optional<T>(defaultValue), what + " get with default");
  }
}

TEST(PropertyTree, convertsDataExactlyAsPtreeDid) {
  for (const Conversion& conversion : CONVERSIONS) {
    PropertyTree tree;
    tree.addChild("value", PropertyTree(conversion.data));
    const std::string what = "data \"" + conversion.data + "\"";
    expectConversion<double>(tree, "value", conversion.asDouble, /*defaultValue=*/42.0, what + " as double");
    expectConversion<float>(tree, "value", conversion.asFloat, /*defaultValue=*/42.0f, what + " as float");
    expectConversion<int>(tree, "value", conversion.asInt, /*defaultValue=*/42, what + " as int");
    expectConversion<long>(tree, "value", conversion.asLong, /*defaultValue=*/42l, what + " as long");
    expectConversion<size_t>(tree, "value", conversion.asSize, /*defaultValue=*/42ul, what + " as size_t");
    expectConversion<bool>(tree, "value", conversion.asBool, /*defaultValue=*/true, what + " as bool");
    expectConversion<char>(tree, "value", conversion.asChar, /*defaultValue=*/'*', what + " as char");
    expectConversion<std::string>(tree, "value", conversion.asString, /*defaultValue=*/std::string("<default>"), what + " as string");
  }
}

TEST(PropertyTree, aStringIsTheDataVerbatimAndBoolAcceptsOnlyNumbersAndTrueFalse) {
  for (const char* data : {"", "  padded  ", "0x1F", "1.50", "true", "yes"}) {
    EXPECT_EQ(PropertyTree(data).getValue<std::string>(), data);
  }
  // YAML 1.1's other spellings of a boolean are not booleans to the tree
  for (const char* data : {"yes", "no", "on", "off", "y", "True", "FALSE", "2", "-1", "1.0"}) {
    EXPECT_FALSE(PropertyTree(data).getValueOptional<bool>().has_value()) << data;
  }
}

/** a: {b: {c: 1, list: [x, y]}, "(0,1)": 2, "key.with.dot": 3, "": empty, dup: first, dup: second} */
PropertyTree sampleTree() {
  PropertyTree root;
  PropertyTree& a = root.addChild("a");
  PropertyTree& b = a.addChild("b");
  b.addChild("c", PropertyTree("1"));
  PropertyTree& list = b.addChild("list");
  list.addChild("[0]", PropertyTree("x"));
  list.addChild("[1]", PropertyTree("y"));
  a.addChild("(0,1)", PropertyTree("2"));
  root.addChild("key.with.dot", PropertyTree("3"));
  root.addChild("", PropertyTree("empty"));
  root.addChild("dup", PropertyTree("first"));
  root.addChild("dup", PropertyTree("second"));
  return root;
}

TEST(PropertyTree, pathsAreKeysJoinedByDots) {
  const PropertyTree tree = sampleTree();
  EXPECT_EQ(tree.get<int>("a.b.c"), 1);
  EXPECT_EQ(tree.get<std::string>("a.b.list.[1]"), "y");
  EXPECT_EQ(tree.get<int>("a.(0,1)"), 2);
  // the empty path is the node itself, and a trailing separator is dropped
  EXPECT_EQ(&tree.getChild(""), &tree);
  EXPECT_EQ(&tree.getChild("a."), &tree.getChild("a"));
  // an empty key between two separators, or before the first, is the key ""
  EXPECT_EQ(tree.get<std::string>("."), "empty");
  EXPECT_EQ(tree.findChild(".a"), nullptr);
  EXPECT_EQ(tree.findChild("a..b"), nullptr);
  // a key with a '.' in it cannot be reached by path, but is an immediate child
  EXPECT_EQ(tree.findChild("key.with.dot"), nullptr);
  EXPECT_EQ(tree.count("key.with.dot"), 1u);
  EXPECT_EQ(tree.find("key.with.dot")->second.data(), "3");
  // a duplicate key: lookups find the first, count sees both
  EXPECT_EQ(tree.get<std::string>("dup"), "first");
  EXPECT_EQ(tree.count("dup"), 2u);
  // find and count are immediate-child lookups, not paths
  EXPECT_EQ(tree.find("a.b"), tree.end());
  EXPECT_EQ(tree.count("a.b"), 0u);
  EXPECT_EQ(tree.count("missing"), 0u);
  EXPECT_TRUE(loadData::containsPtreeValueFind(tree, "a"));
  EXPECT_FALSE(loadData::containsPtreeValueFind(tree, "a.b"));
}

TEST(PropertyTree, missingNodesAndUnconvertibleDataThrowNamedErrors) {
  const PropertyTree tree = sampleTree();
  try {
    tree.getChild("a.b.missing");
    ADD_FAILURE() << "no throw";
  } catch (const PropertyTreeBadPath& error) {
    EXPECT_EQ(std::string(error.what()), "No such node (a.b.missing)");
    EXPECT_EQ(error.path(), "a.b.missing");
  }
  EXPECT_THROW(tree.get<double>("a.missing"), PropertyTreeBadPath);
  try {
    tree.get<double>("a.b.list.[0]");
    ADD_FAILURE() << "no throw";
  } catch (const PropertyTreeBadData& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("\"x\""), std::string::npos) << message;
    EXPECT_NE(message.find("double"), std::string::npos) << message;
    EXPECT_NE(message.find("(a.b.list.[0])"), std::string::npos) << message;
    EXPECT_EQ(error.data(), "x");
  }
  // both are PropertyTreeErrors and std::runtime_errors, as ptree_bad_path and ptree_bad_data were ptree_errors
  EXPECT_THROW(tree.get<int>("missing"), PropertyTreeError);
  EXPECT_THROW(tree.get<int>("a.b.list.[0]"), PropertyTreeError);
  EXPECT_THROW(tree.get<int>("a.b.list.[0]"), std::runtime_error);
  // the lookups that do not throw
  EXPECT_EQ(tree.findChild("a.b.missing"), nullptr);
  EXPECT_FALSE(tree.getOptional<double>("a.missing").has_value());
  EXPECT_FALSE(tree.getOptional<double>("a.b.list.[0]").has_value());
  EXPECT_EQ(tree.get<double>("a.missing", /*defaultValue=*/2.5), 2.5);
  EXPECT_EQ(tree.get<double>("a.b.list.[0]", /*defaultValue=*/2.5), 2.5);
  EXPECT_EQ(tree.get("a.missing", /*defaultValue=*/"text"), "text");
  EXPECT_EQ(tree.get("a.b.c", /*defaultValue=*/"text"), "1");
}

TEST(PropertyTree, buildingKeepsOrderAndReferences) {
  PropertyTree tree;
  EXPECT_TRUE(tree.empty());
  PropertyTree& first = tree.addChild("first", PropertyTree("1"));
  for (int i = 0; i < 100; ++i) {
    tree.addChild("k" + std::to_string(i));
  }
  first.setData("changed");  // still the child: references survive more children
  EXPECT_EQ(tree.get<std::string>("first"), "changed");
  EXPECT_EQ(tree.size(), 101u);
  EXPECT_FALSE(tree.empty());
  std::vector<std::string> keys;
  for (const PropertyTree::value_type& child : tree) {
    keys.push_back(child.first);
  }
  EXPECT_EQ(keys.front(), "first");
  EXPECT_EQ(keys.back(), "k99");

  PropertyTree::iterator pushed = tree.push_back(PropertyTree::value_type("pushed", PropertyTree("p")));
  EXPECT_EQ(pushed->first, "pushed");
  EXPECT_EQ(std::prev(tree.end()), pushed);

  // a mutable lookup changes the tree
  tree.findChild("first")->setData("again");
  tree.getChild("pushed").addChild("leaf", PropertyTree("l"));
  EXPECT_EQ(tree.get<std::string>("first"), "again");
  EXPECT_EQ(tree.get<std::string>("pushed.leaf"), "l");

  // equality is data and ordered children, recursively; a copy is deep
  PropertyTree copy = tree;
  EXPECT_TRUE(copy == tree);
  copy.getChild("pushed.leaf").setData("other");
  EXPECT_TRUE(copy != tree);
  EXPECT_EQ(tree.get<std::string>("pushed.leaf"), "l");

  tree.clear();
  EXPECT_TRUE(tree.empty());
  EXPECT_EQ(tree.data(), "");
}

TEST(PropertyTree, assignmentReplacesTheWholeTree) {
  PropertyTree source;
  loadData::readPropertyTreeFromString("a: {b: 1, c: [x, y]}\nd: 2\n", source);

  // a copy assignment over a non-empty tree is a deep copy that leaves nothing of the old tree behind
  PropertyTree assigned("old data");
  assigned.addChild("old", PropertyTree("o"));
  assigned = source;
  EXPECT_TRUE(assigned == source);
  EXPECT_EQ(assigned.data(), "");
  EXPECT_EQ(assigned.count("old"), 0u);
  assigned.getChild("a.b").setData("changed");
  EXPECT_EQ(source.get<std::string>("a.b"), "1");

  // a move assignment takes the children over
  PropertyTree moved;
  PropertyTree copy = source;
  moved = std::move(copy);
  EXPECT_TRUE(moved == source);

  // assigning a node of the tree to the tree itself, by copy or by move, keeps that node's contents
  PropertyTree tree = source;
  tree = tree.getChild("a");
  EXPECT_EQ(tree.size(), 2u);
  EXPECT_EQ(tree.get<std::string>("b"), "1");
  EXPECT_EQ(tree.get<std::string>("c.[1]"), "y");
  tree = std::move(tree.getChild("c"));
  EXPECT_EQ(tree.size(), 2u);
  EXPECT_EQ(tree.get<std::string>("[0]"), "x");

  const PropertyTree& self = tree;
  tree = self;
  EXPECT_EQ(tree.get<std::string>("[1]"), "y");

  // swap exchanges data and children, and a reference to a child follows it
  PropertyTree left("l");
  PropertyTree& leftChild = left.addChild("only", PropertyTree("1"));
  PropertyTree right("r");
  left.swap(right);
  EXPECT_EQ(left.data(), "r");
  EXPECT_TRUE(left.empty());
  EXPECT_EQ(right.data(), "l");
  EXPECT_EQ(&right.getChild("only"), &leftChild);

  tree.clear();
  EXPECT_TRUE(tree.empty());
  EXPECT_EQ(tree.data(), "");
}

TEST(PropertyTree, yamlBecomesTheTreeTheLoadersRead) {
  PropertyTree tree;
  loadData::readPropertyTreeFromString(
      "scalar: 1.50\n"
      "quoted: ' padded '\n"
      "nulls: {a: , b: ~, c: null, d: [], e: {}}\n"
      "seq: [1, [2, 3], {k: v}]\n"
      "matrix: {scaling: 2.0, \"(1,0)\": 3}\n"
      "flag: true\n",
      tree);
  // scalars keep their text
  EXPECT_EQ(tree.get<std::string>("scalar"), "1.50");
  EXPECT_EQ(tree.get<std::string>("quoted"), " padded ");
  EXPECT_TRUE(tree.get<bool>("flag"));
  // a YAML null, an empty sequence and an empty map are all an empty node: no data, no children
  for (const char* key : {"nulls.a", "nulls.b", "nulls.c", "nulls.d", "nulls.e"}) {
    const PropertyTree& node = tree.getChild(key);
    EXPECT_TRUE(node.empty()) << key;
    EXPECT_EQ(node.data(), "") << key;
  }
  // sequences are children keyed [i], nested ones too
  EXPECT_EQ(tree.get<int>("seq.[0]"), 1);
  EXPECT_EQ(tree.get<int>("seq.[1].[1]"), 3);
  EXPECT_EQ(tree.get<std::string>("seq.[2].k"), "v");
  EXPECT_EQ(tree.getChild("seq").size(), 3u);
  // the order of a map is the order of the file
  std::vector<std::string> keys;
  for (const PropertyTree::value_type& child : tree) {
    keys.push_back(child.first);
  }
  EXPECT_EQ(keys, (std::vector<std::string>{"scalar", "quoted", "nulls", "seq", "matrix", "flag"}));
}

/** Writes `yaml` to a file under the test's temporary directory. */
std::string writeYaml(const std::string& name, const std::string& yaml) {
  const std::string path = (std::filesystem::path(testing::TempDir()) / name).string();
  std::ofstream out(path);
  out << yaml;
  return path;
}

/** Captures std::cerr while alive. */
class CerrCapture {
 public:
  CerrCapture() : old_(std::cerr.rdbuf(buffer_.rdbuf())) {}
  ~CerrCapture() { std::cerr.rdbuf(old_); }
  std::string text() const { return buffer_.str(); }

 private:
  std::stringstream buffer_;
  std::streambuf* old_;
};

TEST(LoadData, loadPtreeValueKeepsTheValueOfAMissingKeyAndThrowsOnBadData) {
  PropertyTree tree;
  loadData::readPropertyTreeFromString("a: {x: 2.5, bad: abc}\n", tree);
  double value = 1.0;
  loadData::loadPtreeValue(tree, value, "a.x", /*verbose=*/false);
  EXPECT_EQ(value, 2.5);
  loadData::loadPtreeValue(tree, value, "a.missing", /*verbose=*/false);
  EXPECT_EQ(value, 2.5);
  EXPECT_THROW(loadData::loadPtreeValue(tree, value, "a.bad", /*verbose=*/false), PropertyTreeBadData);
  EXPECT_EQ(value, 2.5);

  CerrCapture capture;
  loadData::loadPtreeValue(tree, value, "a.x", /*verbose=*/true);
  loadData::loadPtreeValue(tree, value, "a.missing", /*verbose=*/true);
  const std::string printout = capture.text();
  EXPECT_NE(printout.find(" #### 'x'"), std::string::npos) << printout;
  EXPECT_NE(printout.find("2.5\n"), std::string::npos) << printout;
  EXPECT_NE(printout.find(" #### 'missing'"), std::string::npos) << printout;
  EXPECT_NE(printout.find("2.5 (default)\n"), std::string::npos) << printout;
}

TEST(LoadData, loadEigenMatrixAppliesScalingAndDefaultLikePtreeDid) {
  const std::string file = writeYaml("loadEigenMatrix.yaml",
                                     "m: {scaling: 2.0, default: 0.5, \"(0,0)\": 1, \"(1,1)\": 2, \"(0,1)\": bad}\n"
                                     "badScaling: {scaling: two, \"(0,0)\": 3}\n"
                                     "empty: {scaling: 2.0}\n");
  Eigen::MatrixXd m = Eigen::MatrixXd::Constant(2, 2, 42.0);
  {
    CerrCapture capture;
    loadData::loadEigenMatrix(file, "m", m);
    EXPECT_NE(capture.text().find("WARNING: Loaded at least one default value in matrix: \"m\""), std::string::npos);
  }
  // an entry that is missing, or that does not convert, takes the default; every entry is scaled
  EXPECT_EQ(m(0, 0), 2.0);
  EXPECT_EQ(m(1, 1), 4.0);
  EXPECT_EQ(m(0, 1), 1.0);
  EXPECT_EQ(m(1, 0), 1.0);

  // a scaling that does not convert is ignored (1.0), as ptree's get(path, default) ignored it
  Eigen::VectorXd v = Eigen::VectorXd::Zero(1);
  loadData::loadEigenMatrix(file, "badScaling", v);
  EXPECT_EQ(v(0), 3.0);

  // no entry at all is an error naming the matrix and the file; so is an empty matrix
  Eigen::MatrixXd unloaded = Eigen::MatrixXd::Zero(2, 2);
  try {
    loadData::loadEigenMatrix(file, "empty", unloaded);
    ADD_FAILURE() << "no throw";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("\"empty\""), std::string::npos);
    EXPECT_NE(std::string(error.what()).find(file), std::string::npos);
  }
  Eigen::MatrixXd none(0, 3);
  EXPECT_THROW(loadData::loadEigenMatrix(file, "m", none), std::runtime_error);
}

TEST(LoadData, loadStdVectorStopsAtTheFirstMissingOrBadItem) {
  const std::string file = writeYaml("loadStdVector.yaml",
                                     "numbers: [1.5, -2, 3e2, x, 6]\n"
                                     "holey: {\"[0]\": 1, \"[2]\": 3}\n"
                                     "names: [a, \" b \"]\n");
  std::vector<double> numbers;
  loadData::loadStdVector(file, "numbers", numbers, /*verbose=*/false);
  EXPECT_EQ(numbers, (std::vector<double>{1.5, -2.0, 300.0}));
  std::vector<int> holey;
  loadData::loadStdVector(file, "holey", holey, /*verbose=*/false);
  EXPECT_EQ(holey, (std::vector<int>{1}));
  std::vector<std::string> names;
  loadData::loadStdVector(file, "names", names, /*verbose=*/false);
  EXPECT_EQ(names, (std::vector<std::string>{"a", " b "}));
  // nothing loaded: the previous contents are kept
  std::vector<double> kept = {7.0, 8.0};
  loadData::loadStdVector(file, "missing", kept, /*verbose=*/false);
  EXPECT_EQ(kept, (std::vector<double>{7.0, 8.0}));

  CerrCapture capture;
  loadData::loadStdVector(file, "names", names, /*verbose=*/true);
  EXPECT_EQ(capture.text(), " #### 'names': {a,  b , \b\b}\n");
}

TEST(LoadData, loadCppDataTypeThrowsOnAMissingKey) {
  const std::string file = writeYaml("loadCppDataType.yaml", "a: {b: 3}\n");
  size_t value = 0;
  loadData::loadCppDataType(file, "a.b", value);
  EXPECT_EQ(value, 3u);
  EXPECT_THROW(loadData::loadCppDataType(file, "a.c", value), PropertyTreeBadPath);
}

}  // namespace
}  // namespace ocs2
