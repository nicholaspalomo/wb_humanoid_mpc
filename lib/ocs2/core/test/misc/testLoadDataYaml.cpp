/******************************************************************************
Copyright (c) 2025. All rights reserved.

Tests for the YAML config loading in LoadData.h: the loaders read YAML task files into an ocs2::PropertyTree.
******************************************************************************/

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <ocs2_core/misc/LoadData.h>

namespace {
const std::string dataFolder = std::filesystem::path(__FILE__).parent_path().generic_string() + "/data/";
const std::string yamlFile = dataFolder + "testConfig.yaml";
const std::string gaitYamlFile = dataFolder + "testGait.yaml";
}  // namespace

// =============================================================================
// Test: loadCppDataType from YAML
// =============================================================================
TEST(LoadDataYaml, loadCppDataType_Scalar) {
  int centroidalModelType = -1;
  ocs2::loadData::loadCppDataType(yamlFile, "centroidalModelType", centroidalModelType);
  EXPECT_EQ(centroidalModelType, 0);
}

TEST(LoadDataYaml, loadCppDataType_Bool) {
  bool verbose = false;
  ocs2::loadData::loadCppDataType(yamlFile, "interface.verbose", verbose);
  EXPECT_TRUE(verbose);
}

TEST(LoadDataYaml, loadCppDataType_String) {
  std::string robotName;
  ocs2::loadData::loadCppDataType(yamlFile, "model_settings.robotName", robotName);
  EXPECT_EQ(robotName, "test_robot");
}

TEST(LoadDataYaml, loadCppDataType_NestedScalar) {
  double posErrGain = 0.0;
  ocs2::loadData::loadCppDataType(yamlFile, "model_settings.foot_constraint.positionErrorGain_z", posErrGain);
  EXPECT_DOUBLE_EQ(posErrGain, 5.0);
}

// =============================================================================
// Test: loadPtreeValue from YAML (via PropertyTree)
// =============================================================================
TEST(LoadDataYaml, loadPtreeValue_Double) {
  ocs2::PropertyTree pt;
  ocs2::loadData::readPropertyTree(yamlFile, pt);

  double timeHorizon = 0.0;
  ocs2::loadData::loadPtreeValue(pt, timeHorizon, "mpc.timeHorizon", /*verbose=*/false);
  EXPECT_DOUBLE_EQ(timeHorizon, 1.2);

  int mpcFreq = 0;
  ocs2::loadData::loadPtreeValue(pt, mpcFreq, "mpc.mpcDesiredFrequency", /*verbose=*/false);
  EXPECT_EQ(mpcFreq, 80);
}

TEST(LoadDataYaml, loadPtreeValue_NestedMap) {
  ocs2::PropertyTree pt;
  ocs2::loadData::readPropertyTree(yamlFile, pt);

  std::string armJointName;
  ocs2::loadData::loadPtreeValue(pt, armJointName, "model_settings.armJointNames.left_shoulder_y", /*verbose=*/false);
  EXPECT_EQ(armJointName, "left_shoulder_pitch_joint");
}

// =============================================================================
// Test: loadStdVector from YAML
// =============================================================================
TEST(LoadDataYaml, loadStdVector_Strings) {
  std::vector<std::string> fixedJoints;
  ocs2::loadData::loadStdVector(yamlFile, "model_settings.fixedJointNames", fixedJoints, /*verbose=*/false);
  ASSERT_EQ(fixedJoints.size(), 2u);
  EXPECT_EQ(fixedJoints[0], "left_wrist_roll_joint");
  EXPECT_EQ(fixedJoints[1], "left_wrist_pitch_joint");
}

TEST(LoadDataYaml, loadStdVector_Doubles) {
  // Use the gait yaml file which has switchingTimes arrays
  std::vector<double> switchingTimes;
  ocs2::loadData::loadStdVector(gaitYamlFile, "stance.switchingTimes", switchingTimes, /*verbose=*/false);
  ASSERT_EQ(switchingTimes.size(), 2u);
  EXPECT_DOUBLE_EQ(switchingTimes[0], 0.0);
  EXPECT_DOUBLE_EQ(switchingTimes[1], 0.5);
}

TEST(LoadDataYaml, loadStdVector_GaitList) {
  std::vector<std::string> gaitList;
  ocs2::loadData::loadStdVector(gaitYamlFile, "list", gaitList, /*verbose=*/false);
  ASSERT_EQ(gaitList.size(), 2u);
  EXPECT_EQ(gaitList[0], "stance");
  EXPECT_EQ(gaitList[1], "walk");
}

// =============================================================================
// Test: loadEigenMatrix from YAML
// =============================================================================
TEST(LoadDataYaml, loadEigenMatrix_InitialState) {
  Eigen::VectorXd state(9);
  state.setZero();
  ocs2::loadData::loadEigenMatrix(yamlFile, "initialState", state);

  EXPECT_DOUBLE_EQ(state(0), 0.0);
  EXPECT_DOUBLE_EQ(state(8), 0.79);
}

TEST(LoadDataYaml, loadEigenMatrix_Q_WithScaling) {
  Eigen::MatrixXd Q(4, 4);
  Q.setZero();
  ocs2::loadData::loadEigenMatrix(yamlFile, "Q", Q);

  // scaling = 1.0, so values should match directly
  EXPECT_DOUBLE_EQ(Q(0, 0), 8.0);
  EXPECT_DOUBLE_EQ(Q(1, 1), 8.0);
  EXPECT_DOUBLE_EQ(Q(2, 2), 15.0);
  EXPECT_DOUBLE_EQ(Q(3, 3), 15.0);
  EXPECT_DOUBLE_EQ(Q(0, 1), 0.0);  // off-diagonal should be zero
}

TEST(LoadDataYaml, loadEigenMatrix_R_WithScaling) {
  Eigen::MatrixXd R(3, 3);
  R.setZero();
  ocs2::loadData::loadEigenMatrix(yamlFile, "R", R);

  // scaling = 0.001
  EXPECT_DOUBLE_EQ(R(0, 0), 0.001 * 0.05);
  EXPECT_DOUBLE_EQ(R(1, 1), 0.001 * 0.05);
  EXPECT_DOUBLE_EQ(R(2, 2), 0.001 * 0.01);
}

// =============================================================================
// Test: readPropertyTree reads YAML only
// =============================================================================
TEST(LoadDataYaml, ReadPropertyTree_ReadsYaml) {
  ocs2::PropertyTree pt;
  ocs2::loadData::readPropertyTree(yamlFile, pt);
  EXPECT_EQ(pt.get<int>("centroidalModelType"), 0);
}

TEST(LoadDataYaml, ReadPropertyTree_RefusesOtherExtensions) {
  // The Boost INFO format of upstream OCS2 is no longer read: a file without a .yaml / .yml extension is an error that
  // names the file, whether or not it exists.
  for (const std::string& file : {dataFolder + "testConfig.info", dataFolder + "testConfig", std::string("/no/such/dir.d/task")}) {
    ocs2::PropertyTree pt;
    try {
      ocs2::loadData::readPropertyTree(file, pt);
      ADD_FAILURE() << file << " was read";
    } catch (const std::invalid_argument& e) {
      EXPECT_NE(std::string(e.what()).find(file), std::string::npos) << e.what();
    }
    EXPECT_TRUE(pt.empty());
  }
}

TEST(LoadDataYaml, ReadPropertyTree_AcceptsYmlAndMissingYamlThrows) {
  const std::string yml = (std::filesystem::path(testing::TempDir()) / "testLoadDataYaml.yml").string();
  {
    std::ofstream out(yml);
    out << "a:\n  b: 2\n";
  }
  ocs2::PropertyTree pt;
  ocs2::loadData::readPropertyTree(yml, pt);
  EXPECT_EQ(pt.get<int>("a.b"), 2);

  ocs2::PropertyTree missing;
  EXPECT_THROW(ocs2::loadData::readPropertyTree(dataFolder + "doesNotExist.yaml", missing), YAML::Exception);
}

TEST(LoadDataYaml, ReadPropertyTreeFromString_MatchesTheFile) {
  std::ifstream stream(yamlFile);
  std::stringstream text;
  text << stream.rdbuf();

  ocs2::PropertyTree fromFile;
  ocs2::loadData::readPropertyTree(yamlFile, fromFile);
  ocs2::PropertyTree fromString;
  ocs2::loadData::readPropertyTreeFromString(text.str(), fromString);
  EXPECT_TRUE(fromString == fromFile);
  EXPECT_FALSE(fromString.empty());

  ocs2::PropertyTree malformed;
  EXPECT_THROW(ocs2::loadData::readPropertyTreeFromString("a: [1, 2\n", malformed), YAML::Exception);
}

TEST(LoadDataYaml, ReadPropertyTree_AppendsToTheTree) {
  ocs2::PropertyTree pt;
  ocs2::loadData::readPropertyTreeFromString("a: 1\n", pt);
  ocs2::loadData::readPropertyTreeFromString("a: 2\nb: 3\n", pt);
  EXPECT_EQ(pt.size(), 3u);
  EXPECT_EQ(pt.count("a"), 2u);
  EXPECT_EQ(pt.get<int>("a"), 1);  // the first child with the key
  EXPECT_EQ(pt.get<int>("b"), 3);
}

// =============================================================================
// Test: Gait sequence loading from YAML
// =============================================================================
TEST(LoadDataYaml, GaitWalkSequence) {
  std::vector<std::string> modeSeq;
  ocs2::loadData::loadStdVector(gaitYamlFile, "walk.modeSequence", modeSeq, /*verbose=*/false);
  ASSERT_EQ(modeSeq.size(), 4u);
  EXPECT_EQ(modeSeq[0], "LF");
  EXPECT_EQ(modeSeq[1], "STANCE");
  EXPECT_EQ(modeSeq[2], "RF");
  EXPECT_EQ(modeSeq[3], "STANCE");

  std::vector<double> switchTimes;
  ocs2::loadData::loadStdVector(gaitYamlFile, "walk.switchingTimes", switchTimes, /*verbose=*/false);
  ASSERT_EQ(switchTimes.size(), 5u);
  EXPECT_DOUBLE_EQ(switchTimes[0], 0.0);
  EXPECT_DOUBLE_EQ(switchTimes[1], 0.6);
  EXPECT_DOUBLE_EQ(switchTimes[4], 1.4);
}
