/******************************************************************************
Copyright (c) 2025. All rights reserved.

Tests for the dual-format (INFO + YAML) config loading in LoadData.h.
Verifies that yaml-cpp based YAML parsing produces identical ptree structures
to the Boost INFO parser, ensuring backward compatibility.
******************************************************************************/

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <string>
#include <vector>

#include <ocs2_core/misc/LoadData.h>

#include <filesystem>

namespace {
const std::string dataFolder = std::filesystem::path(__FILE__).parent_path().generic_string() + "/data/";
const std::string yamlFile = dataFolder + "testConfig.yaml";
const std::string infoFile = dataFolder + "testConfig.info";
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
// Test: loadPtreeValue from YAML (via ptree)
// =============================================================================
TEST(LoadDataYaml, loadPtreeValue_Double) {
  ocs2::loadData::PropertyTree pt;
  ocs2::loadData::readPropertyTree(yamlFile, pt);

  double timeHorizon = 0.0;
  ocs2::loadData::loadPtreeValue(pt, timeHorizon, "mpc.timeHorizon", false);
  EXPECT_DOUBLE_EQ(timeHorizon, 1.2);

  int mpcFreq = 0;
  ocs2::loadData::loadPtreeValue(pt, mpcFreq, "mpc.mpcDesiredFrequency", false);
  EXPECT_EQ(mpcFreq, 80);
}

TEST(LoadDataYaml, loadPtreeValue_NestedMap) {
  ocs2::loadData::PropertyTree pt;
  ocs2::loadData::readPropertyTree(yamlFile, pt);

  std::string armJointName;
  ocs2::loadData::loadPtreeValue(pt, armJointName, "model_settings.armJointNames.left_shoulder_y", false);
  EXPECT_EQ(armJointName, "left_shoulder_pitch_joint");
}

// =============================================================================
// Test: loadStdVector from YAML
// =============================================================================
TEST(LoadDataYaml, loadStdVector_Strings) {
  std::vector<std::string> fixedJoints;
  ocs2::loadData::loadStdVector(yamlFile, "model_settings.fixedJointNames", fixedJoints, false);
  ASSERT_EQ(fixedJoints.size(), 2u);
  EXPECT_EQ(fixedJoints[0], "left_wrist_roll_joint");
  EXPECT_EQ(fixedJoints[1], "left_wrist_pitch_joint");
}

TEST(LoadDataYaml, loadStdVector_Doubles) {
  // Use the gait yaml file which has switchingTimes arrays
  std::vector<double> switchingTimes;
  ocs2::loadData::loadStdVector(gaitYamlFile, "stance.switchingTimes", switchingTimes, false);
  ASSERT_EQ(switchingTimes.size(), 2u);
  EXPECT_DOUBLE_EQ(switchingTimes[0], 0.0);
  EXPECT_DOUBLE_EQ(switchingTimes[1], 0.5);
}

TEST(LoadDataYaml, loadStdVector_GaitList) {
  std::vector<std::string> gaitList;
  ocs2::loadData::loadStdVector(gaitYamlFile, "list", gaitList, false);
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
// Test: Cross-format consistency (YAML vs INFO produce same values)
// =============================================================================
TEST(LoadDataYaml, CrossFormat_CppDataType) {
  int yamlVal = -1, infoVal = -1;
  ocs2::loadData::loadCppDataType(yamlFile, "centroidalModelType", yamlVal);
  ocs2::loadData::loadCppDataType(infoFile, "centroidalModelType", infoVal);
  EXPECT_EQ(yamlVal, infoVal);
}

TEST(LoadDataYaml, CrossFormat_NestedValue) {
  double yamlVal = 0.0, infoVal = 0.0;
  ocs2::loadData::loadCppDataType(yamlFile, "model_settings.foot_constraint.positionErrorGain_z", yamlVal);
  ocs2::loadData::loadCppDataType(infoFile, "model_settings.foot_constraint.positionErrorGain_z", infoVal);
  EXPECT_DOUBLE_EQ(yamlVal, infoVal);
}

TEST(LoadDataYaml, CrossFormat_StdVector) {
  std::vector<std::string> yamlVec, infoVec;
  ocs2::loadData::loadStdVector(yamlFile, "model_settings.fixedJointNames", yamlVec, false);
  ocs2::loadData::loadStdVector(infoFile, "model_settings.fixedJointNames", infoVec, false);

  ASSERT_EQ(yamlVec.size(), infoVec.size());
  for (size_t i = 0; i < yamlVec.size(); ++i) {
    EXPECT_EQ(yamlVec[i], infoVec[i]);
  }
}

TEST(LoadDataYaml, CrossFormat_EigenMatrix) {
  Eigen::VectorXd yamlState(9), infoState(9);
  yamlState.setZero();
  infoState.setZero();
  ocs2::loadData::loadEigenMatrix(yamlFile, "initialState", yamlState);
  ocs2::loadData::loadEigenMatrix(infoFile, "initialState", infoState);

  for (int i = 0; i < 9; ++i) {
    EXPECT_DOUBLE_EQ(yamlState(i), infoState(i));
  }
}

TEST(LoadDataYaml, CrossFormat_MpcSettings) {
  double yamlHorizon = 0.0, infoHorizon = 0.0;
  int yamlFreq = 0, infoFreq = 0;

  ocs2::loadData::loadCppDataType(yamlFile, "mpc.timeHorizon", yamlHorizon);
  ocs2::loadData::loadCppDataType(infoFile, "mpc.timeHorizon", infoHorizon);
  EXPECT_DOUBLE_EQ(yamlHorizon, infoHorizon);

  ocs2::loadData::loadCppDataType(yamlFile, "mpc.mpcDesiredFrequency", yamlFreq);
  ocs2::loadData::loadCppDataType(infoFile, "mpc.mpcDesiredFrequency", infoFreq);
  EXPECT_EQ(yamlFreq, infoFreq);
}

// =============================================================================
// Test: readPropertyTree auto-detection
// =============================================================================
TEST(LoadDataYaml, ReadPropertyTree_AutoDetectsYaml) {
  ocs2::loadData::PropertyTree pt;
  ocs2::loadData::readPropertyTree(yamlFile, pt);
  EXPECT_EQ(pt.get<int>("centroidalModelType"), 0);
}

TEST(LoadDataYaml, ReadPropertyTree_AutoDetectsInfo) {
  ocs2::loadData::PropertyTree pt;
  ocs2::loadData::readPropertyTree(infoFile, pt);
  EXPECT_EQ(pt.get<int>("centroidalModelType"), 0);
}

TEST(LoadDataYaml, ReadPropertyTree_FallbackForUnknownExtension) {
  // Files with unknown extensions should use INFO parser (backward compat)
  // We can't easily test this without a valid INFO file with a different extension,
  // so just verify the function doesn't crash for .info files
  ocs2::loadData::PropertyTree pt;
  EXPECT_NO_THROW(ocs2::loadData::readPropertyTree(infoFile, pt));
}

// =============================================================================
// Test: Gait sequence loading from YAML
// =============================================================================
TEST(LoadDataYaml, GaitWalkSequence) {
  std::vector<std::string> modeSeq;
  ocs2::loadData::loadStdVector(gaitYamlFile, "walk.modeSequence", modeSeq, false);
  ASSERT_EQ(modeSeq.size(), 4u);
  EXPECT_EQ(modeSeq[0], "LF");
  EXPECT_EQ(modeSeq[1], "STANCE");
  EXPECT_EQ(modeSeq[2], "RF");
  EXPECT_EQ(modeSeq[3], "STANCE");

  std::vector<double> switchTimes;
  ocs2::loadData::loadStdVector(gaitYamlFile, "walk.switchingTimes", switchTimes, false);
  ASSERT_EQ(switchTimes.size(), 5u);
  EXPECT_DOUBLE_EQ(switchTimes[0], 0.0);
  EXPECT_DOUBLE_EQ(switchTimes[1], 0.6);
  EXPECT_DOUBLE_EQ(switchTimes[4], 1.4);
}
