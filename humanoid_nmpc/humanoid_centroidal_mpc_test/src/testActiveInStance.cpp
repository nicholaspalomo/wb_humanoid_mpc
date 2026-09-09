/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.
******************************************************************************/

#include <gtest/gtest.h>
#include <boost/property_tree/ptree.hpp>
#include <filesystem>
#include <iostream>
#include <memory>

#include <humanoid_common_mpc/gait/GaitSchedule.h>
#include <humanoid_common_mpc/pinocchio_model/createPinocchioModel.h>
#include <humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h>
#include <humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h>
#include <ocs2_centroidal_model/CentroidalModelInfo.h>
#include <ocs2_centroidal_model/FactoryFunctions.h>
#include <ocs2_core/misc/LoadData.h>
#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_centroidal_mpc/cost/CentroidalMpcEndEffectorFootCost.h"

using namespace ocs2;
using namespace ocs2::humanoid;

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: Verify YAML Parsing for activeInStance across boolean and integer formats
// ─────────────────────────────────────────────────────────────────────────────
TEST(ActiveInStanceTest, VerifyYamlParsing) {
  const std::string atlasTaskFile =
      "/wb_humanoid_mpc_ws/workspace/wb_humanoid_mpc/robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml";
  ASSERT_TRUE(std::filesystem::exists(atlasTaskFile)) << "Atlas task.yaml not found: " << atlasTaskFile;

  boost::property_tree::ptree pt;
  loadData::readPropertyTree(atlasTaskFile, pt);

  bool activeInStance = false;
  EXPECT_NO_THROW({ loadData::loadPtreeValue(pt, activeInStance, "task_space_foot_cost_weights.activeInStance", false); });
  EXPECT_TRUE(activeInStance) << "Atlas task.yaml must have activeInStance = true";

  // Test with GUI format: "activeInStance: 1"
  std::string tempLiveYaml = std::filesystem::temp_directory_path() / "test_live.yaml";
  {
    std::ofstream ofs(tempLiveYaml);
    ofs << "task_space_foot_cost_weights:\n  activeInStance: 1\n";
  }
  boost::property_tree::ptree ptLive;
  loadData::readPropertyTree(tempLiveYaml, ptLive);
  bool activeInStanceLive = false;
  loadData::loadPtreeValue(ptLive, activeInStanceLive, "task_space_foot_cost_weights.activeInStance", false);
  EXPECT_TRUE(activeInStanceLive) << "activeInStance: 1 from GUI should parse as true";

  // Test with false format: "activeInStance: false"
  std::string tempFalseYaml = std::filesystem::temp_directory_path() / "test_false.yaml";
  {
    std::ofstream ofs(tempFalseYaml);
    ofs << "task_space_foot_cost_weights:\n  activeInStance: false\n";
  }
  boost::property_tree::ptree ptFalse;
  loadData::readPropertyTree(tempFalseYaml, ptFalse);
  bool activeInStanceFalse = true;
  loadData::loadPtreeValue(ptFalse, activeInStanceFalse, "task_space_foot_cost_weights.activeInStance", false);
  EXPECT_FALSE(activeInStanceFalse) << "activeInStance: false should parse as false";

  // Test with integer 0 format: "activeInStance: 0"
  std::string tempZeroYaml = std::filesystem::temp_directory_path() / "test_zero.yaml";
  {
    std::ofstream ofs(tempZeroYaml);
    ofs << "task_space_foot_cost_weights:\n  activeInStance: 0\n";
  }
  boost::property_tree::ptree ptZero;
  loadData::readPropertyTree(tempZeroYaml, ptZero);
  bool activeInStanceZero = true;
  loadData::loadPtreeValue(ptZero, activeInStanceZero, "task_space_foot_cost_weights.activeInStance", false);
  EXPECT_FALSE(activeInStanceZero) << "activeInStance: 0 should parse as false";

  std::filesystem::remove(tempLiveYaml);
  std::filesystem::remove(tempFalseYaml);
  std::filesystem::remove(tempZeroYaml);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: Verify CentroidalMpcEndEffectorFootCost::isActive(time) logic
// ─────────────────────────────────────────────────────────────────────────────
TEST(ActiveInStanceTest, VerifyFootCostIsActiveBehaviorWithAtlasModel) {
  const std::string atlasTaskFile =
      "/wb_humanoid_mpc_ws/workspace/wb_humanoid_mpc/robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml";
  const std::string atlasUrdfFile =
      "/wb_humanoid_mpc_ws/workspace/wb_humanoid_mpc/robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";
  const std::string atlasReferenceFile =
      "/wb_humanoid_mpc_ws/workspace/wb_humanoid_mpc/robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/command/reference.yaml";

  ASSERT_TRUE(std::filesystem::exists(atlasTaskFile));
  ASSERT_TRUE(std::filesystem::exists(atlasUrdfFile));
  ASSERT_TRUE(std::filesystem::exists(atlasReferenceFile));

  ModelSettings modelSettings(atlasTaskFile, atlasUrdfFile, "drc_atlas", false);
  PinocchioInterface pinocchioInterface(createCustomPinocchioInterface(atlasTaskFile, atlasUrdfFile, modelSettings, false));

  CentroidalModelInfo centroidalModelInfo = centroidal_model::createCentroidalModelInfo(
      pinocchioInterface, centroidal_model::loadCentroidalType(atlasTaskFile),
      centroidal_model::loadDefaultJointState(pinocchioInterface.getModel().nq - 6, atlasReferenceFile), modelSettings.contactNames3DoF,
      modelSettings.contactNames6DoF);

  CentroidalMpcRobotModel<scalar_t> mpcRobotModel(modelSettings, pinocchioInterface, centroidalModelInfo);
  CentroidalMpcRobotModel<ad_scalar_t> mpcRobotModelAD(modelSettings, pinocchioInterface.toCppAd(), centroidalModelInfo.toCppAd());

  std::unique_ptr<SwingTrajectoryPlanner> swingTrajectoryPlanner(
      new SwingTrajectoryPlanner(loadSwingTrajectorySettings(atlasTaskFile, "swing_trajectory_config", false), 2));

  auto gaitSchedule = GaitSchedule::loadGaitSchedule(atlasReferenceFile, modelSettings, false);
  ModeSchedule initialModeSchedule = gaitSchedule->getModeSchedule(0.0, 1.0);

  auto referenceManager = std::make_shared<SwitchedModelReferenceManager>(std::move(gaitSchedule), std::move(swingTrajectoryPlanner),
                                                                          pinocchioInterface, mpcRobotModel);
  referenceManager->setModeSchedule(initialModeSchedule);
  referenceManager->preSolverRun(0.0, 1.0, vector_t::Zero(centroidalModelInfo.stateDim), 3);

  // Time t=0.0 is double stance for DRC Atlas
  const scalar_t stanceTime = 0.0;
  ASSERT_TRUE(referenceManager->isInContact(stanceTime, 0)) << "Atlas left foot should be in contact at t=0";
  ASSERT_TRUE(referenceManager->isInContact(stanceTime, 1)) << "Atlas right foot should be in contact at t=0";

  EndEffectorKinematicsWeights weights = EndEffectorKinematicsWeights::getWeights(atlasTaskFile, "task_space_foot_cost_weights.", false);

  // 1. Check with activeInStance = false:
  CentroidalMpcEndEffectorFootCost footCostInactive(*referenceManager, weights, pinocchioInterface, mpcRobotModelAD, 0,
                                                    "foot_l_contact_TaskSpaceKinematicsCost", modelSettings, false /* activeInStance */);

  EXPECT_FALSE(footCostInactive.getActiveInStance());
  EXPECT_FALSE(footCostInactive.isActive(stanceTime)) << "When activeInStance is false, foot cost MUST be inactive during stance";

  // 2. Check with activeInStance = true:
  CentroidalMpcEndEffectorFootCost footCostActive(*referenceManager, weights, pinocchioInterface, mpcRobotModelAD, 0,
                                                  "foot_l_contact_TaskSpaceKinematicsCost", modelSettings, true /* activeInStance */);

  EXPECT_TRUE(footCostActive.getActiveInStance());
  EXPECT_TRUE(footCostActive.isActive(stanceTime)) << "When activeInStance is true, foot cost MUST BE ACTIVE during stance!";

  // 3. Test runtime dynamic toggling via setActiveInStance:
  footCostActive.setActiveInStance(false);
  EXPECT_FALSE(footCostActive.isActive(stanceTime));

  footCostActive.setActiveInStance(true);
  EXPECT_TRUE(footCostActive.isActive(stanceTime));

  // 4. Test master switch setActive:
  footCostActive.setActive(false);
  EXPECT_FALSE(footCostActive.isActive(stanceTime));

  footCostActive.setActive(true);
  EXPECT_TRUE(footCostActive.isActive(stanceTime));

  // 5. Test clone preserves activeInStance:
  std::unique_ptr<CentroidalMpcEndEffectorFootCost> clonedCost(footCostActive.clone());
  EXPECT_TRUE(clonedCost->getActiveInStance());
  EXPECT_TRUE(clonedCost->isActive(stanceTime));
}
