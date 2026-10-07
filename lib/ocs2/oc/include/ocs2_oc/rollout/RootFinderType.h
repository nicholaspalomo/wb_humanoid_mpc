#pragma once

namespace ocs2 {

/**
 * Enum class for specifying root finding algorithm type.
 */
// LINT.IfChange(root_finder_types)
enum class RootFinderType { ANDERSON_BJORCK, PEGASUS, ILLINOIS, REGULA_FALSI };
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/config/solver/SolverSettingsFromConfig.cpp:root_finders, //humanoid_nmpc/humanoid_mpc_config/rollout_settings_config.proto:root_finder_names)
// clang-format on

}  // namespace ocs2
