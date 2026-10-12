/******************************************************************************
Copyright (c) 2020, Farbod Farshidian. All rights reserved.

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

#include <ocs2_core/dynamics/SystemDynamicsBaseAD.h>

namespace ocs2 {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
SystemDynamicsBaseAD::SystemDynamicsBaseAD() : SystemDynamicsBase() {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
SystemDynamicsBaseAD::SystemDynamicsBaseAD(const SystemDynamicsBaseAD& rhs)
    : SystemDynamicsBase(rhs),
      flowMapADInterfacePtr_(new CppAdInterface(*rhs.flowMapADInterfacePtr_)),
      jumpMapADInterfacePtr_(new CppAdInterface(*rhs.jumpMapADInterfacePtr_)),
      tapedTimeStateInput_(rhs.tapedTimeStateInput_.size()),
      tapedTimeState_(rhs.tapedTimeState_.size()) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void SystemDynamicsBaseAD::initialize(size_t stateDim, size_t inputDim, const std::string& modelName, const std::string& modelFolder,
                                      bool recompileLibraries, bool verbose) {
  tapedTimeStateInput_.resize(1 + stateDim + inputDim);
  tapedTimeState_.resize(1 + stateDim);

  auto flowMap = [this, stateDim, inputDim](const ad_vector_t& x, ad_vector_t& y) {
    const ad_scalar_t time = x(0);
    const ad_vector_t state = x.segment(1, stateDim);
    const ad_vector_t input = x.tail(inputDim);
    y = this->systemFlowMap(time, state, input);
  };
  flowMapADInterfacePtr_.reset(new CppAdInterface(flowMap, 1 + stateDim + inputDim, modelName + "_flow_map", modelFolder));

  auto jumpMap = [this, stateDim](const ad_vector_t& x, ad_vector_t& y) {
    const ad_scalar_t time = x(0);
    const ad_vector_t state = x.tail(stateDim);
    y = this->systemJumpMap(time, state);
  };
  jumpMapADInterfacePtr_.reset(new CppAdInterface(jumpMap, 1 + stateDim, modelName + "_jump_map", modelFolder));

  if (recompileLibraries) {
    flowMapADInterfacePtr_->createModels(CppAdInterface::ApproximationOrder::First, verbose);
    jumpMapADInterfacePtr_->createModels(CppAdInterface::ApproximationOrder::First, verbose);
  } else {
    flowMapADInterfacePtr_->loadModelsIfAvailable(CppAdInterface::ApproximationOrder::First, verbose);
    jumpMapADInterfacePtr_->loadModelsIfAvailable(CppAdInterface::ApproximationOrder::First, verbose);
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
vector_t SystemDynamicsBaseAD::computeFlowMap(scalar_t t, const vector_t& x, const vector_t& u, const PreComputation& preComputation) {
  tapedTimeStateInput_ << t, x, u;
  return flowMapADInterfacePtr_->getFunctionValue(tapedTimeStateInput_);
}

/*******************q**********************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
vector_t SystemDynamicsBaseAD::computeJumpMap(scalar_t t, const vector_t& x, const PreComputation& preComputation) {
  tapedTimeState_ << t, x;
  return jumpMapADInterfacePtr_->getFunctionValue(tapedTimeState_);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
VectorFunctionLinearApproximation SystemDynamicsBaseAD::linearApproximation(scalar_t t, const vector_t& x, const vector_t& u,
                                                                            const PreComputation& preComputation) {
  tapedTimeStateInput_ << t, x, u;
  const matrix_t flowJacobian = flowMapADInterfacePtr_->getJacobian(tapedTimeStateInput_);

  VectorFunctionLinearApproximation approximation;
  approximation.dfdx = flowJacobian.middleCols(1, x.rows());
  approximation.dfdu = flowJacobian.rightCols(u.rows());
  approximation.f = flowMapADInterfacePtr_->getFunctionValue(tapedTimeStateInput_);
  return approximation;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
VectorFunctionLinearApproximation SystemDynamicsBaseAD::jumpMapLinearApproximation(scalar_t t, const vector_t& x,
                                                                                   const PreComputation& preComputation) {
  tapedTimeState_ << t, x;
  const matrix_t jumpJacobian = jumpMapADInterfacePtr_->getJacobian(tapedTimeState_);

  VectorFunctionLinearApproximation approximation;
  approximation.dfdx = jumpJacobian.rightCols(x.rows());
  approximation.dfdu.setZero(jumpJacobian.rows(), 0);
  approximation.f = jumpMapADInterfacePtr_->getFunctionValue(tapedTimeState_);
  return approximation;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
ad_vector_t SystemDynamicsBaseAD::systemJumpMap(ad_scalar_t time, const ad_vector_t& state) const {
  return state;
}

}  // namespace ocs2
