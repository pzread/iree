// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

namespace mlir::iree_compiler::GlobalOptimization {

namespace {

struct HoistLoopInvariantOpsPass
    : public HoistLoopInvariantOpsBase<HoistLoopInvariantOpsPass> {
  void runOnOperation() override;
};

} // namespace

void HoistLoopInvariantOpsPass::runOnOperation() {
  getOperation()->walk([&](Operation *op) {
    auto loopOp = dyn_cast<LoopLikeOpInterface>(op);
    if (!loopOp) {
      return;
    }
    auto lb = loopOp.getSingleLowerBound();
    auto ub = loopOp.getSingleUpperBound();
    if (!lb || !ub) {
      return;
    }
    std::optional<int64_t> constantLb = getConstantIntValue(*lb);
    std::optional<int64_t> constantUb = getConstantIntValue(*ub);
    if (constantLb && constantUb && (*constantLb < *constantUb)) {
      moveLoopInvariantCode(
          loopOp.getLoopRegions(),
          [&](Value value, Region *) {
            return loopOp.isDefinedOutsideOfLoop(value);
          },
          [&](Operation *op, Region *) {
            return isMemoryEffectFree(op) && isSpeculatable(op);
          },
          [&](Operation *op, Region *) { loopOp.moveOutOfLoop(op); });
    }
  });
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createHoistLoopInvariantOpsPass() {
  return std::make_unique<HoistLoopInvariantOpsPass>();
}

} // namespace mlir::iree_compiler::GlobalOptimization