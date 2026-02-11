// Copyright © 2023-2024 Apple Inc.

#pragma once

#include "mlx/backend/metal/device.h"

namespace mlx::core {

// Dispatch a quantized matmul (qmm/qmv) on the GPU.
// Extracted from QuantizedMatmul::eval_gpu() so that other primitives
// (e.g. CCELoss) can perform quantized matmul without going through
// the full QuantizedMatmul op.
void dispatch_quantized_matmul(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s);

} // namespace mlx::core
