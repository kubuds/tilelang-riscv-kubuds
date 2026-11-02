/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tl/riscv/target.cc
 * \brief RISC-V target kind registration.
 *
 * Design note (why a separate target kind instead of reusing "c"):
 *   The "c" kind is architecture-neutral: it carries no notion of ISA
 *   extensions, vector length, or vendor matrix units.  Codegen and the
 *   host-side toolchain both need those facts (see src/riscv/codegen and
 *   tilelang/riscv/toolchain.py), and TVM's target attribute system is the
 *   only mechanism that carries them from Python down to C++ without a
 *   side channel.  Registering "riscv" as a kDLCPU kind keeps every
 *   existing CPU pass pipeline applicable while making the ISA explicit.
 *
 * Attribute set:
 *   march  gcc -march string.  "rv64gc" is the RV64GC baseline (no vector);
 *          "rv64gcv" enables the V extension (step 2 of the plan).
 *   mabi   gcc -mabi string.  "lp64d" matches Bianbu/SG2042 userland.
 *   mcpu   reserved for scheduling decisions (e.g. "spacemit-x100").
 *   vlen   vector length in bits when known (256 on K3 X100, 1024 on A100).
 *          0 means "unknown, do not assume".
 *   ime2   whether the SpacemiT IME2 matrix unit is available.  Reserved for
 *          step 3; it must not silently change numerics, so it defaults off.
 */

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>

#include "dlpack/dlpack.h"

namespace tvm {

namespace refl = tvm::ffi::reflection;

TVM_REGISTER_TARGET_KIND("riscv", kDLCPU)
    .add_attr_option<ffi::String>("march", refl::DefaultValue("rv64gc"))
    .add_attr_option<ffi::String>("mabi", refl::DefaultValue("lp64d"))
    .add_attr_option<ffi::String>("mcpu", refl::DefaultValue("generic-rv64"))
    // 0 = unspecified.  Kernels must then be VLEN-agnostic (see step 2).
    .add_attr_option<int64_t>("vlen", refl::DefaultValue(0))
    // SpacemiT IME2 matrix unit.  Off by default: enabling it changes the
    // numeric path, so it must be an explicit opt-in.
    .add_attr_option<bool>("ime2", refl::DefaultValue(false))
    // Optional BLAS to link when T.gemm is lowered to a vendor call ("openblas").
    // Empty means "pure generated code".  Linked, never vendored: OpenBLAS is
    // BSD-3-Clause, compatible with this repository's Apache-2.0 licence.
    .add_attr_option<ffi::String>("blas", refl::DefaultValue(""))
    // Same meaning as for "c": kernels can return an int32 status code.
    .add_attr_option<bool>("supports_kernel_status_return",
                           refl::DefaultValue(true))
    .set_default_keys({"riscv", "cpu"});

} // namespace tvm
