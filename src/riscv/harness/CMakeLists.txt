# Cross-build of the TileLang RISC-V kernel harness together with the TVM
# runtime (and the tvm-ffi FFI runtime it embeds).  Configure with a RISC-V
# toolchain file, e.g.:
#
#   cmake -S src/riscv/harness -B build/riscv-harness \
#         -DCMAKE_TOOLCHAIN_FILE=src/riscv/harness/riscv64-linux-gnu.cmake \
#         -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/riscv-harness -j
#
# Results:
#   * `riscv_kernel_harness` -- a riscv64 executable that loads and invokes
#     TileLang-exported kernel libraries (see riscv_kernel_harness.cc for the
#     wire protocol).  Run it natively on RISC-V hardware or under
#     qemu-riscv64 user-mode emulation.
#   * `libspacemit_ime2_sim.a` -- the SpacemiT IME-2.0 MMA reference
#     simulator (see ../ime2/ime2_sim.cc).
#
# The harness links the TVM *runtime* (not only tvm-ffi): kernel libraries
# allocate local buffers through `TVMBackendAllocWorkspace`, and the
# `__TVMBackend*` context symbols that fill the kernels' function pointers are
# registered by the TVM runtime's static initializers.

cmake_minimum_required(VERSION 3.18)
project(tilelang_riscv_harness CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The bundled TVM (runtime + tvm-ffi live under this tree).
set(TVM_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../3rdparty/tvm" CACHE PATH "TVM source directory")

if(NOT EXISTS "${TVM_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "TVM sources not found at ${TVM_DIR}; run `git submodule update --init --recursive` first")
endif()

# Runtime-only configuration: no code generators are needed in the harness.
set(USE_LLVM OFF CACHE STRING "" FORCE)
set(USE_CUDA OFF CACHE STRING "" FORCE)
set(USE_ROCM OFF CACHE STRING "" FORCE)
set(USE_VULKAN OFF CACHE STRING "" FORCE)
set(USE_METAL OFF CACHE STRING "" FORCE)
set(USE_OPENCL OFF CACHE STRING "" FORCE)
set(USE_MICRO OFF CACHE STRING "" FORCE)
set(USE_RPC OFF CACHE BOOL "" FORCE)
set(USE_GRAPH_EXECUTOR OFF CACHE BOOL "" FORCE)
set(USE_PROFILER OFF CACHE BOOL "" FORCE)
set(USE_RELAY_DEBUG OFF CACHE BOOL "" FORCE)
set(USE_GTEST OFF CACHE BOOL "" FORCE)
set(HIDE_PRIVATE_SYMBOLS ON CACHE BOOL "" FORCE)
set(TVM_FFI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
# libbacktrace (autotools external project) does not cross-build cleanly and
# only improves backtrace cosmetics.
set(TVM_FFI_USE_LIBBACKTRACE OFF CACHE BOOL "" FORCE)
# Keep the toolchain's GNU ld: GCC's exception tables for the runtime sources
# trip lld's --gc-sections handling ("relocation refers to a symbol in a
# discarded section").
set(USE_ALTERNATIVE_LINKER OFF CACHE BOOL "" FORCE)

add_subdirectory(${TVM_DIR} tvm EXCLUDE_FROM_ALL)

add_executable(riscv_kernel_harness riscv_kernel_harness.cc)
# --no-as-needed: the harness itself only calls tvm-ffi entry points, but the
# linked TVM *runtime* must stay a dependency so its static initializers
# register the `__TVMBackend*` context symbols that kernel libraries bind
# (workspace allocation for local buffers, parallel launch, ...).
target_link_libraries(riscv_kernel_harness PRIVATE "-Wl,--no-as-needed" tvm_runtime "-Wl,--as-needed" tvm_ffi_header)

# SpacemiT IME-2.0 MMA reference simulator (see ime2/ime2_sim.cc).  Kernels
# using the tilelang.riscv.ime intrinsics link this static library; on real
# hardware the vendor runtime provides the same entry points.
add_library(spacemit_ime2_sim STATIC "${CMAKE_CURRENT_LIST_DIR}/../ime2/ime2_sim.cc")
set_target_properties(spacemit_ime2_sim PROPERTIES POSITION_INDEPENDENT_CODE ON)
