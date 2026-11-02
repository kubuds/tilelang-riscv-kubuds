# Cross-build of the TileLang RISC-V kernel harness together with the TVM FFI
# runtime.  Configure with a RISC-V toolchain file, e.g.:
#
#   cmake -S src/riscv/harness -B build/riscv-harness \
#         -DCMAKE_TOOLCHAIN_FILE=src/riscv/harness/riscv64-linux-gnu.cmake \
#         -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/riscv-harness -j
#
# The result is `riscv_kernel_harness`, a riscv64 executable that loads and
# invokes TileLang-exported kernel libraries (see riscv_kernel_harness.cc for
# the wire protocol).  Run it natively on RISC-V hardware or under
# qemu-riscv64 user-mode emulation.

cmake_minimum_required(VERSION 3.18)
project(tilelang_riscv_harness CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Where the tvm-ffi sources live (shipped as part of the bundled TVM).
set(TVM_FFI_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../3rdparty/tvm/3rdparty/tvm-ffi" CACHE PATH "tvm-ffi source directory")

if(NOT EXISTS "${TVM_FFI_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "tvm-ffi sources not found at ${TVM_FFI_DIR}; run `git submodule update --init --recursive` first")
endif()

# Build the FFI runtime as part of this project so the toolchain applies to it.
set(TVM_FFI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${TVM_FFI_DIR} tvm_ffi EXCLUDE_FROM_ALL)

add_executable(riscv_kernel_harness riscv_kernel_harness.cc)
target_include_directories(riscv_kernel_harness PRIVATE "${TVM_FFI_DIR}/include" "${TVM_FFI_DIR}/3rdparty/dlpack/include")
target_link_libraries(riscv_kernel_harness PRIVATE tvm_ffi_shared)
