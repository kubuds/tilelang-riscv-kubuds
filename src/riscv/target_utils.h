/*!
 * \file riscv/target_utils.h
 * \brief RISC-V target attribute helpers for the TileLang compiler passes.
 *
 * The helpers inspect the LLVM target attributes TileLang's RISC-V backend
 * attaches to a target (see tilelang/riscv/target.py): the riscv64 triple and
 * the RVV vector-length hint ``+zvl<bits>b`` from the RISC-V Unprivileged ISA
 * specification.
 */

#ifndef TVM_TL_RISCV_TARGET_UTILS_H_
#define TVM_TL_RISCV_TARGET_UTILS_H_

#include <tvm/target/target.h>

namespace tvm {
namespace tl {

/*! \return whether the target generates code for 64-bit RISC-V. */
bool TargetIsRISCV(const Target &target);

/*!
 * \return the architectural vector register length (VLEN) in bits for RVV
 * targets, or 0 when the target does not enable the V extension.  Derived
 * from the ``+zvl<bits>b`` attributes (VLEN is at least 128 whenever V is
 * present).
 */
int RISCVVLenBits(const Target &target);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_RISCV_TARGET_UTILS_H_
