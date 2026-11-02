/*!
 * \file riscv/target_utils.cc
 * \brief RISC-V target attribute helpers for the TileLang compiler passes.
 */

#include "riscv/target_utils.h"

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/string.h>

#include <algorithm>
#include <cctype>
#include <string>

namespace tvm {
namespace tl {

namespace {

using ffi::Array;
using ffi::String;

std::string Lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

} // namespace

bool TargetIsRISCV(const Target &target) {
  auto mtriple = target->GetAttr<String>("mtriple");
  if (!mtriple) {
    return false;
  }
  std::string triple = Lower(mtriple.value());
  return triple.rfind("riscv64", 0) == 0;
}

int RISCVVLenBits(const Target &target) {
  if (!TargetIsRISCV(target)) {
    return 0;
  }
  auto mattr = target->GetAttr<Array<String>>("mattr");
  if (!mattr) {
    return 0;
  }
  bool has_vector = false;
  int vlen = 0;
  for (const String &entry : mattr.value()) {
    std::string name = Lower(entry);
    // Normalize "+zvl256b" / "-zvl256b" / "zvl256b" to "zvl256b".
    if (!name.empty() && (name[0] == '+' || name[0] == '-')) {
      bool disabled = name[0] == '-';
      name = name.substr(1);
      if (disabled) {
        continue;
      }
    }
    if (name == "v" || name == "zve64d" || name == "zve64f") {
      has_vector = true;
    } else if (name.rfind("zvl", 0) == 0 && name.size() > 4 && name.back() == 'b') {
      int bits = 0;
      bool numeric = true;
      for (size_t i = 3; i + 1 < name.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(name[i]))) {
          numeric = false;
          break;
        }
        bits = bits * 10 + (name[i] - '0');
      }
      if (numeric && bits > 0) {
        vlen = std::max(vlen, bits);
      }
    }
  }
  if (!has_vector) {
    return 0;
  }
  // The V extension guarantees VLEN >= 128.
  return std::max(vlen, 128);
}

} // namespace tl
} // namespace tvm
