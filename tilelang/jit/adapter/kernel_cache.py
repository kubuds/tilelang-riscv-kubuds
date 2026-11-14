import os

from tilelang.cache.kernel_cache import KernelCache
from tilelang.jit import JITKernel


class TVMFFIKernelCache(KernelCache):
    kernel_lib_path = "executable.so"

    @staticmethod
    def _get_export_kwargs(kernel: JITKernel) -> dict:
        artifact = getattr(kernel, "artifact", None)
        target_host = getattr(artifact, "target_host", None)
        target_host_kind = getattr(getattr(target_host, "kind", None), "name", None)
        kwargs = (
            KernelCache._get_source_compile_args()
            if target_host_kind == "c"
            else KernelCache._get_export_link_args()
        )
        # RISC-V: the host-side link needs the ISA flags (-march/-mabi/-mrvv-*)
        # and the tl_templates include path, otherwise the cached executable is
        # built with generic options and fails to find tl_templates/cpp/.
        target = getattr(kernel, "target", None)
        if target is not None and getattr(target.kind, "name", None) == "riscv":
            from tilelang.riscv.toolchain import get_compile_flags, get_export_kwargs

            kwargs = dict(kwargs)
            flags = get_compile_flags(target)
            if flags:
                kwargs["options"] = list(kwargs.get("options", [])) + flags
            kwargs.update(get_export_kwargs(target))
        return kwargs

    def _save_wrapper_kernel_code_to_disk(self, kernel: JITKernel, cache_path: str, verbose: bool = False):
        host_kernel_path = os.path.join(cache_path, self.host_kernel_path)
        if verbose:
            self.logger.debug(f"Saving wrapped kernel source code to file: {host_kernel_path}")
        KernelCache._safe_write_file(host_kernel_path, "w", lambda file: file.write(kernel.adapter.get_host_source()))

    def _save_so_cubin_to_disk(self, kernel: JITKernel, cache_path: str, verbose: bool = False):
        kernel_lib_path = os.path.join(cache_path, self.kernel_lib_path)
        executable = kernel.adapter.get_exportable_executable()
        if verbose:
            self.logger.debug(f"Saving kernel executable to file: {executable}")
        KernelCache._safe_write_executable(executable, kernel_lib_path, export_kwargs=self._get_export_kwargs(kernel))
