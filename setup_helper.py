from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import COMMON_NVCC_FLAGS
from torch.utils.cpp_extension import _is_cuda_file, _join_cuda_home, _get_cuda_arch_flags
import copy

from setuptools.command.build_ext import build_ext


class BuildRDCExtension(BuildExtension):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # copy from BuildExtension with added separate compilation
    def build_extensions(self):
        self._check_abi()
        for extension in self.extensions:
            self._add_compile_flag(extension, '-DTORCH_API_INCLUDE_EXTENSION_H')
            self._define_torch_extension_name(extension)
            self._add_gnu_cpp_abi_flag(extension)

        # Register .cu and .cuh as valid source extensions.
        self.compiler.src_extensions += ['.cu', '.cuh']
        # Save the original _compile method for later.
        if self.compiler.compiler_type == 'msvc':
            self.compiler._cpp_extensions += ['.cu', '.cuh']
            original_compile = self.compiler.compile
        else:
            original_compile = self.compiler._compile

        def unix_wrap_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            # Copy before we make any modifications.
            cflags = copy.deepcopy(extra_postargs)
            cc_args_copy = copy.deepcopy(cc_args)
            try:
                original_compiler = self.compiler.compiler_so
                if _is_cuda_file(src):
                    nvcc = _join_cuda_home('bin', 'nvcc')
                    if not isinstance(nvcc, list):
                        nvcc = [nvcc]
                    self.compiler.set_executable('compiler_so', nvcc)
                    if isinstance(cflags, dict):
                        cflags = cflags['nvcc']

                    cflags = COMMON_NVCC_FLAGS + ['--compiler-options', "'-fPIC'"] + cflags + _get_cuda_arch_flags(cflags)
                else:
                    if isinstance(cflags, dict):
                        cflags = cflags['cxx']

                # NVCC does not allow multiple -std to be passed, so we avoid
                # overriding the option if the user explicitly passed it.
                if not any(flag.startswith('-std=') for flag in cflags):
                    cflags.append('-std=c++11')

                # NVCC: Separate compilation
                if '-rdc=true' in cflags:
                    obj_path_list = obj.split('/')
                    obj_link = '/'.join(obj_path_list[:-1] + [('link_' + obj_path_list[-1])])
                    if '-lcudadevrt' in cflags:
                        cflags.remove('-lcudadevrt')
                    original_compile(obj, src, ext, cc_args_copy, cflags, pp_opts)          # compile
                    cflags.append('-lcudadevrt')
                    cc_args_copy.remove('-c')
                    cc_args_copy.append('-dlink')
                    original_compile(obj_link, obj, '.o', cc_args_copy, cflags, pp_opts)    # link
                else:
                    original_compile(obj, src, ext, cc_args_copy, cflags, pp_opts)
            finally:
                # Put the original compiler back in place.
                self.compiler.set_executable('compiler_so', original_compiler)

        self.compiler._compile = unix_wrap_compile
        build_ext.build_extensions(self)
