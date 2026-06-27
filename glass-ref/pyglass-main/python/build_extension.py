import os
import sys
import numpy as np
import pybind11
import setuptools
from setuptools import Extension
from setuptools.command.build_ext import build_ext


def get_include_dirs():
    include_dirs = [
        pybind11.get_include(),
        np.get_include(),
    ]

    # compatibility when run in python_bindings
    bindings_dir = "python"
    if bindings_dir in os.path.basename(os.getcwd()):
        include_dirs.extend(["../", "../third_party/helpa"])
    else:
        include_dirs.extend(["./", "./third_party/helpa"])

    return include_dirs


def get_source_files():
    bindings_dir = "python"
    if bindings_dir in os.path.basename(os.getcwd()):
        return ["./bindings.cc"]
    else:
        return ["./python/bindings.cc"]


def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile

    f = tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False)
    f.write("int main (int argc, char **argv) { return 0; }")
    f.close()
    try:
        compiler.compile([f.name], extra_postargs=[flagname])
    except setuptools.distutils.errors.CompileError:
        return False
    finally:
        os.unlink(f.name)
    return True


def cpp_flag(compiler):
    if compiler.compiler_type == "msvc":
        return "/std:c++20"
    if has_flag(compiler, "-std=c++20"):
        return "-std=c++20"
    elif has_flag(compiler, "/std:c++20"):
        return "/std:c++20"
    else:
        raise RuntimeError(
            "Unsupported compiler -- at least C++20 support " "is needed!"
        )


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    c_opts = {
        "unix": "-Ofast -lrt -march=native -fpic -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0".split(),
        "msvc": "/O2 /arch:AVX2 /openmp /EHsc /D__SSE2__ /D__AVX__ /D__AVX2__ /D__F16C__ /D_USE_MATH_DEFINES /D_CRT_SECURE_NO_WARNINGS /wd4267 /wd4244 /wd4996 /wd4819 /bigobj".split(),
    }

    link_opts = {
        "unix": [],
        "msvc": [],
    }

    c_opts["unix"].append("-fopenmp")
    link_opts["unix"].extend(["-fopenmp", "-pthread"])

    def build_extensions(self):
        plat = "unix" if self.compiler.compiler_type != "msvc" else "msvc"
        opts = self.c_opts.get(plat, [])
        opts.append(f'-DVERSION_INFO="{self.distribution.get_version()}"')
        opts.append(cpp_flag(self.compiler))
        if plat == "unix" and has_flag(self.compiler, "-fvisibility=hidden"):
            opts.append("-fvisibility=hidden")

        for ext in self.extensions:
            ext.extra_compile_args.extend(opts)
            ext.extra_link_args.extend(self.link_opts.get(plat, []))

        build_ext.build_extensions(self)


def create_extension(version_info):
    libraries = []
    extra_objects = []

    ext_modules = [
        Extension(
            "glass",
            get_source_files(),
            include_dirs=get_include_dirs(),
            libraries=libraries,
            language="c++",
            extra_objects=extra_objects,
        ),
    ]

    return ext_modules


def get_build_ext_cmdclass():
    return {"build_ext": BuildExt}
