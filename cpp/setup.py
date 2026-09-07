"""
setup.py — Build the _qs_fast C++ extension for QuantumSentinel.

Usage
-----
# From the project root (MinGW/GCC on PATH):
    pip install -e cpp/

# Or explicitly:
    cd cpp
    python setup.py build_ext --inplace

MinGW configuration
-------------------
On Windows with MinGW (WinLibs POSIX/UCRT), set compiler before building:
    python setup.py build_ext --inplace --compiler=mingw32

MSVC (Visual Studio Build Tools 2019/2022)
-----------------------------------------
python setup.py build_ext --inplace   # MSVC auto-detected
"""

from setuptools import setup, Extension
import sys
import os

try:
    import pybind11
    # get_include(user=True) is a no-op since pybind11 2.6 and the `user`
    # parameter was marked for removal in 3.x.  Call once and deduplicate.
    _pb_inc = pybind11.get_include()
    PYBIND_INCS = list(dict.fromkeys([_pb_inc]))   # ordered, deduplicated
except ImportError:
    raise RuntimeError(
        "pybind11 is required to build _qs_fast. "
        "Run: pip install pybind11"
    )

_HERE = os.path.dirname(os.path.abspath(__file__))
sources = [os.path.join(_HERE, "qs_fast.cpp")]

# C++ standard: C++17 for std::optional etc.
extra_compile_args = []
extra_link_args = []

if sys.platform == "win32":
    if "mingw" in os.environ.get("CC", "").lower() or "--compiler=mingw32" in sys.argv:
        # MinGW / WinLibs
        extra_compile_args = ["-O3", "-std=c++17", "-march=native",
                              "-ffast-math", "-Wall"]
        extra_link_args = ["-static-libgcc", "-static-libstdc++"]
    else:
        # MSVC
        extra_compile_args = ["/O2", "/std:c++17", "/W3", "/EHsc"]
elif sys.platform == "darwin":
    # macOS (Clang / Apple Silicon) — Apple Clang does not support -march=native on ARM64
    extra_compile_args = ["-O3", "-std=c++17", "-ffast-math", "-Wall", "-fvisibility=hidden"]
else:
    # Linux (GCC or Clang)
    extra_compile_args = ["-O3", "-std=c++17", "-march=native",
                          "-ffast-math", "-Wall", "-fvisibility=hidden"]

ext = Extension(
    name="_qs_fast",
    sources=sources,
    include_dirs=PYBIND_INCS,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++",
)

setup(
    name="qs-fast",
    version="4.0.0",
    description="QuantumSentinel high-performance C++ kernels",
    long_description=__doc__,
    ext_modules=[ext],
    python_requires=">=3.10",
    install_requires=["pybind11>=2.12", "numpy>=1.26"],
)
