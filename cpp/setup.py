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
    PYBIND_INC = pybind11.get_include()
except ImportError:
    raise RuntimeError(
        "pybind11 is required to build _qs_fast. "
        "Run: pip install pybind11"
    )

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
else:
    # Linux / macOS (GCC or Clang)
    extra_compile_args = ["-O3", "-std=c++17", "-march=native",
                          "-ffast-math", "-Wall", "-fvisibility=hidden"]

ext = Extension(
    name="_qs_fast",
    sources=["qs_fast.cpp"],
    include_dirs=[PYBIND_INC],
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
