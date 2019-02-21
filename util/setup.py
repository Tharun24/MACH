from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(ext_modules = cythonize(Extension(
           "util",                                # the extension name
           sources=["util.pyx", "Gather.cpp"], # the Cython source and additional C++ source files
           language="c++",                        # generate and compile C++ code
           include_dirs=[numpy.get_include()],
           extra_compile_args=["-std=c++11", "-fopenmp", "-fopenmp-simd"],
           extra_link_args=["-fopenmp", "-fopenmp-simd"]
      )))
