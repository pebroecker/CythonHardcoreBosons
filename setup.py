from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name="hardcore_bosons", ext_modules=cythonize('hardcore_bosons.pyx', annotate=True), include_dirs=[numpy.get_include()])
