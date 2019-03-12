from distutils.core import setup
from Cython.Build import cythonize
import os
import numpy
# for some reason I have to include manually to make it work ..

setup(
	name = 'data_classes_markers`',
	ext_modules= cythonize("data_classes_markers.pyx", include_path = [numpy.get_include()]),

)
