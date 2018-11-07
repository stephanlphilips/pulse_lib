from distutils.core import setup
from Cython.Build import cythonize
import os
import numpy
# for some reason I have to include manually to make it work ..
os.environ['CFLAGS'] = '-I' + numpy.get_include()

setup(
	name = 'segments_c_func`',
	ext_modules= cythonize("segments_c_func.pyx", include_path = [numpy.get_include()]),
)
