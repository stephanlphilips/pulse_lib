from distutils.core import setup
from Cython.Build import cythonize
import os
import numpy
# for some reason I have to include manually to make it work ..
os.environ['CFLAGS'] = '-I' + numpy.get_include()

setup(
	name = 'test`',
	ext_modules= cythonize("test.pyx", include_path = [numpy.get_include()]),
)
