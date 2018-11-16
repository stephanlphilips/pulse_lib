from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize


import os
import numpy
# for some reason I have to include manually to make it work ..
os.environ['CFLAGS'] = '-I' + numpy.get_include()

packages = find_packages()
print('packages: %s' % packages)

extensions = [
    Extension(
        "pulse_lib.segments.segments_c_func",
        ["pulse_lib/segments/segments_c_func.pyx"],
        include_dirs=[numpy.get_include()], 
        library_dirs=['/some/path/to/include/'],
    ),
]



setup(name="pulse_lib",
	version="1.1",
	# package_dir={'pulse_lib':'pulse_lib'},
	packages = find_packages(),
	ext_modules = cythonize(extensions)
	)

