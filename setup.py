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
        "pulse_lib.segments.utility.segments_c_func",
        ["pulse_lib/segments/utility/segments_c_func.pyx"],
        include_dirs=[numpy.get_include()], 
    ),
    Extension(
        "pulse_lib.segments.data_classes.data_pulse_core",
        ["pulse_lib/segments/data_classes/data_pulse_core.pyx"],
        include_dirs=[numpy.get_include()], 
    ),
    ]



setup(name="pulse_lib",
	version="1.1",
	packages = find_packages(),
	ext_modules = cythonize(extensions),
    install_requires=['si_prefix', ],
	)

