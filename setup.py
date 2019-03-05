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
    )]

if os.name == 'nt':
    extensions += [Extension("pulse_lib.keysight.uploader_core.uploader", 
            sources = ["pulse_lib/keysight/uploader_core/uploader.pyx",
                    "pulse_lib/keysight/uploader_core/mem_ctrl.cpp", 
                    "pulse_lib/keysight/uploader_core/keysight_awg_post_processing_and_upload.cpp"],
            include_dirs=[numpy.get_include(),"C://Program Files (x86)//Keysight//SD1"],
            libraries =["SD1core", "SD1pxi"],
            library_dirs =["C://Program Files (x86)//Keysight//SD1//shared//"],
            language='c++',
            extra_compile_args=['/openmp'],
            ) ]
else:
    extensions += [Extension("pulse_lib.keysight.uploader_core.uploader", 
            sources = ["pulse_lib/keysight/uploader_core/uploader.pyx",
                    "pulse_lib/keysight/uploader_core/mem_ctrl.cpp", 
                    "pulse_lib/keysight/uploader_core/keysight_awg_post_processing_and_upload.cpp"],
            include_dirs=[numpy.get_include(),"/usr/local/include/Keysight/SD1/cpp", "/usr/local/include/Keysight/SD1/common"],
            libraries=["SD1core", "SD1pxi", "gomp"],
            library_dirs=["/usr/local/lib/Keysight/SD1/"],
            language='c++',
            extra_compile_args=['-fopenmp'],
            ) ]



setup(name="pulse_lib",
	version="1.1",
	# package_dir={'pulse_lib':'pulse_lib'},
	packages = find_packages(),
	ext_modules = cythonize(extensions)
	)

