from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

<<<<<<< HEAD
extensions = [
    Extension("uploader", 
    	sources = ["uploader.pyx","mem_ctrl.cpp", "keysight_awg_post_processing_and_upload.cpp"],
        include_dirs=[numpy.get_include(),"/usr/local/include/Keysight/SD1", ],
        libraries=["SD1core", "SD1pxi", "gomp", "boost_python37"],
        library_dirs=["/usr/local/lib/Keysight/SD1/"],
        language='c++',
		extra_compile_args=['-fopenmp'],
        )
=======
import os
>>>>>>> f42340300141c5c8523f9a32feb242b9e51a8dc1

if os.name == 'nt':
	extensions = [
		Extension("uploader", 
			sources = ["uploader.pyx","mem_ctrl.cpp", "keysight_awg_post_processing_and_upload.cpp"],
			include_dirs=[numpy.get_include(),"C://Program Files (x86)//Keysight//SD1//Libraries//include//cpp//",
					"C://Program Files (x86)//Keysight//SD1//Libraries//include//common//"],
			libraries=["SD1core", "SD1pxi"],
			library_dirs=["C://Program Files (x86)//Keysight//SD1//shared//"],
			language='c++',
			extra_compile_args=['/openmp'],
			)

	]
else:
	extensions = [
		Extension("uploader", 
			sources = ["uploader.pyx","mem_ctrl.cpp", "keysight_awg_post_processing_and_upload.cpp"],
			include_dirs=[numpy.get_include(),"/usr/local/include/Keysight/SD1/cpp", "/usr/local/include/Keysight/SD1/common"],
			libraries=["SD1core", "SD1pxi", "gomp"],
			library_dirs=["/usr/local/lib/Keysight/SD1/"],
			language='c++',
			extra_compile_args=['-fopenmp'],
			)

	]
setup(
    name="Keysight_uploader",
    ext_modules=cythonize(extensions),
)
