from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("uploader", 
    	sources = ["uploader.pyx","keysight_awg_post_processing_and_upload.cpp"],
        include_dirs=[numpy.get_include(),"/usr/local/include/Keysight/SD1/cpp", "/usr/local/include/Keysight/SD1/common"],
        libraries=["SD1core", "SD1pxi", "gomp", "boost_python37"],
        library_dirs=["/usr/local/lib/Keysight/SD1/"],
        language='c++',
		extra_compile_args=['-fopenmp'],
        )

]
setup(
    name="Keysight_uploader",
    ext_modules=cythonize(extensions),
)