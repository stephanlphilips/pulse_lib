from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("uploader", ["uploader.pyx"],
        include_dirs=[numpy.get_include(),"/usr/local/include/Keysight/SD1/cpp", "/usr/local/include/Keysight/SD1/common"],
        libraries=["SD1core", "SD1pxi"],
        library_dirs=["/usr/local/lib/Keysight/SD1/"],
        language='c++'
        )

]
setup(
    name="Keysight_uploader",
    ext_modules=cythonize(extensions),
)