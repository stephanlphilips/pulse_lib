from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

from Cython.Build import cythonize
import numpy

extensions = [
    Extension("rectangle",
        ["rectangle.pyx", "Rectangle.cpp"],
        language='c++',
        # sources = ["test.h",],
		# extra_compile_args=['-fopenmp'],
        )

]
setup(
    name="uploader",
    ext_modules=cythonize(extensions),
    cmdclass = {'build_ext': build_ext}
)