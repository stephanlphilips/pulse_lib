import cython
import numpy as np
cimport numpy as np

@cython.final
cdef class test_class:
    cdef int number_of_bla

    def __cinit__(self, number):
        self.number_of_bla = number

    def get_number(self):
        return self.number_of_bla

    cdef int get_number_fast(self) nogil:
        return self.number_of_bla



cdef class data_container(np.ndarray):
    def __cinit__(subtype, input_type=None, shape = (1,)):
        obj = super(data_container, subtype).__new__(subtype, shape, object)
        
        if input_type is not None:
            obj[0] = input_type

        # return obj

cdef test_class my_test = test_class(1)
cdef int undef

np_arr = np.empty([1,5], dtype=test_class)
cdef test_class [:,:] arr = np_arr
np_arr[0,1] = test_class(2)
print(arr.base)
with nogil:
    undef  = my_test.get_number_fast()
    undef  = arr[0,1].get_number_fast()
print(undef)