import cython
import numpy as np
cimport numpy as np
import time
# cdef class test:
#     cdef int a
#     def __init__(self):
#         self.a  = 5

#     cdef void increment_a(self) nogil:
#         self.a +=1

# t = test()
# t.increment_a()

# @cython.final
# cdef class test_class:
#     cdef int number_of_bla
#     cdef test t
#     def __cinit__(self, number):
#         self.number_of_bla = number
#         self.t = test()
#     def get_number(self):
#         return self.number_of_bla

#     cdef int get_number_fast(self) nogil:
#         self.t.increment_a()
#         return self.number_of_bla



# cdef class data_container(np.ndarray):
#     def __cinit__(subtype, input_type=None, shape = (1,)):
#         obj = super(data_container, subtype).__new__(subtype, shape, object)
        
#         if input_type is not None:
#             obj[0] = input_type

#         # return obj

# cdef test_class my_test = test_class(1)
# cdef int undef
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:] fast_linspace(double start, double stop, int N):
	cdef double[:] lin = np.empty([N], dtype=np.double)
	cdef double cache = start
	cdef double delta = (stop - start)/(N-1)
	
	cdef int i = 0

	while (i < N):
		lin[i] = cache
		cache += delta
		i += 1

	return lin


t0 = time.time()

cdef int i = 0 
cdef double [:] t
for i in range(100):
	t = np.linspace(0,10000.,500000)

t1 = time.time()

print((t1-t0)/100)

t0 = time.time()

for i in range(100):
	t = fast_linspace(0,10000.,500000)

t1 = time.time()

print("linspace test",(t1-t0)/100)

# np_arr = np.empty([1,5], dtype=test_class)
# cdef test_class [:,:] arr = np_arr
# np_arr[0,1] = test_class(2)
# print(arr.base)
# with nogil:
#     undef  = my_test.get_number_fast()
#     undef  = arr[0,1].get_number_fast()
# print(undef)