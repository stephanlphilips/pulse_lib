import numpy as np
import pulse_lib.segments.segments_c_func as seg_func

class test(object):
	"""docstring for test"""
	def __init__(self, testing):
		super(test, self).__init__()
		self.testing = testing

	def __getitem__(self, *key):
		return self.testing[key[0]]


def test_function(a=0, b):
	print(a,b)

test_function(5)

t = test(np.zeros([5,5]))
print(t[1,2])
		