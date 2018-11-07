import numpy as np
import copy

class test(object):
	"""docstring for test"""
	def __init__(self, arg):
		super(test, self).__init__()
		self.arg = arg
		

data = np.empty([1], dtype=object)
data[0] = test(56)

print(data[0].arg)