import numpy as np 

class linspace():
	"""docstring for linspace"""
	def __init__(self, start, stop, n_steps = 50, name = "undefined", unit = 'a.u.', axis = -1):
		self.data = np.linspace(start, stop, n_steps)
		self.name = name
		self.unit = unit
		self.axis = axis
	def __len__(self):
		return len(self.data)

