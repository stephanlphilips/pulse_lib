import numpy as np 


class loop_obj():
	"""objecet that initializes some standard fields that need to be there in a loop object"""
	def __init__(self):
		# little inspiration from qcodes parameter ...
		self.names = list()
		self.units = list()
		self.axis = list()
		self.dtype = None
	
	def add_data(self, data, axis = None, names = None, units = None):
		self.data = data
		self.dtype = data.dtype
		if axis is None:
			self.axis = [-1]*len(data)
		else:
			if len(axis) != len(data.shape):
				raise ValueError("Provided incorrect dimensions for the axis.")
			self.axis = axis
		
		if names is None:
			self.names = ["undefined"]*len(data)
		else:
			if len(names) != len(data.shape):
				raise ValueError("Provided incorrect dimensions for the axis.")
			self.names = names

		if units is None:
			self.units = ["a.u"]*len(data)
		else:
			if len(units) != len(data.shape):
				raise ValueError("Provided incorrect dimensions for the axis.")
			self.units = units

	def __len__(self):
		return len(self.data)

	@property
	def shape(self):
		return self.data.shape

	def __getitem__(self, key):
		if len(self.axis) == 1:
			return self.data[key]
		else:
			partial = loop_obj()
			partial.names =self.names[1:] 
			partial.units = self.units[1:]
			partial.axis = self.axis[1:]
			partial.dtype = self.dtype
			partial.data = self.data[key]
			return partial

	
class linspace(loop_obj):
	"""docstring for linspace"""
	def __init__(self, start, stop, n_steps = 50, name = "undefined", unit = 'a.u.', axis = -1):
		super().__init__()

		self.data = np.linspace(start, stop, n_steps)
		self.names = [name]
		self.units = [unit]
		self.axis = [axis]




