import numpy as np 
import copy

class loop_obj():
	"""object that initializes some standard fields that need to be there in a loop object"""
	def __init__(self):
		# little inspiration from qcodes parameter ...
		self.names = list()
		self.units = list()
		self.axis = list()
		self.dtype = None
	
	def add_data(self, data, axis = None, names = None, units = None):
		self.data = np.asarray(data)
		self.dtype = self.data.dtype

		if axis is None:
			self.axis = [-1]*len(self.data.shape)
		elif type(axis) == int:
			self.axis = [axis]
		else:
			if len(axis) != len(self.data.shape):
				raise ValueError("Provided incorrect dimensions for the axis.")
			self.axis = axis
		
		if names is None:
			self.names = ["undefined"]*len(self.data)
		elif type(names) == str:
			self.names = [names]
		else:
			if len(names) != len(self.data.shape):
				raise ValueError("Provided incorrect dimensions for the axis.")
			self.names = names

		if units is None:
			self.units = ["a.u"]*len(self.data)
		elif type(units) == str:
			self.units = [units]
		else:
			if len(units) != len(self.data.shape):
				raise ValueError("Provided incorrect dimensions for the axis.")
			self.units = units

	def __len__(self):
		return len(self.data)

	@property
	def shape(self):
		return self.data.shape

	@property
	def ndim(self):
		return self.data.ndim

	def __getitem__(self, key):
		if self.ndim == 1:
			return self.data[key]
		else:
			partial = loop_obj()
			partial.names =self.names[1:] 
			partial.units = self.units[1:]
			partial.axis = self.axis[1:]
			partial.dtype = self.dtype
			partial.data = self.data[key]
			return partial

	def __add__(self, other):
		cpy = copy.copy(self)
		cpy.data += other
		return cpy
	def __mul__(self, other):
		cpy = copy.copy(self)
		cpy.data *= other
		return cpy

	def __sub__(self, other):
		cpy = copy.copy(self)
		cpy.data -= other
		return cpy
	def __truediv__(self, other):
		cpy = copy.copy(self)
		cpy.data += self.data/other
		return cpy

	def __copy__(self):
		cpy = loop_obj()
		cpy.names = copy.copy(self.names)
		cpy.units = copy.copy(self.units)
		cpy.axis = copy.copy(self.axis)
		cpy.dtype = copy.copy(self.dtype)

		if hasattr(self, 'data'):
			cpy.data= copy.copy(self.data)
		return cpy


class linspace(loop_obj):
	"""docstring for linspace"""
	def __init__(self, start, stop, n_steps = 50, name = "undefined", unit = 'a.u.', axis = -1):
		super().__init__()

		self.data = np.linspace(start, stop, n_steps)
		self.names = [name]
		self.units = [unit]
		self.axis = [axis]


