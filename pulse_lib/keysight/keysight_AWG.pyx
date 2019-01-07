from pulse_lib.keysight.AWG_memory_manger import Memory_manager

cdef class keysight_AWG():
	"""Object that takes care of cumiunicating with the AWG"""
	def __init__(self):
		