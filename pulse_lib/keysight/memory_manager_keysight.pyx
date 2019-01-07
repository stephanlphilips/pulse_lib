import numpy as np
cimport numpy as np


 from cpython cimport array


cdef class Memory_manager():
	"""
	Object that manages the occupation of the memory of the AWG.
	
	The Memory is pre-segmented on the AWG. E.g you have to say that you want to be able to upload 100 sequences with lenth L.
	Currently this is a simple system that just assigns memory locations to certain segments.
	There is are currently no advanced capabilities build in to reuse the memory in the AWG. This could be done later if performance would be limiting.

	A compelling reason not to reuse segments would be in the assumption one would regularly use DSP which would make it hard to reuse things.
	"""
	cdef double[:,:] memory_cells
	cdef segment_occupation_AWG segm_occup

	def __init__(self):
		'''
		Initialize memory management object.
		'''
		
		# make division of memory[ HARDCODED ...]: (assume sorted from big to small)
		self.memory_cells = np.array([
			[1e8,4],
			[5e7,4],
			[1e7,4],
			[5e6,8],
			[1e6,40],
			[1e5,500],
			[1e4,1400],
			# [5e3,20000],
			# [1e3,50000]
		], dtype=double)

		self.segm_occup = segment_occupation_AWG(self.memory_cells)

	cpdef int get_upload_slot(self, int n_points):
		'''
		get a location where the data can be uploaded in the memory.
		Args:
			n_points (int) : number of points that will be saved in the memory
		returns:
			segment_number (int) : the segment number where the segment can be uploaded to in the memory of the AWG.
		'''
		return self.segm_occup.request_new_segment(n_points)
		
	cpdef void release_memory(self, int[:] segments):
		'''
		release memory when segments are not longer needed.
		Args:
			segments (array<int>) : list with segments number to be released
		'''
		for i in segments:
			self.segm_occup.free_segment(i)

cdef class segment_occupation_AWG():
	"""
	Object that manages the occupation of the memory of the AWG.
	
	The Memory is pre-segmented on the AWG. E.g you have to say that you want to be able to upload 100 sequences with lenth L.
	Currently this is a simple system that just assigns memory locations to certain segments.
	There is are currently no advanced capabilities build in to reuse the memory in the AWG. This could be done later if performance would be limiting.

	A compelling reason not to reuse segments would be in the assumption one would regularly use DSP which would make it hard to reuse things.
	"""
	cdef double[:,:] segment_distribution
	cdef int[:] memory_sizes
	cdef int[:] index_info
	cdef array.array seg_data
	def __init__(self, double[:,:] segment_distribution):
		self.segment_distribution = segment_distribution
		self.memory_sizes = np.sort(self.segment_distribution[:,0])
		self.index_info = np.zeros([len(self.memory_sizes)])
		seg_data = array.array([]
		k = 0
		for i in segment_distribution:
			seg_numbers = np.arange(i[1]) + k
			self.index_info[len(seg_data)] = k
			k += i[1]
			seg_data.append((i[0], seg_numbers))

		self.seg_data = dict(seg_data)

	cpdef request_new_segment(self, int size):
		'''
		Request a new segment in the memory with a certain size
		Args:
			size (int) :  size you want to reserve in the AWG memory
		'''
		cdef array.array valid_sizes = ('int', [1, 2, 3])

		for i in range(len(self.memory_sizes)):
			if size < self.memory_sizes[i]:
				valid_sizes.append(i)

		for i in valid_sizes:
			seg_numbers = self.seg_data[self.memory_sizes[i]]
			if len(seg_numbers) > 0 :
				my_segnumber = seg_numbers[0]
				self.seg_data[self.memory_sizes[i]] = seg_numbers[1:]
				return my_segnumber, self.memory_sizes[i]

			# means no segment available
			return -1, 0

	cpdef free_segment(self, int seg_number):
		'''
		adds a segment as free back to the usable segment pool
		Args:
			seg_number (int), number of the segment in the ram of the AWG
		'''

		seg_loc = np.max(np.where(seg_number >= self.index_info)[0])+1
		seg_numbers = self.seg_data[self.memory_sizes[-seg_loc]]
		seg_numbers = np.append(seg_numbers, seg_number)

		self.seg_data[self.memory_sizes[-seg_loc]] = seg_numbers