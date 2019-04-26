/*
Here classes can be found that control the memory occupation on each AWG module.
*/

#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include "mem_ctrl.h"
std::vector<int> int_linspace(int start_in, int end_in){
	std::vector<int> int_linspaced;

	for (int i = start_in; i < end_in; ++i){
		int_linspaced.push_back(i);
	}

	return int_linspaced;
}



segment_occupation::segment_occupation(std::map<int, int>* mem_layout){
	/*
	Object that manages the occupation of the memory of the AWG.

	The Memory is pre-segmented on the AWG. E.g you have to say that you want to be able to upload 100 sequences with lenth L.
	Currently this is a simple system that just assigns memory locations to certain segments.
	There is are currently no advanced capabilities build in to reuse the memory in the AWG. This could be done later if performance would be limiting.

	A compelling reason not to reuse segments would be in the assumption one would regularly use DSP which would make it hard to reuse things.
	*/
	memory_layout = mem_layout;

	int mem_number = 0;
	for (auto it = memory_layout->begin(); it != memory_layout->end(); ++it){
		memory_sizes.push_back(it->first);
	}
	
	std::sort(memory_sizes.begin(), memory_sizes.end());

	for (size_t i = 0; i < memory_sizes.size(); i++){
		seg_data[memory_sizes[i]] = int_linspace(mem_number, mem_number + memory_layout->find(memory_sizes[i])->second);
		mem_number +=  memory_layout->find(memory_sizes[i])->second;
		index_info.push_back(mem_number);
	}

}

std::pair<int, int> segment_occupation::request_new_segment(int size){
	/*
	Request a new segment in the memory with a certain size
	Args:
		size :  size you want to reserve in the AWG memory
	returns:
		new_segment_info : segment_number and maxisize
	*/
	std::vector<int> valid_sizes;
	std::pair<int,int> new_segment_info = std::make_pair(-1,0);
	for (size_t i = 0; i < memory_sizes.size(); ++i){
		if (size <= memory_sizes[i]){
			valid_sizes.push_back(memory_sizes[i]);
		}
	}

	std::vector<int> *mem_locations;
	for (size_t i = 0; i < valid_sizes.size(); ++i)
	{
		mem_locations = &seg_data.find(valid_sizes[i])->second;
		if (mem_locations->size() > 0){
			new_segment_info.first = mem_locations->at(0);
			new_segment_info.second = valid_sizes[i];
			mem_locations->erase(mem_locations->begin());
			break;
		}
	}

	return new_segment_info;
}
void segment_occupation::free_segment(int seg_number){
	/*
	adds a segment as free back to the usable segment pool
	Args:
		seg_number number of the segment in the ram of the AWG
	*/
	int seg_loc = 0;

	for (size_t i = 0; i < index_info.size(); ++i)	{
		if (seg_number < index_info[i]){
			seg_loc = i;
			break;
		}
	}
	seg_data.find(memory_sizes[seg_loc])->second.push_back(seg_number);
}

std::vector<int> *segment_occupation::get_memory_sizes(){
	return &memory_sizes;
}
std::map<int, std::vector<int>> *segment_occupation::get_seg_data(){
	return &seg_data;
}


mem_ctrl::mem_ctrl(){
	// Initialize memory management object.
	// memory_layout[1e8] = 4;
	// memory_layout[5e7] = 4;
	// memory_layout[1e7] = 4;
	// memory_layout[5e6] = 8;
	// memory_layout[1e6] = 10;
	memory_layout[1e5] = 100;
	// memory_layout[1e4] = 500;

	seg_occ = new segment_occupation(&memory_layout);
};
	
mem_ctrl::~mem_ctrl(){
	delete seg_occ;
}

std::pair<int, int> mem_ctrl::get_upload_slot(int n_points){
	/*
	get a location where the data can be uploaded in the memory.
	Args:
		n_points (int) : number of points that will be saved in the memory
	returns:
		segment_number, segment_size : the segment number where the segment can be uploaded to in the memory of the AWG.
	*/
	return seg_occ->request_new_segment(n_points);
}
	
void mem_ctrl::release_memory(std::vector<int> segments_numbers){
	/*
		release memory when segments are not longer needed.
	Args:
		segments_numbers (array<int>) : list with segments number to be released
	*/
	for (size_t i = 0; i < segments_numbers.size(); ++i){
		seg_occ->free_segment(segments_numbers[i]);
	}
}

segment_occupation* mem_ctrl::get_segment_occupation(){
	return seg_occ;
}