/*
Here classes can be found that control the memory occupation on each AWG module.
*/

#include <iostream>
#include <vector>
#include <map>
#include <algorithm>

class segment_occupation
{
	std::map<int, int> *memory_layout;
	std::map<int, std::vector<int>> seg_data;
public:
	std::vector<int> memory_sizes;
	std::vector<int> index_info;
	segment_occupation(std::map<int, int>* mem_layout);

	std::pair<int, int> request_new_segment(int size);
	void free_segment(int seg_number);
};

class mem_ctrl
{
	std::map<int, int> memory_layout;
public:
	segment_occupation* seg_occ;
	
	mem_ctrl();
	~mem_ctrl();

	std::pair<int, int> get_upload_slot(int n_points);
	void release_memory(std::vector<int> segments_numbers);
};