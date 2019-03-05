#include "keysight_awg_post_processing_and_upload.h"

#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>

cpp_uploader::cpp_uploader(){}
cpp_uploader::~cpp_uploader(){}

void cpp_uploader::add_awg_module(std::string name, std::string module_type, int chassis, int slot){
	SD_Module *my_AWG_module;
	my_AWG_module = new SD_Module(0);


	int error_handle;
	error_handle = my_AWG_module->open(module_type.c_str(), chassis, slot, 1);
	check_error(my_AWG_module, &error_handle);

	AWG_modules[name] = my_AWG_module;
	mem_mgr[name] = new mem_ctrl();
	error_handles[name] = error_handle;
}

void cpp_uploader::add_upload_job(std::map<std::string, std::map<int, waveform_raw_upload_data*>> *upload_data){
	
	// make first time estimation to determine if multithreading useful (overhead ~ 7.5ms) (times in ms)
	double time_no_multi_upload = *upload_data->begin()->second.begin()->second->npt * 4*upload_data->size()/10e3;
	double time_multi_upload = *upload_data->begin()->second.begin()->second->npt*4/10e3 + 7.5;

	#pragma omp parallel for if(time_multi_upload < time_no_multi_upload)
	for (int i = 0; i < upload_data->size(); ++i){
		auto AWG_iterator = upload_data->begin();
		advance(AWG_iterator, i);
		for (auto channel_iterator = AWG_iterator->second.begin(); channel_iterator != AWG_iterator->second.end(); ++channel_iterator){
			rescale_concatenate_and_convert_to_16_bit_number(channel_iterator->second);
			// TODO : add DSP function here.
			load_data_on_awg(AWG_iterator->first, channel_iterator->second);
			free_cache(channel_iterator->second);
		}	
	}
}

void cpp_uploader::rescale_concatenate_and_convert_to_16_bit_number(waveform_raw_upload_data* upload_data){
	/*
	low level function the voltages to 0/1 range and making a conversion to 16 bits.
	All voltages are also concatenated
	*/
	upload_data->upload_data = new short[*upload_data->npt];
	
	double *wvf_ptr;
	static double v_offset = (upload_data->min_max_voltage->second + upload_data->min_max_voltage->first)/2;
	static double v_pp = upload_data->min_max_voltage->second - upload_data->min_max_voltage->first;

	static double offset_factor = v_offset + v_pp*0.5/65535.;
	static double rescaling_factor = 65535./v_pp;

	size_t i = 0;
	for(size_t wvf_id = 0; wvf_id < upload_data->wvf_npt->size(); ++wvf_id ){
		wvf_ptr = upload_data->wvf_data->at(wvf_id);
		for(int idx_view = 0; idx_view < upload_data->wvf_npt->at(wvf_id); ++ idx_view){
			upload_data->upload_data[i] = ( wvf_ptr[idx_view] - offset_factor)*rescaling_factor;
			++i;
		}
	}
}

void cpp_uploader::load_data_on_awg(std::string awg_name, waveform_raw_upload_data* upload_data){
	
	int segment_location = mem_mgr.find(awg_name)->second->get_upload_slot(*upload_data->npt).first;

	if (segment_location == -1)
		throw std::invalid_argument("No segments available on the AWG/segment is too long ..");

	// error_handles[awg_name] = AWG_modules[awg_name]->waveformReLoad(segment_location, *upload_data->npt, upload_data->upload_data, 0, 0);
	// check_error(AWG_modules[awg_name], &error_handles[awg_name]);

	upload_data->data_location_on_AWG.push_back(segment_location);
}


void cpp_uploader::free_cache(waveform_raw_upload_data* upload_data){
	delete[] upload_data->upload_data;
}

void cpp_uploader::release_memory(std::map<std::string, std::map<int, waveform_raw_upload_data*>>* upload_data){
	// release memory in memory manager object
	for (auto AWG_iterator = upload_data->begin(); AWG_iterator != upload_data->end(); ++AWG_iterator){
		for (auto channel_iterator = AWG_iterator->second.begin(); channel_iterator != AWG_iterator->second.end(); ++channel_iterator){
			mem_mgr.find(AWG_iterator->first)->second->release_memory(channel_iterator->second->data_location_on_AWG);
		}
	}
}
void cpp_uploader::check_error(SD_Module *AWG_module, int *error_handle){
	if (error_handle != 0){
		std::cout << (AWG_module->getErrorMessage(*error_handle)) << "\n";
		// throw std::invalid_argument(AWG_module->getErrorMessage(*error_handle));
	}
}