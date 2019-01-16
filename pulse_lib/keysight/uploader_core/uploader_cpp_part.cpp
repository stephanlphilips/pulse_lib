#include <Keysight/SD1/cpp/SD_Module.h>

#include <map>
#include <string>
#include <vector>
#include <stdexcept>

struct waveform_raw_upload_data{
	std::vector<double*> *wvf_data;
	std::vector<int> *wvf_npt;
	std::pair<double, double> *min_max_voltage;
	std::vector<double*> *DSP_param;
	short *upload_data;
	int npt;
};

class cpp_uploader
{
	std::map<std::string, SD_Module*> AWG_modules;
	std::map<std::string, int> error_handles;

public:
	cpp_uploader();
	~cpp_uploader();

	void add_awg_module(std::string name, std::string module_type, int chassis, int slot){
		SD_Module *my_AWG_module;
		my_AWG_module = new SD_Module(0);

		int error_handle;
		error_handle = my_AWG_module->open(module_type.c_str(), chassis, slot, 1);
		check_error(my_AWG_module, &error_handle);

		AWG_modules[name] = my_AWG_module;
		error_handles[name] = error_handle;
	}

	void add_upload_job(std::map<std::string, std::map<int, waveform_raw_upload_data>> *upload_data){
		// std::map<std::string, std::map<int, waveform_raw_upload_data>>::iterator AWG_iterator;
		// std::map<int, waveform_raw_upload_data>::iterator channel_iterator;

		// #pragma omp parallel for private(channel_iterator)
		// for (AWG_iterator = upload_data->begin(); AWG_iterator != upload_data->end(); ++AWG_iterator){
			
		// 	for (channel_iterator = AWG_iterator->second.begin(); channel_iterator != AWG_iterator->second.end(); ++channel_iterator){
		// 		rescale_concatenate_and_convert_to_16_bit_number(&channel_iterator->second);
		// 		// TODO : add DSP function here.
		// 		load_data_on_awg(AWG_iterator->first, &channel_iterator->second);
		// 		free_memory(&channel_iterator->second);
		// 	}	
		// }
		#pragma omp parallel for
		for (int i = 0; i < upload_data->size(); ++i){
			// std::map<int, waveform_raw_upload_data>::iterator channel_iterator;
			auto AWG_iterator = upload_data->begin();
			advance(AWG_iterator, i);
			for (auto channel_iterator = AWG_iterator->second.begin(); channel_iterator != AWG_iterator->second.end(); ++channel_iterator){
				rescale_concatenate_and_convert_to_16_bit_number(&channel_iterator->second);
				// TODO : add DSP function here.
				load_data_on_awg(AWG_iterator->first, &channel_iterator->second);
				free_memory(&channel_iterator->second);
			}	
		}
	}

private:
	void rescale_concatenate_and_convert_to_16_bit_number(waveform_raw_upload_data* upload_data){
		/*
		low level function the voltages to 0/1 range and making a conversion to 16 bits.
		All voltages are also concatenated
		*/
		upload_data->upload_data = new short[upload_data->npt];

		double *wvf_ptr;
		static double v_offset = (upload_data->min_max_voltage->second + upload_data->min_max_voltage->first)/2;
		static double v_pp = upload_data->min_max_voltage->second - upload_data->min_max_voltage->first;

		static double offset_factor = v_offset + v_pp*0.5/65535.;
		static double rescaling_factor = 65535./v_pp;

		size_t i = 0;
		for(size_t wvf_id = 0; wvf_id > upload_data->wvf_npt->size(); ++wvf_id ){
			wvf_ptr = upload_data->wvf_data->at(wvf_id);
			for(size_t idx_view = 0; idx_view < upload_data->wvf_npt->at(wvf_id); ++ idx_view){
				upload_data->upload_data[i] = ( wvf_ptr[idx_view] - offset_factor)*rescaling_factor;
				++i;
			}
		}
	}

	void load_data_on_awg(std::string awg_name, waveform_raw_upload_data* upload_data){
		// todo add waveform_number
		error_handles[awg_name] = AWG_modules[awg_name]->waveformReLoad(6, upload_data->npt, upload_data->upload_data, 0, 0);
		check_error(AWG_modules[awg_name], &error_handles[awg_name]);
	}

	void free_memory(waveform_raw_upload_data* upload_data){
		delete[] upload_data->upload_data;
	}

	void check_error(SD_Module *AWG_module, int *error_handle){
		if (error_handle != 0){
			throw std::invalid_argument(AWG_module->getErrorMessage(*error_handle));
		}
	}
};

int main(int argc, char const *argv[])
{
	// cpp_uploader up = cpp_uploader();
	return 0;
}