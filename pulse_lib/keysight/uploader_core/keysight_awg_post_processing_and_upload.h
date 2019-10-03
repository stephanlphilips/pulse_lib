#ifdef linux
#include <Keysight/SD1/cpp/SD_Module.h>
#include <Keysight/SD1/cpp/SD_AIO.h>
#endif

#ifdef _WIN32
#include <Libraries/include/cpp/SD_Module.h>
#include <Libraries/include/cpp/SD_AIO.h>
#endif

#include "mem_ctrl.h"

#include <map>
#include <string>
#include <vector>

struct waveform_raw_upload_data{
	std::vector<double*> *wvf_data;
	std::vector<int> *wvf_npt;
	std::pair<double, double> *min_max_voltage;
	std::vector<double*> *DSP_param;
	short *upload_data;
	int *npt;
	std::vector<int> data_location_on_AWG;
};

class cpp_uploader
{
	std::map<std::string, SD_Module*> SD_modules;
	std::map<std::string, SD_AIO*> AWG_modules;
	std::map<std::string, mem_ctrl*> mem_mgr;
	std::map<std::string, int> error_handles;
public:
	cpp_uploader();
	~cpp_uploader();

	void add_awg_module(std::string AWG_name, int chassis, int slot);
	void add_upload_job(std::map<std::string, std::map<int, waveform_raw_upload_data*>> *upload_data);
	void release_memory(std::map<std::string, std::map<int, waveform_raw_upload_data*>>* upload_data);
	void resegment_memory();
private:
	void rescale_concatenate_and_convert_to_16_bit_number(waveform_raw_upload_data* upload_data);
	void load_data_on_awg(std::string awg_name, waveform_raw_upload_data* upload_data);
	void free_cache(waveform_raw_upload_data* upload_data);
	void check_error(SD_Module *AWG_module, int *error_handle);
};
