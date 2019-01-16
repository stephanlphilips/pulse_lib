#include <Keysight/SD1/cpp/SD_Module.h>

#include <map>
#include <string>
#include <vector>

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

	void add_awg_module(std::string name, std::string module_type, int chassis, int slot);
	void add_upload_job(std::map<std::string, std::map<int, waveform_raw_upload_data>> *upload_data);
private:
	void rescale_concatenate_and_convert_to_16_bit_number(waveform_raw_upload_data* upload_data);
	void load_data_on_awg(std::string awg_name, waveform_raw_upload_data* upload_data);
	void free_memory(waveform_raw_upload_data* upload_data);
	void check_error(SD_Module *AWG_module, int *error_handle);
};