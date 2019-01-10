from libcpp cimport bool




cdef extern from "SD_WAVE.h":
	cdef cppclass SD_Wave:
		SD_Wave(const char *waveformFile, char *name = 0);
		SD_Wave(int waveformType, int waveformPoints, double *waveformDataA, double *waveformDataB = 0);
		SD_Wave(int waveformType, int waveformPoints, int *waveformDataA, int *waveformDataB = 0);
		SD_Wave(const SD_Wave *waveform);

		short *getPointVector() const;
		int getPoints() const;
		int getStatus() const;
		int getType() const;

cdef extern from "SD_Module.h":
	cdef cppclass SD_Module:
		SD_Module(int)
		int moduleCount()
		int getProductName(int , int , char *)
		int getProductName(int , char *)
		int getSerialNumber(int , int , char *)
		int getSerialNumber(int , char *)
		int getType(int , int )
		int getType(int )
		int getChassis(int )
		int getSlot(int )

		int open(const char *partNumber, const char *serialNumber);
		int open(const char *partNumber, int nChassis, int nSlot);
		int open(const char* productName, const char* serialNumber, int compatibility);
		int open(const char* productName, int chassis, int slot, int compatibility);
		bool isOpened() const;
		int close();

		int runSelfTest();
		int getStatus() const;
		char *getSerialNumber(char *serialNumber) const;
		char *getProductName(char *productName) const;
		double getFirmwareVersion() const;
		double getHardwareVersion() const;
		int getChassis() const;
		int getSlot() const;
		const char *moduleName() const;

		# # //FPGA
		# int FPGAreadPCport(int port, int *buffer, int nDW, int address, SD_AddressingMode::SD_AddressingMode addressMode = SD_AddressingMode::AUTOINCREMENT, SD_AccessMode::SD_AccessMode accessMode = SD_AccessMode::DMA);
		# int FPGAwritePCport(int port, int *buffer, int nDW, int address, SD_AddressingMode::SD_AddressingMode addressMode = SD_AddressingMode::AUTOINCREMENT, SD_AccessMode::SD_AccessMode accessMode = SD_AccessMode::DMA);
		# int FPGAload(const char *fileName);
		# int FPGAreset(SD_ResetMode::SD_ResetMode mode = SD_ResetMode::PULSE);

		# //HVI Variables
		int readRegister(int varNumber, int &errorOut) const;
		int readRegister(const char *varName, int &errorOut) const;
		double readRegister(int varNumber, const char *unit, int &errorOut) const;
		double readRegister(const char *varName, const char *unit, int &errorOut) const;
		int writeRegister(int varNumber, int varValue);
		int writeRegister(const char *varName, int varValue);
		int writeRegister(int varNumber, double value, const char *unit);
		int writeRegister(const char *varName, double value, const char *unit);

		# //PXItrigger
		int PXItriggerWrite(int nPXItrigger, int value);
		int PXItriggerRead(int nPXItrigger) const;

		# //DAQ
		int DAQconfig(int nDAQ, int nDAQpointsPerCycle, int nCycles, int prescaler, int triggerMode);
		int DAQbufferPoolRelease(int nDAQ);
		int DAQcounterRead(int nDAQ) const;
		int DAQtrigger(int nDAQ);
		int DAQstart(int nDAQ);
		int DAQpause(int nDAQ);
		int DAQresume(int nDAQ);
		int DAQflush(int nDAQ);
		int DAQstop(int nDAQ);
		int DAQtriggerMultiple(int DAQmask);
		int DAQstartMultiple(int DAQmask);
		int DAQpauseMultiple(int DAQmask);
		int DAQresumeMultiple(int DAQmask);
		int DAQflushMultiple(int DAQmask);
		int DAQstopMultiple(int DAQmask);

		# //Extenal Trigger
		int translateTriggerPXItoExternalTriggerLine(int trigger) const;
		int translateTriggerIOtoExternalTriggerLine(int trigger) const;
		int WGtriggerExternalConfig(int nAWG, int externalSource, int triggerBehavior, bool sync = true);
		int DAQtriggerExternalConfig(int nDAQ, int externalSource, int triggerBehavior, bool sync = false);

		# //AWG
		int waveformGetAddress(int waveformNumber);
		int waveformGetMemorySize(int waveformNumber);
		int waveformMemoryGetWriteAddress();
		int waveformMemorySetWriteAddress(int writeAddress);

		int waveformReLoad(int waveformType, int waveformPoints, short *waveformDataRaw, int waveformNumber, int paddingMode = 0);
		int waveformReLoad(SD_Wave *waveformObject, int waveformNumber, int paddingMode = 0);

		int waveformLoad(int waveformType, int waveformPoints, short *waveformDataRaw, int waveformNumber, int paddingMode = 0);
		int waveformLoad(SD_Wave *waveformObject, int waveformNumber, int paddingMode = 0);
		int waveformFlush();

		# //HVI management
		int openHVI(const char *fileHVI);
		int compileHVI();
		int compilationErrorMessageHVI(int errorIndex, char *message, int maxSize);
		int loadHVI();

		# //HVI Control
		int startHVI();
		int pauseHVI();
		int resumeHVI();
		int stopHVI();
		int resetHVI();

		# // P2P
		int DAQp2pStop(int nChannel);
		unsigned long long pipeSinkAddr(int nPipeSink) const;
		int DAQp2pConfig(int nChannel, int dataSize, int timeOut, unsigned long long pipeSink) const;

cdef SD_Module* test
test = new SD_Module(1)

cdef char* module_name
print(test.getProductName(0,1, module_name))

