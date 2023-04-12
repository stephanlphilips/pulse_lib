from pulse_lib.segments.utility.looping import loop_obj
from pulse_lib.configuration.iq_channels import FrequencyUndefined

def get_iq_nco_idle_frequency(job, qubit_channel, index):
    '''
    Returns IQ / NCO frequency to use between pulses.
    This frequency is required for coherent driving of the qubit.

    The NCO frequency is derived from resonance frequency and LO frequency.
    The resonance frequency is retrieved from sequence or pulse_lib object.
    The former overrules the latter.

    The NCO frequency will be set to 0.0 if the qubit frequency is set to FrequencyUndefined.
    '''
    try:
        frequency = job.qubit_resonance_frequency[qubit_channel.channel_name]
        if isinstance(frequency, loop_obj):
            frequency = frequency.at(index)
    except:
        frequency = qubit_channel.resonance_frequency
    if frequency is FrequencyUndefined:
        return 0.0
    if frequency is None:
        return None
    return frequency - qubit_channel.iq_channel.LO
