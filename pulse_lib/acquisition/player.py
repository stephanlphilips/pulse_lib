
# TODO:
# depending on configuration:
# - generate upload
# - upload data
# - play: start one or more iterations
# - retrieve data
# - return data of 1 iteration
# - start next generation and or upload in backgound
# - do not upload if index didn't change and no upload in between


class SequencePlayer:
    def __init__(self, sequencer):
        self._sequencer = sequencer

    def get_channel_data(self):
        seq = self._sequencer
        # upload and play index is seq.sweep_index[::-1]
        # TODO @@@ improve: only upload when index changed or SOME OTHER SETTING has changed...
        #          Changing certain settings should invalidate uploaded waveforms. Dependencies should be stored.
        #          Settings: channel delay, LO frequency,
        # if self.prev_index != seq.sweep_index:
        #    seq.release(self.prev_index)
        #    self.prev_index = seq.sweep_index.copy()
        seq.upload()
        seq.play()
        return seq.get_channel_data()

