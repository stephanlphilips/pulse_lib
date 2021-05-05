
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as pt

from qcodes.instrument.base import Instrument

@dataclass
class InstructionBase:
    address: int
    wait_after: float
    jump_address: Optional[int] = None

# NOTE: n_cycles > 1 cannot be combined with threshold
@dataclass
class DigitizerInstruction(InstructionBase):
    t_measure: Optional[float] = None
    multiplier: Optional[float] = None
    n_cycles: int = 1
    threshold: Optional[float] = None
    pxi: Optional[int] = None
    measurement_id: Optional[int] = None


class SequencerChannel:
    def __init__(self, instrument, number):
        self._instrument = instrument
        self._number = number
        self._schedule = []


    def load_schedule(self, schedule:List[DigitizerInstruction]):
        self._schedule = schedule

    def describe(self):
        print(f'seq {self._number} schedule')
        for inst in self._schedule:
            print(inst)


# mock for M3102A
class MockM3102A_QS(Instrument):

    def __init__(self, name, chassis, slot):
        super().__init__(name)

        self.chassis = chassis
        self.slot = slot

        self._sequencers = {}
        for i in range(1,5):
            self._sequencers[i] = SequencerChannel(self, i)

    def slot_number(self):
        return self._slot_number

    def chassis_number(self):
        return self._chassis_numnber

    def get_sequencer(self, number):
        return self._sequencers[number]

    def plot(self):
        # @@@ plot acquisition interval
        pass

    def describe(self):
        for i,seq in self._sequencers.items():
            seq.describe()
