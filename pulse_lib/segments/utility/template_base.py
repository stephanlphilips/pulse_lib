from abc import ABC, abstractmethod
from pulse_lib.segments.segment_container import segment_container

class pulse_template(ABC):

    @abstractmethod
    def build(self, seg:segment_container, **kwargs):
        pass

    @abstractmethod
    def replace(self, **kwargs):
        pass
