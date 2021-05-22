from dataclasses import dataclass
from typing import Optional

@dataclass
class awg_slave:
    awg_name: str
    marker_name: str
    sync_latency: Optional[float] = None