import numpy as np
from typing import Dict, Optional, Tuple, Any
from .base import BaseFaultDetector

class BoltLoosenDetector(BaseFaultDetector):
    def __init__(self, name: str, config: Dict[str, Any], global_config: Dict = None):
        super().__init__(name, config)
        self.freq_bands = self.params.get("freq_bands", [[100, 200], [500, 800]])
        self.window_size = self.params.get("window_size", 20)
        self.trigger_ratio = self.params.get("trigger_ratio", 2.0)
        self.energy_history = {}
    
    def update(self, sensor_name: str, data_packet: Dict[str, Any]) -> Tuple[bool, Optional[Dict]]:
        fft_all = data_packet.get("fft_all")
        if not fft_all:
            return False, None
        
        spectrum = fft_all["fft"]
        freqs = fft_all["index"]
        
        # 计算指定频段总能量
        band_energy = 0.0
        for low, high in self.freq_bands:
            idx_low = np.argmin(np.abs(freqs - low))
            idx_high = np.argmin(np.abs(freqs - high))
            band_energy += np.sum(spectrum[idx_low:idx_high+1] ** 2)
        
        history = self.energy_history.get(sensor_name, [])
        history.append(band_energy)
        if len(history) > self.window_size:
            history.pop(0)
        self.energy_history[sensor_name] = history
        
        if len(history) < self.window_size:
            return False, None
        
        baseline = np.mean(history)
        is_fault = band_energy > baseline * self.trigger_ratio
        
        extra_info = {
            "fault_type": self.name,
            "band_energy": band_energy,
            "baseline": baseline,
            "ratio": band_energy / baseline if baseline else 0
        }
        return is_fault, extra_info
    
    def reset(self, sensor_name: Optional[str] = None):
        if sensor_name:
            self.energy_history.pop(sensor_name, None)
        else:
            self.energy_history.clear()