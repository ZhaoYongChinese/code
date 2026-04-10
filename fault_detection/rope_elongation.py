import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Dict, Optional, Tuple, Any
from .base import BaseFaultDetector

class RopeElongationDetector(BaseFaultDetector):
    def __init__(self, name: str, config: Dict[str, Any], global_config: Dict = None):
        super().__init__(name, config)
        self.trend_window = self.params.get("trend_window", 100)
        self.slope_threshold = self.params.get("slope_threshold", 0.01)
        self.rms_history = {}
    
    def update(self, sensor_name: str, data_packet: Dict[str, Any]) -> Tuple[bool, Optional[Dict]]:
        rms = data_packet["rms_value"]
        history = self.rms_history.get(sensor_name, [])
        history.append(rms)
        if len(history) > self.trend_window:
            history.pop(0)
        self.rms_history[sensor_name] = history
        
        if len(history) < self.trend_window:
            return False, None
        
        X = np.arange(len(history)).reshape(-1, 1)
        y = np.array(history)
        model = LinearRegression().fit(X, y)
        slope = model.coef_[0]
        
        is_fault = slope > self.slope_threshold
        extra_info = {
            "fault_type": self.name,
            "slope": slope,
            "threshold": self.slope_threshold,
            "latest_rms": rms
        }
        return is_fault, extra_info
    
    def reset(self, sensor_name: Optional[str] = None):
        if sensor_name:
            self.rms_history.pop(sensor_name, None)
        else:
            self.rms_history.clear()