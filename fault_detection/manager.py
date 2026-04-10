from typing import Dict, List, Tuple, Any, Optional
from loguru import logger as mylog
from .base import BaseFaultDetector
from .bearing_fault import BearingFaultDetector
from .bolt_loosen import BoltLoosenDetector
from .rope_elongation import RopeElongationDetector

class FaultManager:
    def __init__(self, global_config: Dict):
        self.global_config = global_config
        self.detectors: Dict[str, BaseFaultDetector] = {}      # name -> detector instance
        self.sensor_map: Dict[str, List[str]] = {}            # sensor_name -> list of detector names
        self._init_detectors()
    
    def _init_detectors(self):
        det_cfg = self.global_config.get("fault_detectors", {})
        for det_name, det_conf in det_cfg.items():
            det_type = det_conf.get("type")
            sensors = det_conf.get("sensors", [])
            
            # 根据类型实例化检测器
            if det_type == "bearing":
                detector = BearingFaultDetector(det_name, det_conf, self.global_config)
            elif det_type == "bolt_loosen":
                detector = BoltLoosenDetector(det_name, det_conf)
            elif det_type == "rope_elongation":
                detector = RopeElongationDetector(det_name, det_conf)
            else:
                mylog.error(f"Unknown detector type: {det_type}")
                continue
            
            self.detectors[det_name] = detector
            
            # 建立传感器到检测器的映射
            for sensor in sensors:
                if sensor not in self.sensor_map:
                    self.sensor_map[sensor] = []
                self.sensor_map[sensor].append(det_name)
            
            mylog.info(f"Initialized detector '{det_name}' for sensors: {sensors}")
    
    def process(self, sensor_name: str, data_packet: Dict) -> List[Tuple[str, bool, Dict]]:
        """
        处理传感器数据，返回所有触发的故障
        返回格式: [(detector_name, is_fault, extra_info), ...]
        """
        results = []
        detector_names = self.sensor_map.get(sensor_name, [])
        for det_name in detector_names:
            detector = self.detectors.get(det_name)
            if detector is None:
                continue
            try:
                is_fault, extra = detector.update(sensor_name, data_packet)
                if is_fault:
                    results.append((det_name, is_fault, extra))
            except Exception as e:
                mylog.error(f"Error in detector '{det_name}' for sensor '{sensor_name}': {e}")
        return results
    
    def reset_detector(self, detector_name: str, sensor_name: Optional[str] = None):
        if detector_name in self.detectors:
            self.detectors[detector_name].reset(sensor_name)
    
    def reset_all(self):
        for detector in self.detectors.values():
            detector.reset()