#!/usr/bin/env python3
"""
钢丝绳健康基线标定脚本
运行方式：python calibrate_rope_baseline.py
功能：采集多帧稳态数据，计算并保存基线到 rope_baseline.json
"""
import sys
import os
import json
import time
import numpy as np
import zmq
from scipy.signal import find_peaks

# 添加项目路径以便导入故障检测器中的辅助函数（直接复制关键函数也可）
sys.path.append(os.path.join(sys.path[0], 'fault_detection'))
# 简单起见，直接定义所需的函数
def find_natural_freqs(spectrum, freqs, low=5, high=200):
    mask = (freqs >= low) & (freqs <= high)
    sub_spec = spectrum[mask]
    sub_freq = freqs[mask]
    if len(sub_spec) == 0:
        return []
    peaks, _ = find_peaks(sub_spec, height=0.2 * np.max(sub_spec), distance=5)
    if len(peaks) == 0:
        return []
    idx_sorted = sorted(peaks, key=lambda x: sub_spec[x], reverse=True)[:3]
    f_list = sub_freq[idx_sorted].tolist()
    f_list.sort()
    return f_list

def get_fp_amplitude(spectrum, freqs, rope_speed, lay_length=0.12):
    fp = rope_speed / lay_length
    idx = np.argmin(np.abs(freqs - fp))
    return float(spectrum[idx])

def main():
    # 连接到 ZMQ 数据源
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect("tcp://localhost:33333")  # 根据实际修改

    # 需要标定的传感器列表
    sensors_to_calibrate = ["WXT02_p004_9003", "WXT03_p005_9004"]  # 根据实际填写

    baseline = {}
    frames_needed = 10  # 采集帧数

    print("开始基线标定，请确保钢丝绳处于健康状态且电梯匀速运行...")
    collected = {s: [] for s in sensors_to_calibrate}

    while True:
        try:
            data = socket.recv_pyobj(flags=zmq.NOBLOCK)
        except zmq.Again:
            time.sleep(0.1)
            continue

        sensor = data.get("sensor_name")
        if sensor not in sensors_to_calibrate:
            continue
        if data.get("running_state") != "steady":
            continue

        fft_all = data.get("fft_all", {})
        if "Z" not in fft_all:
            continue

        spec_z = np.array(fft_all["Z"]["fft"])
        freq_z = np.array(fft_all["Z"]["index"])

        # 提取特征
        f3 = find_natural_freqs(spec_z, freq_z)
        rope_speed = data.get("rope_speed")
        if rope_speed is None:
            # 若无实时值，使用默认值（应与配置一致）
            rope_speed = 2.5
        fp_amp = get_fp_amplitude(spec_z, freq_z, rope_speed)

        collected[sensor].append((f3, fp_amp))
        print(f"传感器 {sensor} 已采集 {len(collected[sensor])}/{frames_needed} 帧")

        # 检查是否所有传感器都已采集足够帧数
        all_done = all(len(v) >= frames_needed for v in collected.values())
        if all_done:
            break

    # 计算平均值作为基线
    for sensor, data_list in collected.items():
        f3_sum = np.zeros(3)
        fp_amp_sum = 0.0
        valid_f3_count = 0
        for f3, fp_amp in data_list:
            if len(f3) >= 3:
                f3_sum += np.array(f3[:3])
                valid_f3_count += 1
            fp_amp_sum += fp_amp
        if valid_f3_count > 0:
            avg_f3 = (f3_sum / valid_f3_count).tolist()
        else:
            avg_f3 = []
        avg_fp_amp = fp_amp_sum / len(data_list)
        baseline[sensor] = {"f3": avg_f3, "fp_amp": avg_fp_amp}
        print(f"传感器 {sensor} 基线: f3={avg_f3}, fp_amp={avg_fp_amp:.4f}")

    # 保存到 JSON 文件
    output_path = os.path.join(sys.path[0], "rope_baseline.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)
    print(f"基线已保存至: {output_path}")

if __name__ == "__main__":
    main()