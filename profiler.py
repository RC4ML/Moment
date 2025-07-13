import os
import subprocess
import re
from pathlib import Path
from typing import Tuple

def get_hardware():
    max_slots = {
        'A': 1, 
        'B': 9, 
        'C': 4, 
        'D': 5
    }
    return max_slots

SSD_PAT = re.compile(
    r"bandwidth[^0-9+-.]*([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?\d+)?)\s*([kMGT]i?B/s)",
    re.I,
)

PCIE_PAT = re.compile(
    r"""Host\s*                 # 'Host' 后可有空格
        (?:→|->)\s*Device:      # 支持 '→' 或 '->'
        \s*([0-9]+(?:\.[0-9]+)?)# 捕获带宽数值
        \s*(Gi?B/s)             # 捕获单位（GiB/s 或 GB/s）
    """,
    re.I | re.X                 # 忽略大小写 + 允许空白注释
)

def measure_ssd_bw(num_ssd: int,
                   granularity: int,
                   exe_path: str | Path = "./IOStack/test_ssd",
                   use_sudo: bool = True,
                   verbose: bool = False) -> Tuple[float, float, str]:
    cmd = ([ "sudo" ] if use_sudo else []) + \
          [ str(exe_path), str(num_ssd), str(granularity) ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # 把 stderr 合并进 stdout
        text=True,
        check=True,
    )

    if verbose:
        print("=== raw output ===")
        print(proc.stdout)

    for line in proc.stdout.splitlines():
        m = SSD_PAT.search(line)
        if m:
            total_bw = float(m.group(1))
            unit = m.group(2)
            per_ssd = total_bw / num_ssd if num_ssd else 0.0
            return total_bw, per_ssd, unit

    raise RuntimeError("Error")


def measure_pcie_bw(
                   exe_path: str | Path = "./IOStack/test_pcie",
                   use_sudo: bool = True,
                   verbose: bool = False) -> Tuple[float, float, str]:
    cmd = ([ "sudo" ] if use_sudo else []) + \
          [ str(exe_path), str(40), str(4) ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # 把 stderr 合并进 stdout
        text=True,
        check=True,
    )

    if verbose:
        print("=== raw output ===")
        print(proc.stdout)

    for line in proc.stdout.splitlines():
        m = PCIE_PAT.search(line)
        if m:
            total_bw = float(m.group(1))
            # unit = m.group(2)
            return total_bw#, unit

    raise RuntimeError("Error")

def run_profiler(num_gpu, num_ssd, dim):
    print("Start Profilling")
    total, ssd_bd, unit = measure_ssd_bw(num_ssd, dim*4)
    ssd_bd = ssd_bd / 1000.0
    print("SSD bandwidth {}GiB/s".format(ssd_bd))
    pcie_bd = measure_pcie_bw()
    print("PCIe bandwidth {}GiB/s".format(pcie_bd))

    os.system("./sampling_server/build/bin/server {} {}".format(num_gpu, 0))
    return ssd_bd, pcie_bd

def read_access_times(file_path):
    """ Read access times from a given file. Each line should contain an integer representing the access count for a sample. """
    access_times = []
    total_lines = sum(1 for line in open(file_path, 'r'))  # Quickly count lines for progress tracking
    print(f"Total lines to read: {total_lines}")
    try:
        with open(file_path, 'r') as file:
            for i, line in enumerate(file):
                access = int(line.strip())
                access_times.append(access)
                if (i + 1) % (total_lines // 10) == 0:  # Print progress every 10%
                    print(f"Read Progress: {(i + 1) / total_lines * 100:.2f}%")
    except Exception as e:
        print(f"Error reading the file: {e}")
    return access_times

if __name__ == "__main__":
    DEBUG = True 
    max_slots = get_hardware()
    print("max_slots =", max_slots)
    run_profiler(2, 3, 1024)
