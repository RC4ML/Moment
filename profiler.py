import os
import subprocess
import re
import string
from pathlib import Path
from typing import Tuple
from prune_tree.auto_prune_pcie_tree import auto_gen_maxslots_and_connections

def get_hardware():
    print("Get Hardware: Slots and Connections")
    module_max_slots, connections = auto_gen_maxslots_and_connections()

    letters = string.ascii_uppercase[:len(module_max_slots)]
    ordered_keys = list(module_max_slots.keys())

    letter_mapping = dict(zip(letters, ordered_keys))
    max_slots = {letter: module_max_slots[key]
                 for letter, key in zip(letters, ordered_keys)}

    return max_slots, connections, letter_mapping


def parse_bandwidth_file(path="bandwidth_results.txt"):
    with open(path, "r") as f:
        line1 = f.readline().strip()
        ssd_bd = float(line1)

        line2 = f.readline().strip()
        pcie_bd = float(line2)

    return ssd_bd, pcie_bd

def run_profiler(num_gpu, num_ssd, dim):
    print("Start Profilling")
    os.system("./IOStack/test_ssd {} {} > /dev/null 2>&1".format(num_ssd, dim*4))
    os.system("./IOStack/test_pcie {} {} > /dev/null 2>&1".format(40, 4))
    ssd_bd, pcie_bd = parse_bandwidth_file()
    ssd_bd = ssd_bd / num_ssd
    print("SSD bandwidth {}GiB/s".format(ssd_bd))
    print("PCIe bandwidth {}GiB/s".format(pcie_bd))
    os.system("./sampling_server/build/bin/server {} {} ".format(num_gpu, 0))
    return ssd_bd, pcie_bd

def read_access_times(file_path):
    """ Read access times from a given file. Each line should contain an integer representing the access count for a sample. """
    access_times = []
    total_lines = sum(1 for line in open(file_path, 'r'))  # Quickly count lines for progress tracking
    print(f"Read Access Time")
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
    max_slots, connections = get_hardware()
    # print("max_slots =", max_slots)
    # run_profiler(2, 3, 1024)
    print(max_slots)
