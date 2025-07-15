import heapq
import random
import numpy as np
from maxflow import run_maxflow
from ddak import run_ddak
from profiler import get_hardware, run_profiler, read_access_times

file_path = "/share/gnn_data/igb260m/IGB-Datasets/data/"  # Replace with your file path
feature_dim = 1024
num_gpu = 2
num_ssd = 6
num_cpu = 2

max_slots, connections, mapping = get_hardware()
# print(max_slots, connections, mapping)
ssd_bd, pcie_bd = run_profiler(num_gpu, num_ssd, feature_dim) # Measure hardware bandwidth and measure hotness by pre-sampling
# ssd_bd, pcie_bd = 4.8, 22.39
access_times = read_access_times(file_path + "accesstimes") 
num_node = num_ssd + num_cpu
hotness, capacity = run_maxflow(max_slots, connections, mapping, num_gpu, num_ssd, num_cpu, ssd_bd, pcie_bd, feature_dim, access_times, CPU_ratio=1)
run_ddak(file_path, access_times, hotness, capacity, num_node)