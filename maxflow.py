# get_result.py
import networkx as nx
import random
from collections import defaultdict
from get_pcie import generate_combinations, generate_topology, get_origin_topos
from math import ceil
from typing import Sequence, Union

def top_percent_cumsum(
    access_times, 
    percent
) -> Union[int, float]:
    if not 0 < percent <= 100:
        raise ValueError("percent has to be in (0, 100]")
    
    desc_sorted = sorted(access_times, reverse=True)
    
    k = max(1, ceil(len(desc_sorted) * percent / 100.0))
    # print(k)
    return sum(desc_sorted[:k])

def max_flow_with_time(t, edges, front_pcie_bd, ssd_bd, n, source, sink, cpu_limits):
    G = nx.DiGraph()
    for u, v, capacity in edges:
        if isinstance(capacity, str) and 'SSD' in capacity:
            ssd_count = int(capacity.split('*')[-1])
            board_part = u.split('_')[0] 
            if board_part in ['C', 'D']:
                G.add_edge(u, v, capacity=ssd_bd * ssd_count * t)
            else:
                G.add_edge(u, v, capacity=front_pcie_bd * ssd_count * t)
        elif isinstance(capacity, (int, float)):
            G.add_edge(u, v, capacity=capacity * t)
        else:
            G.add_edge(u, v, capacity=float('inf'))
    
    for cpu, limit in cpu_limits.items():
        if G.has_node(cpu):
            for _, v in G.edges(cpu):
                original_capacity = G[cpu][v]['capacity']
                G[cpu][v]['capacity'] = min(original_capacity, limit)


    G_ssd = G.copy()
    cpu_edges = [(u, v) for u, v in G.edges() if 'CPU' in u]
    G_ssd.remove_edges_from(cpu_edges)

    # for u, v in list(G_ssd.edges()):
    #     if "CPU" in u:
    #         G_ssd.remove_edge(u, v)
    
    flow_value_ssd, flow_dict_ssd = nx.maximum_flow(G_ssd, source, sink)
    
    for u in flow_dict_ssd:
        for v in flow_dict_ssd[u]:
            if G.has_edge(u, v):
                G[u][v]['capacity'] -= flow_dict_ssd[u][v]
    
    flow_value, flow_dict = nx.maximum_flow(G, source, sink)
    
    for u in flow_dict_ssd:
        for v in flow_dict_ssd[u]:
            if u in flow_dict and v in flow_dict[u]:
                flow_dict[u][v] += flow_dict_ssd[u][v]
            else:
                if u not in flow_dict:
                    flow_dict[u] = {}
                flow_dict[u][v] = flow_dict_ssd[u][v]
    
    total_flow_value = flow_value + flow_value_ssd
    return total_flow_value, flow_dict

def find_min_time(edges, n, source, sink, front_pcie_bd, ssd_bd, lower_bound, upper_bound, tolerance, cpu_limits):
    while upper_bound - lower_bound > tolerance:
        mid = (lower_bound + upper_bound) / 2
        flow_value, _ = max_flow_with_time(mid, edges, front_pcie_bd, ssd_bd, n, source, sink, cpu_limits)
        # print(flow_value, mid)
        if flow_value >= n:
            upper_bound = mid
        else:
            lower_bound = mid
    return (lower_bound + upper_bound) / 2

def get_best_combinations(best_combinations, best_time, front_pcie_bd, ssd_bd, n_gb, source, sink, cpu_limits, total_combinations, dim, num_cpus, num_gpus):
    cpu_list = [f"CPU{i}" for i in range(1, num_cpus + 1)]
    gpu_list = [f"GPU{i}" for i in range(1, num_gpus + 1)]

    granularity = 4096
    BYTES_IN_TB = 1024 * 1024 * 1024

    index = random.randint(0, len(best_combinations) - 1)
    comb, edges, ssd_desc = best_combinations[index]

    print(f"One of the Best Combination:")
    for part, devices in comb.items():
        # print(f"{part}: (GPU: {devices.count('GPU')}, SSD: {devices.count('SSD')})")
        print(f"{part}: (GPU: {devices.count('GPU')}, SSD: {devices.count('SSD')}) -> {devices}")

    _, flow_dict = max_flow_with_time(best_time, edges, front_pcie_bd, ssd_bd, n_gb, source, sink, cpu_limits)

    hotness = []
    
    for cpu in cpu_list:
        if cpu in flow_dict:
            total_flow = sum(flow_dict[cpu].values())
            print(f"{cpu} traffic: {total_flow:.4f} GB")
            hotness_value = total_flow * BYTES_IN_TB / granularity
            hotness.append(hotness_value)
    
    ssd_flows = {}
    for ssd, count in ssd_desc.items():
        if ssd in flow_dict:
            total_flow = sum(flow_dict[ssd].values())
            individual_flow = total_flow / count
            ssd_flows[ssd] = individual_flow
            print(f"Traffic of Interconnect Node {ssd}: {total_flow:.2f} GB  ")
            hotness_value = individual_flow * BYTES_IN_TB / granularity
            for _ in range(count):
                hotness.append(hotness_value)

    return hotness


def run_maxflow(max_slots, connections, mapping, num_gpu, num_ssd, num_cpu, ssd_bd, pcie_bd, dim, access_times, CPU_ratio, SSD_cap=3.5):
    n_gb = top_percent_cumsum(access_times, 100) * dim * 4 / 1024 / 1024 / 1024
    # print("n_gb:", n_gb)
    
    lower_bound = 0
    upper_bound = 100000
    tolerance = 0.01

    granularity = 4096
    BYTES_IN_TB = 1024 * 1024 * 1024 * 1024
    
    source = "Src_node"
    sink = "Dst_node"
    # cpu_access_area_1 = 29515791
    # cpu_limit_from_area = cpu_access_area_1 * dim * 4 / 1024 / 1024 / 1024 / num_cpu
    
    cpu_limit_from_area = top_percent_cumsum(access_times, CPU_ratio) * dim * 4 / 1024 / 1024 / 1024 / num_cpu
    # print('cpu_limit_from_area:', cpu_limit_from_area)
    
    cpu_limits = {
        "CPU1": cpu_limit_from_area / 2,
        "CPU2": cpu_limit_from_area / 2
    }

    best_time = float('inf')
    best_combinations = [] 

    combinations = generate_combinations(num_gpu, num_ssd, max_slots)
    # time_set = set()
    t_counts = defaultdict(int)
    total_combinations = len(combinations)
    for comb in combinations:
        edges, ssd_desc = generate_topology(comb, mapping, pcie_bd, max_slots, connections)
        front_pcie_bd = ssd_bd
        min_time = find_min_time(edges, n_gb, source, sink, front_pcie_bd, ssd_bd, lower_bound, upper_bound, tolerance, cpu_limits)
        t_counts[min_time] += 1
        if min_time < best_time:
            best_time = min_time
            best_combinations = [(comb, edges, ssd_desc)]
        elif min_time == best_time:
            best_combinations.append((comb, edges, ssd_desc))
    
    # print(f'best_time: {best_time} s')
    hotness = get_best_combinations(best_combinations, best_time, front_pcie_bd, ssd_bd, n_gb, source, sink, cpu_limits, total_combinations, dim, num_gpu, 2)
    capacity = [int(CPU_ratio*len(access_times)) / num_cpu] * num_cpu + [SSD_cap * BYTES_IN_TB / granularity] * num_ssd 
    
    # print(hotness)
    # print(capacity)
    return hotness, capacity
    

if __name__ == "__main__":
    max_slots = {
        'A': 1, 
        'B': 9, 
        'C': 4, 
        'D': 5
    }
    num_gpu = 2
    num_ssd = 4
    num_cpu = 2
    ssd_bd = front_pcie_bd = 6.4
    pcie_bd = 20
    run_maxflow()
