# get_result.py
import networkx as nx
from collections import defaultdict
from get_pcie import generate_combinations, generate_topology
from math import ceil
from typing import Sequence, Union

def top_percent_cumsum(
    access_times: Sequence[Union[int, float]], 
    percent: float
) -> Union[int, float]:
    if not 0 < percent <= 100:
        raise ValueError("percent has to be in (0, 100]")
    
    desc_sorted = sorted(access_times, reverse=True)
    
    k = max(1, ceil(len(desc_sorted) * percent / 100.0))
    
    return sum(desc_sorted[:k])

def max_flow_with_time(t, edges, front_pcie_bd, ssd_bd, n, source, sink, cpu_limits):
    # 创建原图
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
    
    # 考虑CPU容量问题
    for cpu, limit in cpu_limits.items():
        if G.has_node(cpu):
            for _, v in G.edges(cpu):
                original_capacity = G[cpu][v]['capacity']
                G[cpu][v]['capacity'] = min(original_capacity, limit)


    # 创建仅包含 SSD 边的子图，移除从 CPU 出发的边
    G_ssd = G.copy()
    cpu_edges = [(u, v) for u, v in G.edges() if 'CPU' in u]
    G_ssd.remove_edges_from(cpu_edges)

    # for u, v in list(G_ssd.edges()):
    #     if "CPU" in u:
    #         G_ssd.remove_edge(u, v)
    
    # 计算 SSD 边的最大流
    flow_value_ssd, flow_dict_ssd = nx.maximum_flow(G_ssd, source, sink)
    # print(f"SSD 边的最大流: {flow_value_ssd}")
    
    # 将 SSD 边的流量固定在原图中
    for u in flow_dict_ssd:
        for v in flow_dict_ssd[u]:
            if G.has_edge(u, v):
                G[u][v]['capacity'] -= flow_dict_ssd[u][v]
    
    # 计算包含所有边的最大流
    flow_value, flow_dict = nx.maximum_flow(G, source, sink)
    # print(f"包含所有边的最大流: {flow_value}")
    
    # 合并流量结果
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

def print_best_combinations(best_combinations, best_time, front_pcie_bd, ssd_bd, n_gb, source, sink, cpu_limits, total_combinations):
    print("使用的最优组合和边:")
    for idx, (comb, edges, ssd_desc) in enumerate(best_combinations):
        print(f"最优组合 {idx + 1}:")
        for part, devices in comb.items():
            print(f"{part}: (GPU: {devices.count('GPU')}, SSD: {devices.count('SSD')})")
            # print(f"{part}: (GPU: {devices.count('GPU')}, SSD: {devices.count('SSD')}) -> {devices}")
        # print("边集：")
        # for edge in edges:
        #     print(edge)
        # print("SSD 描述：")
        # for ssd_id, num in ssd_desc.items():
        #     print(f"{ssd_id}: {num}")

        # 计算每个方案的流量分布
        _, flow_dict = max_flow_with_time(best_time, edges, front_pcie_bd, ssd_bd, n_gb, source, sink, cpu_limits)
        print("流量分布：")
        gpu_flows = {}
        for gpu in ["GPU1", "GPU2"]:
            if gpu in flow_dict:
                total_flow = sum(flow_dict[gpu].values())
                gpu_flows[gpu] = total_flow
                print(f"{gpu} 流量: {total_flow:.2f} GB")

        ssd_flows = {}
        for ssd, count in ssd_desc.items():
            if ssd in flow_dict:
                total_flow = sum(flow_dict[ssd].values())
                individual_flow = total_flow / count
                ssd_flows[ssd] = individual_flow
                print(f"{ssd} 总流量: {total_flow:.2f} GB, 每个 {ssd} 的流量: {individual_flow:.2f} GB")

        cpu_flows = {}
        for cpu in ["CPU1", "CPU2"]:
            if cpu in flow_dict:
                total_flow = sum(flow_dict[cpu].values())
                print(f"{cpu} 流量: {total_flow:.4f} GB")

def run_maxflow(max_slots, num_gpu, num_ssd, num_cpu, ssd_bd, pcie_bd, dim, access_times, CPU_ratio):

    n_gb = top_percent_cumsum(access_times, 100) * dim * 4 / 1024 / 1024 / 1024

    lower_bound = 0
    upper_bound = 100000
    tolerance = 0.01

    source = "Src_node"
    sink = "Dst_node"
    # cpu_access_area_1 = 29515791
    # cpu_limit_from_area = cpu_access_area_1 * dim * 4 / 1024 / 1024 / 1024 / num_cpu
    cpu_limit_from_area = top_percent_cumsum(access_times, CPU_ratio) * dim * 4 / 1024 / 1024 / 1024 / num_cpu
    print('cpu_limit_from_area:', cpu_limit_from_area)
    cpu_limits = {
        "CPU1": cpu_limit_from_area,
        "CPU2": cpu_limit_from_area
    }

    best_time = float('inf')
    best_combinations = [] 
    # best_edges = []
    # best_ssd_counts = []

    combinations = generate_combinations(num_gpu, num_ssd, max_slots)
    # time_set = set()
    t_counts = defaultdict(int)
    total_combinations = len(combinations)
    for comb in combinations:
        edges, ssd_desc = generate_topology(comb, pcie_bd)
        min_time = find_min_time(edges, n_gb, source, sink, front_pcie_bd, ssd_bd, lower_bound, upper_bound, tolerance, cpu_limits)
        t_counts[min_time] += 1
        print(f"当前combination时间 t: {min_time:.2f} 秒")
        print("当前组合:")
        for part, devices in comb.items():
            # print(f"(GPU: {devices.count('GPU')}, SSD: {devices.count('SSD')}) -> {part}: {devices} ")
            print(f"(GPU: {devices.count('GPU')}, SSD: {devices.count('SSD')})")
        print()
        if min_time < best_time:
            best_time = min_time
            best_combinations = [(comb, edges, ssd_desc)]
        elif min_time == best_time:
            best_combinations.append((comb, edges, ssd_desc))
    

    print_best_combinations(best_combinations, best_time, front_pcie_bd, ssd_bd, n_gb, source, sink, cpu_limits, total_combinations)
    print(f"总共有 {total_combinations} 种组合")
    print(f"最小时间 t: {best_time:.2f} 秒")
    print("各 t 值及其出现的次数:")
    for t, count in t_counts.items():
        print(f"t = {t:.2f} s: 出现 {count} 次")
        
    hotness = 
    capacity = 
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
