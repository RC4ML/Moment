from itertools import product
import string

def generate_combinations(num_gpu, num_ssd, max_slots):
    total_slots = sum(max_slots.values())
    combinations = []

    def is_valid_combination(combination):
        total_gpus = sum(dev.count('GPU') for dev in combination.values())
        total_ssds = sum(dev.count('SSD') for dev in combination.values())
        return total_gpus <= num_gpu and total_ssds <= num_ssd

    if num_ssd + num_gpu < total_slots:
        for a_slots, b_slots, c_slots, d_slots in product(range(max_slots['A'] + 1), range(max_slots['B'] + 1), range(max_slots['C'] + 1), range(max_slots['D'] + 1)):
            if a_slots + b_slots + c_slots + d_slots != num_ssd + num_gpu:
                continue  
            
            for a_gpu, b_gpu, c_gpu, d_gpu in product(range(a_slots+1), range(1+1),range(c_slots + 1), range(d_slots + 1)):
                if a_gpu + b_gpu + c_gpu + d_gpu != num_gpu:
                    continue  
                if b_gpu > 1:
                    continue 
                
                c_devices = ['GPU'] * c_gpu + ['SSD'] * (c_slots - c_gpu)
                d_devices = ['GPU'] * d_gpu + ['SSD'] * (d_slots - d_gpu)
                a_devices = ['GPU'] * a_gpu + ['SSD'] * (a_slots - a_gpu)
                b_devices = ['GPU'] * b_gpu + ['SSD'] * (b_slots - b_gpu)
                # if b_gpu == 0:
                #     b_devices = ['SSD'] * (b_slots - 1)
                # else:
                #     b_devices = ['GPU'] * b_gpu + ['SSD'] * (b_slots - b_gpu)
                    
                if sum(len(devices) for devices in [a_devices, b_devices, c_devices, d_devices]) == num_ssd + num_gpu:
                    combination = {
                        'A': a_devices,
                        'B': b_devices,
                        'C': c_devices,
                        'D': d_devices
                    }
                    combinations.append(combination)
    else:
        for a_slots, b_slots, c_slots, d_slots in product(range(max_slots['A'] + 1), range(max_slots['B'] + 1), range(max_slots['C'] + 1), range(max_slots['D'] + 1)):
        # for c_slots, d_slots in product(range(max_slots['C'] + 1), range(max_slots['D'] + 1)):
            for a_gpu in range(min(a_slots, num_gpu) + 1):
                for b_gpu in range(min(1, num_gpu) + 1):
                    for c_gpu in range(min(c_slots, num_gpu) + 1):
                        for d_gpu in range(min(d_slots, num_gpu - c_gpu) + 1):
                            if a_gpu + b_gpu + c_gpu + d_gpu < 1:
                                continue

                            if b_gpu != 0:
                                c_devices = ['GPU'] * c_gpu + ['SSD'] * (c_slots - c_gpu)
                                d_devices = ['GPU'] * d_gpu + ['SSD'] * (d_slots - d_gpu)
                                a_devices = ['GPU'] * a_gpu + ['SSD'] * (a_slots - a_gpu)
                                b_devices = ['GPU'] * b_gpu + ['SSD'] * (b_slots - b_gpu)
                            else:
                                c_devices = ['GPU'] * c_gpu + ['SSD'] * (c_slots - c_gpu)
                                d_devices = ['GPU'] * d_gpu + ['SSD'] * (d_slots - d_gpu)
                                a_devices = ['GPU'] * a_gpu + ['SSD'] * (a_slots - a_gpu)
                                b_devices = ['SSD'] * b_slots

                            if sum(len(devices) for devices in [a_devices, b_devices, c_devices, d_devices]) == total_slots:
                                combination = {
                                    'A': a_devices,
                                    'B': b_devices,
                                    'C': c_devices,
                                    'D': d_devices
                                }
                                if is_valid_combination(combination):
                                    combinations.append(combination)

    return combinations

def get_origin_topos(connections, pcie_bd):
    edges = []
    edges.append(("Src_node", "CPU1", float("inf")))
    edges.append(("Src_node", "CPU2", float("inf")))
    
    cpu_directs = set()
    for a, b in connections:
        if a.endswith("_Direct_Buses"):
            cpu_directs.add(a)
        if b.endswith("_Direct_Buses"):
            cpu_directs.add(b)

    for bus in cpu_directs:
        if bus.startswith("CPU1"):
            edges.append(("CPU1", bus, pcie_bd))
            
        elif bus.startswith("CPU2"):
            edges.append(("CPU2", bus, pcie_bd))
            
    for a, b in connections:
        edges.append((a, b, pcie_bd))
        edges.append((b, a, pcie_bd))
        
    return edges

def generate_topology(combination, mapping, pcie_bd, max_slots, connections):
    
    edges = get_origin_topos(connections, pcie_bd)
    
    temp_map = mapping
    
    gpu_count = 0
    ssd_count = {}  
    ssd_desc = {}
    for part, devices in combination.items():
        num_ssds = devices.count('SSD')
        if num_ssds > 0:
            if num_ssds not in ssd_count:
                ssd_count[num_ssds] = 0
            ssd_name = f"{part}_SSD{num_ssds}_{ssd_count[num_ssds]}"
            ssd_count[num_ssds] += 1  
            ssd_desc[ssd_name] = num_ssds
            edges.append(("Src_node", ssd_name, float("inf")))

            temp_name = temp_map[part]
            edges.append((ssd_name, temp_name, f"SSD*{num_ssds}"))

    for part, devices in combination.items():
        num_gpus = devices.count('GPU')
        temp_name = temp_map[part]
        for _ in range(num_gpus):
            gpu_name = f"GPU{gpu_count + 1}"
            edges.append((temp_name, gpu_name, pcie_bd))
            gpu_count += 1

    for i in range(gpu_count):
        edges.append((f"GPU{i + 1}", "Dst_node", float("inf")))

    return edges, ssd_desc

if __name__ == "__main__":
    max_slots = {
        'A': 1,
        'B': 9,
        'C': 4,
        'D': 5
    }
    num_gpu = 4
    num_ssd = 8
    pcie_bd = 20
    combinations = generate_combinations(num_gpu, num_ssd, max_slots)
    total_fa = 0
    for comb in combinations:
        edges, ssd_desc = generate_topology(comb, pcie_bd)
        for edge in edges:
            print(edge) 

