from itertools import product

def generate_combinations(num_gpu, num_ssd, max_slots):
    # 计算总插槽数量
    total_slots = sum(max_slots.values())
    combinations = []

    def is_valid_combination(combination):
        total_gpus = sum(dev.count('GPU') for dev in combination.values())
        total_ssds = sum(dev.count('SSD') for dev in combination.values())
        return total_gpus <= num_gpu and total_ssds <= num_ssd

    if num_ssd + num_gpu < total_slots:
        # 设备总数小于插槽总数，遍历所有可能的设备分配组合
        for a_slots, b_slots, c_slots, d_slots in product(range(max_slots['A'] + 1), range(max_slots['B'] + 1), range(max_slots['C'] + 1), range(max_slots['D'] + 1)):
            if a_slots + b_slots + c_slots + d_slots != num_ssd + num_gpu:
                continue  # 只考虑使用了所有设备的配置
            
            # 遍历所有可能的GPU分配方式
            for a_gpu, b_gpu, c_gpu, d_gpu in product(range(a_slots+1), range(1+1),range(c_slots + 1), range(d_slots + 1)):
                if a_gpu + b_gpu + c_gpu + d_gpu != num_gpu:
                    continue  # 确保使用了正确数量的GPU
                if b_gpu > 1:
                    continue  # 确保B板块最多只有一个GPU
                
                c_devices = ['GPU'] * c_gpu + ['SSD'] * (c_slots - c_gpu)
                d_devices = ['GPU'] * d_gpu + ['SSD'] * (d_slots - d_gpu)
                a_devices = ['GPU'] * a_gpu + ['SSD'] * (a_slots - a_gpu)
                b_devices = ['GPU'] * b_gpu + ['SSD'] * (b_slots - b_gpu)
                # if b_gpu == 0:
                #     b_devices = ['SSD'] * (b_slots - 1)
                # else:
                #     b_devices = ['GPU'] * b_gpu + ['SSD'] * (b_slots - b_gpu)
                    
                # 确保所有部分的设备使用总数符合要求
                if sum(len(devices) for devices in [a_devices, b_devices, c_devices, d_devices]) == num_ssd + num_gpu:
                    combination = {
                        'A': a_devices,
                        'B': b_devices,
                        'C': c_devices,
                        'D': d_devices
                    }
                    combinations.append(combination)
    else:
        # 设备总数大于或等于插槽总数，生成所有可能的填满插槽的组合
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

def generate_topology(combination, pcie_bd):
    edges = [
        ("Src_node", "CPU1", float("inf")), 
        ("Src_node", "CPU2", float("inf")),
        ("CPU1", "Temp3", pcie_bd),
        ("CPU2", "Temp4", pcie_bd),
        ("Temp1", "Temp3", pcie_bd),
        ("Temp3", "Temp1", pcie_bd),
        ("Temp3", "Temp4", pcie_bd),
        ("Temp4", "Temp3", pcie_bd),
        ("Temp2", "Temp1", pcie_bd),
        ("Temp1", "Temp2", pcie_bd)
    ]
    temp_map = {'A': 'Temp3', 'B': 'Temp4', 'C': 'Temp1', 'D': 'Temp2'}
    gpu_count = 0
    ssd_count = {}  # 用来统计各数量SSD的出现次数，确保ID的唯一性
    ssd_desc = {}
    # 构造Src_node到SSD的连接
    for part, devices in combination.items():
        num_ssds = devices.count('SSD')
        if num_ssds > 0:
            # 检查这个数量的SSD是否已经有过，然后相应地更新其ID
            if num_ssds not in ssd_count:
                ssd_count[num_ssds] = 0
            ssd_name = f"{part}_SSD{num_ssds}_{ssd_count[num_ssds]}"
            ssd_count[num_ssds] += 1  # 更新这个数量的SSD的计数器
            ssd_desc[ssd_name] = num_ssds
            edges.append(("Src_node", ssd_name, float("inf")))

            # 构造SSD到Temp的连接
            temp_name = temp_map[part]
            edges.append((ssd_name, temp_name, f"SSD*{num_ssds}"))

    # 构造Temp到GPU的连接
    for part, devices in combination.items():
        num_gpus = devices.count('GPU')
        temp_name = temp_map[part]
        for _ in range(num_gpus):
            gpu_name = f"GPU{gpu_count + 1}"
            edges.append((temp_name, gpu_name, pcie_bd))
            gpu_count += 1

    # 构造GPU到Dst_node的连接
    for i in range(gpu_count):
        edges.append((f"GPU{i + 1}", "Dst_node", float("inf")))

    return edges, ssd_desc

# 示例使用
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
    # 调用函数并打印结果
    combinations = generate_combinations(num_gpu, num_ssd, max_slots)
    total_fa = 0
    for comb in combinations:
        edges, ssd_desc = generate_topology(comb, pcie_bd)
        for edge in edges:
            print(edge) 
    #     if 'GPU' in comb['C'] and 'GPU' in comb['D']:
    #         total_fa += 1
    #         print(comb)
    # print(total_fa)
        # for ssd_id, num in ssd_desc.items():
        #     print(f"{ssd_id}: {num}")
    # for idx, combo in enumerate(combinations):
    #     print(f"组合方案 {idx + 1}:")
    #     for part, devices in combo.items():
    #         print(f"(GPU: {devices.count('GPU')}, SSD: {devices.count('SSD')}) -> {part}: {devices} ")
    #     print()
