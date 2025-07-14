#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_pcie_topology_final_v4.py
------------------------------------------------------------
最终版本。该脚本能：
1. 精确识别独立的CPU直连组和真正的PLX组。 (>=0x80为CPU1, 反之为CPU0)
2. 根据组内目标BDF的范围，动态地为PLX组命名 (e.g., "PLX_3a_3d")。
3. 按“先CPU后PLX，组内按拓扑顺序”的规则进行排序。
4. 生成 max_slots 字典和 connections 列表。
"""
from pprint import pprint
from typing import List, Dict, Any, Set, Tuple

# ─────────────────────────────────────────────
# 输入数据
# ─────────────────────────────────────────────

PCIE_HIERARCHY_DATA = {
  "0000:e2": ["e3", "e4", "e5", "e6"],
  "0000:c9": ["ca", "cb", "cc", "cd"],
  "0000:b0": ["b1"],
  "0000:30": ["31-3d"],
  "31-3d": ["32-3d"],
  "32-3d": ["33-39", "3a", "3b", "3c", "3d"],
  "33-39": ["34-39"],
  "34-39": ["35", "36", "37", "38", "39"],
  "0000:16": ["17"]
}

TARGET_BUS = {'37', 'e4', 'cd', '3c', 'b1', '3b', '3d', '35', '17', 'e3', 'ca', 'cc', '3a', 'e6', '36', 'e5', '38', '39', 'cb'}

# ─────────────────────────────────────────────
# 核心分析逻辑
# ─────────────────────────────────────────────

def parse_bus_range(bus_str: str) -> List[str]:
    if '-' in bus_str:
        start_hex, end_hex = bus_str.split('-')
        start, end = int(start_hex, 16), int(end_hex, 16)
        return [f'{i:02x}' for i in range(start, end + 1)]
    return [bus_str.lower()]

def generate_topology_structures(
    parent_map: Dict[str, List[str]],
    target_buses: Set[str]
) -> (Dict[str, int], List[tuple]):
    """
    主分析函数，集成了动态命名和按值排序的逻辑。
    """
    # 步骤 1: 创建逆向索引 (child -> parent)，用于路径追踪
    child_to_parent = {child: parent for parent, children in parent_map.items() for child in children}

    # 步骤 2: 识别所有潜在的逻辑组
    all_logical_groups = []
    
    # -- 识别 "Direct-Attached" 组 (新规则: >=0x80 为 CPU1) --
    direct_cpu0_buses, direct_cpu1_buses = [], []
    for parent, children in parent_map.items():
        if parent.startswith("0000:"):
            # 新规则: >= 0x80 为 CPU1, 否则为 CPU0
            is_cpu1 = int(parent.split(':')[1], 16) >= 0x80
            for child in children:
                if child not in parent_map:
                    (direct_cpu1_buses if is_cpu1 else direct_cpu0_buses).extend(parse_bus_range(child))
    if direct_cpu0_buses:
        all_logical_groups.append({"name": "CPU1_Direct_Buses", "bdf_list": sorted(direct_cpu0_buses)})
    if direct_cpu1_buses:
        all_logical_groups.append({"name": "CPU2_Direct_Buses", "bdf_list": sorted(direct_cpu1_buses)})

    # -- 识别真正的 "PLX group" --
    for bridge, children in parent_map.items():
        if bridge.startswith("0000:"): continue
        leaf_buses_under_bridge = []
        for child in children:
            if child not in parent_map:
                leaf_buses_under_bridge.extend(parse_bus_range(child))
        if leaf_buses_under_bridge:
            all_logical_groups.append({
                "name": f"PLX group under '{bridge}'",
                "bdf_list": sorted(leaf_buses_under_bridge),
                "direct_parent": bridge
            })

    # 步骤 3: 过滤、动态命名、并收集排序所需信息
    final_groups_unfiltered = []
    for group in all_logical_groups:
        relevant_buses = sorted(list(set(group['bdf_list']) & target_buses))
        if relevant_buses:
            if 'direct_parent' in group:
                min_bdf = relevant_buses[0]
                max_bdf = relevant_buses[-1]
                group['name'] = f"PLX_{min_bdf}_{max_bdf}" if min_bdf != max_bdf else f"PLX_{min_bdf}"
            group['bdf_list'] = relevant_buses
            group['count'] = len(relevant_buses)
            final_groups_unfiltered.append(group)
    
    # 步骤 4: 根据“先CPU后PLX，组内按规则”进行排序
    def get_sort_key(group: Dict[str, Any]) -> Tuple[int, int]:
        """
        为组生成一个元组排序键 (priority, value) 来实现多级排序。
        - 优先级: CPU=0, PLX=1。这确保了CPU组总在前面。
        - 排序值: 
          - CPU组: 按CPU编号(0, 1)升序。
          - PLX组: 按父桥起始总线号的负值升序，等效于按父桥总线号降序，以匹配拓扑。
        """
        if group['name'].startswith('CPU'):
            group_type_priority = 0
            cpu_num = int(group['name'][3]) # 从 "CPU0..." 或 "CPU1..." 中提取数字
            sort_value = cpu_num
            return (group_type_priority, sort_value)
        else: # PLX 组
            group_type_priority = 1
            parent_bus_str = group['direct_parent'].split('-')[0]
            # 使用负值，使得在整体升序排序时，总线号大的PLX组排在前面
            sort_value = -int(parent_bus_str, 16)
            return (group_type_priority, sort_value)

    final_groups = sorted(final_groups_unfiltered, key=get_sort_key)

    # 步骤 5: 从排序后的组列表中生成最终的数据结构
    max_slots = {g['name']: g['count'] for g in final_groups}

    connections = []
    connections.append(("CPU1_Direct_Buses", "CPU2_Direct_Buses"))
    parent_bridge_to_group_name = {g['direct_parent']: g['name'] for g in final_groups if 'direct_parent' in g}
    for group in final_groups:
        if 'direct_parent' not in group: continue
        current_parent = group['direct_parent']
        while current_parent in child_to_parent:
            ancestor_parent = child_to_parent[current_parent]
            if ancestor_parent in parent_bridge_to_group_name:
                connections.append((parent_bridge_to_group_name[ancestor_parent], group['name']))
                break
            current_parent = ancestor_parent
        else:
            root_port = current_parent
            # 确保这里的CPU命名规则与前面一致
            cpu_group_name = "CPU2_Direct_Buses" if int(root_port.split(':')[1], 16) >= 0x80 else "CPU1_Direct_Buses"
            connections.append((cpu_group_name, group['name']))

    return max_slots, connections

if __name__ == "__main__":
    max_slots, connections = generate_topology_structures(PCIE_HIERARCHY_DATA, TARGET_BUS)

    print("--- max_slots ---")
    # 使用 sort_dicts=False (Python 3.7+ 默认为插入顺序) 来展示我们排序后的结果
    pprint(max_slots, sort_dicts=False)
    print("\n" + "="*40 + "\n")

    print("--- connections ---")
    pprint(connections)