#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pprint import pprint
from typing import List, Dict, Any, Set, Tuple

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

    child_to_parent = {child: parent for parent, children in parent_map.items() for child in children}

    all_logical_groups = []
    
    direct_cpu0_buses, direct_cpu1_buses = [], []
    for parent, children in parent_map.items():
        if parent.startswith("0000:"):
            is_cpu1 = int(parent.split(':')[1], 16) >= 0x80
            for child in children:
                if child not in parent_map:
                    (direct_cpu1_buses if is_cpu1 else direct_cpu0_buses).extend(parse_bus_range(child))
    if direct_cpu0_buses:
        all_logical_groups.append({"name": "CPU1_Direct_Buses", "bdf_list": sorted(direct_cpu0_buses)})
    if direct_cpu1_buses:
        all_logical_groups.append({"name": "CPU2_Direct_Buses", "bdf_list": sorted(direct_cpu1_buses)})

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
    
    def get_sort_key(group: Dict[str, Any]) -> Tuple[int, int]:

        if group['name'].startswith('CPU'):
            group_type_priority = 0
            cpu_num = int(group['name'][3]) 
            sort_value = cpu_num
            return (group_type_priority, sort_value)
        else: 
            group_type_priority = 1
            parent_bus_str = group['direct_parent'].split('-')[0]
            sort_value = -int(parent_bus_str, 16)
            return (group_type_priority, sort_value)

    final_groups = sorted(final_groups_unfiltered, key=get_sort_key)

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
            cpu_group_name = "CPU2_Direct_Buses" if int(root_port.split(':')[1], 16) >= 0x80 else "CPU1_Direct_Buses"
            connections.append((cpu_group_name, group['name']))

    return max_slots, connections

if __name__ == "__main__":
    max_slots, connections = generate_topology_structures(PCIE_HIERARCHY_DATA, TARGET_BUS)

    print("--- max_slots ---")
    pprint(max_slots, sort_dicts=False)
    print("\n" + "="*40 + "\n")

    print("--- connections ---")
    pprint(connections)