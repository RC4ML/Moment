#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import subprocess
import sys
import json
import pprint
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict
from pprint import pprint
from typing import List, Dict, Any, Set

IGNORE_PCIE_INFO = {'Serial Attached SCSI controller'}
PRUNED_OUT  = Path("pruned_tree.txt")
SUBSET_JSON = Path("pcie_subset.json")


def collect_bdf_used() -> List[str]:
    slot_raw = subprocess.check_output(['dmidecode', '-t', 'slot'], text=True)
    slots, cur = {}, {}

    for line in slot_raw.splitlines():
        if line.startswith('Handle'):
            if cur and not cur.get('Designation', '').strip().endswith('OCP'):
                slots[cur['Designation']] = cur
            cur = {}
        m = re.match(r'\s*(\w[\w ]+):\s+(.*)', line)
        if m:
            cur[m.group(1).strip()] = m.group(2).strip()
    if cur and not cur.get('Designation', '').strip().endswith('OCP'):
        slots[cur['Designation']] = cur

    bdf_used: List[str] = []
    for info in slots.values():
        bdf = info.get('Bus Address')
        if not bdf or bdf.lower() == 'unknown':
            continue

        pci_info = subprocess.check_output(['lspci', '-s', bdf], text=True).strip()
        if any(key in pci_info for key in IGNORE_PCIE_INFO):
            continue
        bdf_used.append(bdf.lower())

    return bdf_used

def extract_bus_set(bdf_used: List[str]) -> Set[str]:
    return {bdf.split(':')[1] for bdf in bdf_used}

BUS_PAT = re.compile(
    r'\['
    r'(?:([0-9a-fA-F]{4}):)?'  
    r'([0-9a-fA-F]{2})'        
    r'(?:-([0-9a-fA-F]{2}))?'  
    r'\]'
)

def hex_in_range(lo: str, hi: str, targets: Set[str]) -> bool:
    lo_i, hi_i = int(lo, 16), int(hi, 16)
    return any(lo_i <= int(t, 16) <= hi_i for t in targets)

def calc_depth(line: str) -> int:
    return line.split('+-', 1)[0].count('|')

def parse_tree(lines: List[str], targets: Set[str]) -> List[Dict]:
    nodes: List[Dict] = []
    for ln in lines:
        depth  = calc_depth(ln)
        m      = BUS_PAT.search(ln)

        label  = None
        keep   = False
        if m:
            dom = m.group(1)
            lo  = m.group(2).lower()
            hi  = (m.group(3) or m.group(2)).lower()
            label = f"{dom.lower()}:{lo}" if dom and hi == lo else (f"{lo}-{hi}" if hi != lo else lo)
            keep  = hex_in_range(lo, hi, targets)

        nodes.append({'depth': depth,
                      'text':  ln.rstrip('\n'),
                      'label': label,
                      'keep':  keep})
    return nodes

def propagate_keep(nodes: List[Dict]) -> None:
    stack: List[int] = []
    for idx, node in enumerate(nodes):
        depth = node['depth']
        stack = stack[:depth]
        stack.append(idx)
        if node['keep']:
            for anc_idx in stack[:-1]:
                nodes[anc_idx]['keep'] = True

def prune_pcie_tree(target_bus: Set[str]):
    pcie_tree = subprocess.check_output(['lspci', '-t'], text=True)
    raw_lines = [ln for ln in pcie_tree.splitlines() if ln.strip()]

    nodes = parse_tree(raw_lines, target_bus)
    propagate_keep(nodes)

    pruned_lines = [n['text'] for n in nodes if n['keep']]
    return pruned_lines, nodes

def build_parent_children(pruned_lines: List[str]) -> Dict[str, List[str]]:

    parent_children = defaultdict(list)
    stack: List[tuple[int, str]] = []

    BUS_PAT = re.compile(
        r'\['
        r'(?:([0-9a-fA-F]{4}):)?'   
        r'([0-9a-fA-F]{2})'         
        r'(?:-([0-9a-fA-F]{2}))?'   
        r'\]'
    )

    for line in pruned_lines:
        matches = list(BUS_PAT.finditer(line))
        
        for match in matches:
            depth = match.start()

            while stack and stack[-1][0] >= depth:
                stack.pop()

            dom = match.group(1)
            lo = match.group(2).lower()
            hi = (match.group(3) or match.group(2)).lower()
            label = f"{dom.lower()}:{lo}" if dom and hi == lo else (f"{lo}-{hi}" if hi != lo else lo)

            if stack:
                parent_label = stack[-1][1]
                if label not in parent_children[parent_label]:
                    parent_children[parent_label].append(label)

            stack.append((depth, label))
            
    return dict(parent_children)

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
    
    def get_sort_key(group: Dict[str, Any]) -> tuple[int, int]:
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

def auto_gen_maxslots_and_connections():
    bdf_used = collect_bdf_used()
    target_bus = extract_bus_set(bdf_used)

    # print("1.lspci -t Tree ...")
    pruned_lines, nodes = prune_pcie_tree(target_bus)
    PRUNED_OUT.write_text("\n".join(pruned_lines), encoding='utf-8')

    # print("2.Get BDF ...")
    subset_map = build_parent_children(pruned_lines)
    SUBSET_JSON.write_text(json.dumps(subset_map, indent=2))

    # print("3.Get Max Slots and Connection")
    max_slots, connections = generate_topology_structures(subset_map, target_bus)
    
    # print("--- max_slots ---")
    # pprint(max_slots, sort_dicts=False)
    # print("--- connections ---")
    # pprint(connections)
    
    return max_slots, connections
    