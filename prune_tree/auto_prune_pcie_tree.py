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

# ─────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────
IGNORE_PCIE_INFO = {'Serial Attached SCSI controller'}
PRUNED_OUT  = Path("pruned_tree.txt")
SUBSET_JSON = Path("pcie_subset.json")

# ─────────────────────────────────────────────
# 1) 获取已用 BDF
# ─────────────────────────────────────────────
def collect_bdf_used() -> List[str]:
    """解析 dmidecode，返回已用 BDF 列表"""
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

# ─────────────────────────────────────────────
# 2) 生成 bus 集合
# ─────────────────────────────────────────────
def extract_bus_set(bdf_used: List[str]) -> Set[str]:
    return {bdf.split(':')[1] for bdf in bdf_used}

# ─────────────────────────────────────────────
# 3) 解析 / 剪枝 lspci -t
# ─────────────────────────────────────────────
# 方括号标签正则：支持 [0000:e2] / [e3] / [31-3d]
BUS_PAT = re.compile(
    r'\['
    r'(?:([0-9a-fA-F]{4}):)?'   # 可选 domain
    r'([0-9a-fA-F]{2})'         # lo
    r'(?:-([0-9a-fA-F]{2}))?'   # 可选 hi
    r'\]'
)

def hex_in_range(lo: str, hi: str, targets: Set[str]) -> bool:
    lo_i, hi_i = int(lo, 16), int(hi, 16)
    return any(lo_i <= int(t, 16) <= hi_i for t in targets)

def calc_depth(line: str) -> int:
    """depth = 前缀竖线 '|' 数量"""
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
    """子节点 keep=True ⇒ 祖先也 keep=True"""
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

# ─────────────────────────────────────────────
# 4) 构建父 → 子映射
# ─────────────────────────────────────────────
def build_parent_children(pruned_lines: List[str]) -> Dict[str, List[str]]:
    """
    根据 lspci -t 的缩进和行内结构，构建精确的父子 BDF 映射。
    这个版本可以正确处理单行内的多级拓扑。
    """
    parent_children = defaultdict(list)
    # 栈中存放元组 (深度, 标签)，深度即节点在行中的起始列位置
    stack: List[tuple[int, str]] = []

    # 与原脚本相同的正则表达式
    BUS_PAT = re.compile(
        r'\['
        r'(?:([0-9a-fA-F]{4}):)?'   # 可选 domain
        r'([0-9a-fA-F]{2})'         # lo
        r'(?:-([0-9a-fA-F]{2}))?'   # 可选 hi
        r'\]'
    )

    for line in pruned_lines:
        # 找出当前行内所有的 BDF 标签及其位置
        matches = list(BUS_PAT.finditer(line))
        
        for match in matches:
            # 节点的深度由其在行中的起始位置决定
            depth = match.start()

            # 根据深度弹出栈，直到栈顶元素的深度小于当前节点深度
            # 此时的栈顶元素就是当前节点的父节点
            while stack and stack[-1][0] >= depth:
                stack.pop()

            # 构建节点标签，与原脚本逻辑一致
            dom = match.group(1)
            lo = match.group(2).lower()
            hi = (match.group(3) or match.group(2)).lower()
            label = f"{dom.lower()}:{lo}" if dom and hi == lo else (f"{lo}-{hi}" if hi != lo else lo)

            # 如果栈不为空，说明存在父节点，建立映射关系
            if stack:
                parent_label = stack[-1][1]
                # 避免重复添加同一个子节点
                if label not in parent_children[parent_label]:
                    parent_children[parent_label].append(label)

            # 将当前节点压入栈中，作为后续节点的潜在父节点
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
    def get_sort_key(group: Dict[str, Any]) -> tuple[int, int]:
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


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────
def auto_gen_maxslots_and_connections():
    print("1.收集 BDF ...")
    bdf_used = collect_bdf_used()
    target_bus = extract_bus_set(bdf_used)
    print(f"Done, 检测到 {len(bdf_used)} 条有效 BDF: {target_bus}")

    print("2.lspci -t 树 ...")
    pruned_lines, nodes = prune_pcie_tree(target_bus)
    PRUNED_OUT.write_text("\n".join(pruned_lines), encoding='utf-8')
    print(f"Done, 已写入{PRUNED_OUT.resolve()}")

    print("3.获取BDF父子关系 ...")
    subset_map = build_parent_children(pruned_lines)
    SUBSET_JSON.write_text(json.dumps(subset_map, indent=2))
    print(f"Done, 父子映射写入 {SUBSET_JSON.resolve()}")

    print("4.获取模块Slots数量及连接关系")
    max_slots, connections = generate_topology_structures(subset_map, target_bus)
    
    print("--- max_slots ---")
    pprint(max_slots, sort_dicts=False)
    print("--- connections ---")
    pprint(connections)
    print()
    
    return max_slots, connections
    
    
if __name__ == "__main__":
    try:
        max_slots, connections = auto_gen_maxslots_and_connections()
    except subprocess.CalledProcessError as e:
        sys.exit(f"命令执行失败: {e}")