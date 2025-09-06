# ----------- Imports & Setup ----------- #
import socket
import json
import time
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
import itertools
from collections import defaultdict
from collections import deque

# ----------- Map Definition ----------- #
MAPS = [
    {
        "name": "Your Map",
        "description": "Main test map",
        "data": np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 0, 1, 1, 0, 1],
            [1, 0, 1, 1, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
        ])
    },
]

selected_map = MAPS[0]
MAP = selected_map["data"]

# ----------- Utilities ----------- #
def is_valid(cell):
    x, y = cell
    return 0 <= x < MAP.shape[0] and 0 <= y < MAP.shape[1] and MAP[x, y] == 0

def neighbors(pos):
    x, y = pos
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    return [(x+dx, y+dy) for dx, dy in dirs if is_valid((x+dx, y+dy))]

def angle_between(p1, p2, p3):
    v1 = np.array([p2[0]-p1[0], p2[1]-p1[1]])
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]])
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1)*np.linalg.norm(v2)
    if norm == 0: return 0
    angle = math.acos(np.clip(dot/norm, -1.0, 1.0))
    return math.degrees(angle)

# ----------- Graph Communication ----------- #
def send_graph_data(graph, current_node=None):
    try:
        pos = nx.get_node_attributes(graph, 'pos')
        nodes = [{"id": n, "pos": [pos[n][0], pos[n][1]]} for n in graph.nodes()]
        edges = [{"from": u, "to": v, "weight": graph[u][v]['weight']} 
                for u, v in graph.edges()]
        
        data = {
            "nodes": nodes,
            "edges": edges,
            "current_node": current_node
        }
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(('127.0.0.1', 9998))
            s.sendall(json.dumps(data).encode('utf-8'))
    except Exception as e:
        print(f"Failed to send graph data: {e}")


# ----------- Node & Graph Discovery ----------- #
class Explorer:
    def __init__(self, start):
        self.graph = nx.DiGraph()
        self.node_id_counter = itertools.count()
        self.node_positions = {}
        self.node_lookup = {}
        self.start = start
        self.add_node(start)

    def add_node(self, pos):
        node_id = f"N{next(self.node_id_counter)}"
        self.graph.add_node(node_id, pos=pos)
        self.node_positions[node_id] = pos
        self.node_lookup[pos] = node_id
        return node_id

    def explore(self):
        visited = set()
        frontier = [(self.start, [self.start])]

        while frontier:
            current, path = frontier.pop()
            if current in visited:
                continue
            visited.add(current)
            
            for direction in [(0,1), (1,0), (0,-1), (-1,0)]:
                new_path = [current]
                pos = current
                last_dir = direction
                steps = 0
                
                while True:
                    next_pos = (pos[0]+last_dir[0], pos[1]+last_dir[1])
                    if not is_valid(next_pos):
                        break
                    new_path.append(next_pos)
                    steps += 1
                    neighbors_count = len(neighbors(next_pos))
                    
                    if next_pos in self.node_lookup:
                        break
                    if neighbors_count >= 3 or steps > 6:
                        break
                    if len(new_path) >= 3:
                        ang = angle_between(new_path[-3], new_path[-2], new_path[-1])
                        if ang > 75:
                            break
                    pos = next_pos
                
                if len(new_path) > 1:
                    from_pos = new_path[0]
                    to_pos = new_path[-1]
                    from_id = self.node_lookup.get(from_pos) or self.add_node(from_pos)
                    to_id = self.node_lookup.get(to_pos) or self.add_node(to_pos)
                    cost = len(new_path)-1
                    self.graph.add_edge(from_id, to_id, weight=cost)
                    frontier.append((to_pos, [to_pos]))
                    
            # Send graph updates after each exploration step
            send_graph_data(self.graph, path[-1] if path else None)
            time.sleep(0.5)
            
        return self.graph

# ----------- FIFO Search ----------- #
def fifo_search(graph, start, goal=None, step_delay=0.5):
    open_list = deque()
    open_list.append([start])
    visited = {}
    edge_traversals = defaultdict(int)

    while open_list:
        path = open_list.popleft()
        current = path[-1]
        
        if current in visited:
            continue
            
        for i in range(len(path)-1):
            edge_traversals[(path[i], path[i+1])] += 1
            
        cost = sum(graph[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
        visited[current] = (cost, path)
        
        # Send update with current path
        send_graph_data(graph, current)
        print(f"Current: {current}, Path: {path}")
        
        if goal and current == goal:
            break
            
        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                open_list.append(path + [neighbor])
                
        time.sleep(step_delay)

    return visited, edge_traversals

def controlled_fifo_search(graph, start, step_delay=0.5):
    from time import sleep
    visited_nodes = set()
    visited_paths = {}
    edge_traversals = defaultdict(int)
    node_open = defaultdict(deque)   # مسیرهای قابل پیمایش از هر نود
    node_close = defaultdict(set)    # مسیرهای پیمایش‌شده از هر نود

    # شروع با نود ابتدایی
    open_paths = deque()
    open_paths.append([start])

    while open_paths:
        path = open_paths.popleft()
        current = path[-1]

        if current not in visited_nodes:
            visited_nodes.add(current)
            visited_paths[current] = (0, path)  # مقدار اولیه

        # فقط اگر مسیر جدید است (طول بیشتر از 1)، هزینه و تراورس را ثبت کن
        if len(path) > 1:
            cost = sum(graph[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
            visited_paths[path[-1]] = (cost, path)
            for i in range(len(path)-1):
                edge_traversals[(path[i], path[i+1])] += 1
                node_close[path[i]].add(path[i+1])  # ثبت این مسیر به عنوان پیمایش‌شده

        print(f"Current: {current}, Path: {path}")

        # اگر نود تازه است، openList آن را بساز
        if not node_open[current]:
            for neighbor in graph.neighbors(current):
                if neighbor not in visited_nodes:
                    node_open[current].append(neighbor)

        # حالا فقط یکی از مسیرهای باز این نود را دنبال کن
        while node_open[current]:
            neighbor = node_open[current].popleft()
            if neighbor not in visited_nodes:
                open_paths.append(path + [neighbor])
                break  # فقط یکی را جلو ببر

        sleep(step_delay)

    # پس از اتمام جست‌وجوی اصلی، مسیرهای باقی‌مانده را بررسی کن
    for node in graph.nodes():
        while node_open[node]:
            neighbor = node_open[node].popleft()
            if neighbor not in node_close[node]:
                open_paths.append([node, neighbor])

    # تکرار مسیرهای باقیمانده
    while open_paths:
        path = open_paths.popleft()
        current = path[-1]

        if current not in visited_nodes:
            visited_nodes.add(current)

        cost = sum(graph[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
        visited_paths[current] = (cost, path)

        for i in range(len(path)-1):
            edge_traversals[(path[i], path[i+1])] += 1
            node_close[path[i]].add(path[i+1])

        print(f"[Rerun] Current: {current}, Path: {path}")

        # افزودن مسیرهای جدید نود فعلی اگر وجود داشته باشد
        if not node_open[current]:
            for neighbor in graph.neighbors(current):
                if neighbor not in node_close[current]:
                    node_open[current].append(neighbor)

        while node_open[current]:
            neighbor = node_open[current].popleft()
            if neighbor not in node_close[current]:
                open_paths.append(path + [neighbor])
                break

        sleep(step_delay)

    return visited_paths, edge_traversals

# ----------- Main Execution ----------- #
if __name__ == "__main__":
    start_pos = (1,1)
    explorer = Explorer(start=start_pos)
    G = explorer.explore()
    
    # Perform search and visualize
    # visited_nodes, edge_counts = fifo_search(G, start='N0', goal=None, step_delay=0.5)
    visited_nodes, edge_counts = controlled_fifo_search(G, start='N0', step_delay=0.5)

    # Print stats
    print("\nEdge Traversal Count:")
    for (u, v), count in edge_counts.items():
        print(f"{u} → {v}: {count} times")
