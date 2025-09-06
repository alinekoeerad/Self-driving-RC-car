# import networkx as nx
# import matplotlib.pyplot as plt
# import cv2
# import numpy as np
# from cv2 import aruco

# # -------------------------------------
# # Step 1: Define your custom map here
# # -------------------------------------
# def create_ground_truth_map():
#     G = nx.Graph()
#     rows = 4
#     cols = 6

#     # اضافه کردن نودها
#     for row in range(rows):
#         for col in range(cols):
#             idx = row * 6 + col
#             G.add_node(f'N{idx}')

#     # اضافه کردن یال‌ها به چپ، راست، بالا، پایین
#     for row in range(rows):
#         for col in range(cols):
#             idx = row * 6 + col
#             current = f'N{idx}'

#             # Right
#             if col < cols - 1:
#                 right = f'N{row * 6 + (col + 1)}'
#                 G.add_edge(current, right)

#             # Down
#             if row < rows - 1:
#                 down = f'N{(row + 1) * 6 + col}'
#                 G.add_edge(current, down)

#     return G

# # -------------------------------------
# # Step 2: Define grid positions (6x5)
# # -------------------------------------
# def get_node_positions():
#     pos = {}
#     for row in range(4):
#         for col in range(6):
#             idx = row * 6 + col
#             scale = 1.0
#             pos[f'N{idx}'] = (col * scale, (4 - row) * scale)
#     return pos

# # -------------------------------------
# # Step 3: Generate ArUco Markers
# # -------------------------------------
# def generate_aruco_markers(ids, marker_size=100):
#     aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
#     markers = {}
#     for id in ids:
#         marker_img = np.zeros((marker_size, marker_size), dtype=np.uint8)
#         aruco.generateImageMarker(aruco_dict, id, marker_size, marker_img, 1)
#         markers[f'N{id}'] = marker_img
#     return markers

# # -------------------------------------
# # Step 4: Plot graph with ArUco markers
# # -------------------------------------
# def plot_graph_with_aruco(G, pos, markers):
#     fig, ax = plt.subplots(figsize=(15, 15))

#     # Draw edges
#     for u, v in G.edges():
#         x1, y1 = pos[u]
#         x2, y2 = pos[v]
#         ax.plot([x1, x2], [y1, y2], 'black', linewidth=4, zorder=1)

#     # Draw ArUco markers at nodes
#     for node, (x, y) in pos.items():
#         marker = markers.get(node)
#         if marker is not None:
#             marker_rgb = cv2.cvtColor(marker, cv2.COLOR_GRAY2RGB) / 255.0
#             ax.imshow(marker_rgb, extent=(x-0.3, x+0.3, y-0.3, y+0.3), zorder=2)
#             # ax.text(x, y - 0.4, node, ha='center', va='top', fontsize=8, zorder=3)

#     ax.set_xlim(-1, 6)
#     ax.set_ylim(-1, 5)
#     ax.set_aspect('equal')
#     plt.axis('off')
#     plt.title("Ground Truth Map with ArUco Nodes", fontsize=14)
#     plt.tight_layout()
#     plt.show()

# def print_edges_as_code(G):
#     edges = sorted(G.edges(), key=lambda x: (int(x[0][1:]), int(x[1][1:])))
#     line = "    "
#     count = 0
#     for u, v in edges:
#         line += f"('{u}', '{v}'), "
#         count += 1
#         if count % 4 == 0:  # break every 4 edges per line
#             print(line)
#             line = "    "
#     if line.strip():  # print any remaining edges
#         print(line)


# # -------------------------------------
# # Main Runner
# # -------------------------------------
# if __name__ == '__main__':
#     G = create_ground_truth_map()
#     pos = get_node_positions()
#     markers = generate_aruco_markers(range(30))  # N0 to N29
#     plot_graph_with_aruco(G, pos, markers)
#     print("Edge list in code format:")
#     print_edges_as_code(G)


# import networkx as nx
# import matplotlib.pyplot as plt
# from graphviz import Digraph
# import numpy as np

# grid = np.array([
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 9, 9, 0, 9, 0, 9, 1, 1, 1, 9, 1, 9, 1, 1, 1, 1, 1, 1],
#     [1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1],
#     [1, 1, 9, 0, 9, 0, 9, 0, 9, 0, 9, 0, 9, 1, 1, 1, 1, 1, 1],
#     [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
#     [1, 1, 9, 0, 9, 0, 9, 1, 9, 0, 0, 0, 9, 1, 1, 1, 1, 1, 1],
#     [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
#     [1, 1, 0, 1, 9, 0, 9, 0, 9, 0, 9, 1, 9, 0, 9, 1, 1, 1, 1],
#     [1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1],
#     [1, 1, 9, 0, 9, 0, 9, 0, 0, 0, 0, 0, 9, 0, 9, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
#     [1, 1, 1, 1, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
# ])

# rows, cols = grid.shape
# G = nx.DiGraph()
# node_positions = {}

# # کمک‌کننده برای بررسی حرکت‌های قائم (بالا، پایین، چپ، راست)
# directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# # پیدا کردن نودها
# for r in range(rows):
#     for c in range(cols):
#         if grid[r][c] == 9:
#             node_name = f"N{r}_{c}"
#             G.add_node(node_name)
#             node_positions[(r, c)] = node_name

# # بررسی مسیر بین نودها فقط در مسیرهای قائم
# for (r1, c1), node1 in node_positions.items():
#     for dr, dc in directions:
#         r, c = r1 + dr, c1 + dc
#         path = []
#         while 0 <= r < rows and 0 <= c < cols:
#             if grid[r][c] == 1:
#                 break
#             elif grid[r][c] == 9:
#                 node2 = node_positions[(r, c)]
#                 G.add_edge(node1, node2)
#                 break
#             else:
#                 path.append((r, c))
#             r += dr
#             c += dc

# # نمایش با matplotlib (برای تست در محیط‌های آفلاین)
# plt.figure(figsize=(6, 4))
# pos = {v: (c, -r) for (r, c), v in node_positions.items()}
# nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1200, arrows=True)
# plt.title("Graph from Grid Map")
# plt.axis("off")
# plt.show()

# # ساخت گراف Graphviz
# dot = Digraph(format='png')
# for node in G.nodes:
#     dot.node(node)

# for u, v in G.edges:
#     dot.edge(u, v)  
# print(G.edges)

# # ذخیره گراف به فایل (یا نمایش مستقیم در GraphvizOnline)
# dot_code = dot.source
# print(dot_code)


import matplotlib.pyplot as plt
import cv2
import numpy as np
import networkx as nx
from cv2 import aruco

# -------------------------------------
# Step 1: Define your custom map as input
# -------------------------------------
grid = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 9, 0, 9, 1, 9, 0, 0, 0, 0, 0, 9, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 9, 0, 9, 0, 9, 1, 9, 0, 9, 1],
    [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 9, 0, 0, 0, 9, 0, 9, 1, 0, 1, 0, 1], # Central hub area
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],
    [1, 9, 0, 0, 0, 9, 0, 0, 0, 9, 0, 9, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 9, 1, 1, 9, 0, 0, 0, 0, 9, 0, 9, 1], # Long corridor
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ])

# -------------------------------------
# Step 2: Create graph from the grid
# -------------------------------------
def create_ground_truth_map_from_grid(grid):
    """
    گراف را بر اساس ماتریس ورودی ایجاد می‌کند.
    مقدار 9 به عنوان نود و مسیرهای بدون مانع (غیر 1) به عنوان یال در نظر گرفته می‌شوند.
    """
    G = nx.Graph()
    rows, cols = grid.shape
    
    # پیدا کردن تمام نودها (جایی که مقدار 9 است) و اختصاص ID به آنها
    node_coords = {}
    node_id_counter = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 9:
                node_name = f'N{node_id_counter}'
                G.add_node(node_name)
                node_coords[node_name] = (r, c)
                node_id_counter += 1

    coord_to_node = {v: k for k, v in node_coords.items()}

    # پیدا کردن یال‌ها بین نودهای همسایه
    for node_name, (r1, c1) in node_coords.items():
        # جستجوی همسایه‌های افقی (به سمت راست)
        for c2 in range(c1 + 1, cols):
            if all(grid[r1, c] != 1 for c in range(c1 + 1, c2)):
                if (r1, c2) in coord_to_node:
                    neighbor_name = coord_to_node[(r1, c2)]
                    G.add_edge(node_name, neighbor_name)
                    break 
            else:
                break # در صورت وجود دیوار، جستجو متوقف می‌شود

        # جستجوی همسایه‌های عمودی (به سمت پایین)
        for r2 in range(r1 + 1, rows):
            if all(grid[r, c1] != 1 for r in range(r1 + 1, r2)):
                if (r2, c1) in coord_to_node:
                    neighbor_name = coord_to_node[(r2, c1)]
                    G.add_edge(node_name, neighbor_name)
                    break
            else:
                break # در صورت وجود دیوار، جستجو متوقف می‌شود
                
    return G, node_coords

# -------------------------------------
# Step 3: Get node positions for plotting
# -------------------------------------
def get_node_positions_from_grid(node_coords, grid_shape):
    """
    مختصات نودها را برای رسم در نمودار محاسبه می‌کند.
    """
    pos = {}
    max_row = grid_shape[0] - 1
    scale = 1.0 
    for node_name, (r, c) in node_coords.items():
        # محور y برای رسم معکوس می‌شود
        pos[node_name] = (c * scale, (max_row - r) * scale)
    return pos

# -------------------------------------
# Step 4: Generate ArUco Markers
# -------------------------------------
def generate_aruco_markers(ids, marker_size=100):
    """
    برای هر ID یک مارکر ArUco تولید می‌کند.
    """
    # اطمینان از اینکه تعداد مارکرها از ظرفیت دیکشنری بیشتر نشود
    if max(ids, default=-1) >= 50:
        print("Warning: One or more IDs are out of range for DICT_4X4_50 (0-49).")

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    markers = {}
    for id_val in ids:
        node_name = f'N{id_val}'
        marker_img = np.zeros((marker_size, marker_size), dtype=np.uint8)
        aruco.generateImageMarker(aruco_dict, id_val, marker_size, marker_img, 1)
        markers[node_name] = marker_img
    return markers

# -------------------------------------
# Step 5: Plot graph with ArUco markers
# -------------------------------------
def plot_graph_with_aruco(G, pos, markers, grid_shape):
    """
    گراف نهایی را با مارکرهای ArUco در محل نودها رسم می‌کند.
    """
    fig, ax = plt.subplots(figsize=(12, 12))

    # رسم یال‌ها
    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], 'royalblue', linewidth=3, zorder=1)

    # رسم مارکرهای ArUco در محل نودها
    for node, (x, y) in pos.items():
        marker = markers.get(node)
        if marker is not None:
            marker_rgb = cv2.cvtColor(marker, cv2.COLOR_GRAY2RGB) / 255.0
            ax.imshow(marker_rgb, extent=(x-0.4, x+0.4, y-0.4, y+0.4), zorder=2)
            ax.text(x, y - 0.6, node, ha='center', va='top', fontsize=9, color='black', zorder=3)

    # تنظیم محدوده نمودار به صورت داینامیک
    rows, cols = grid_shape
    ax.set_xlim(-1, cols)
    ax.set_ylim(-1, rows)
    
    ax.set_aspect('equal')
    plt.axis('off')
    plt.title("Ground Truth Map with ArUco Nodes (from Grid)", fontsize=16)
    plt.tight_layout()
    plt.show()

def print_edges_as_code(G):
    """
    لیست یال‌های گراف را برای استفاده در کد چاپ می‌کند.
    """
    edges = sorted(G.edges(), key=lambda x: (int(x[0][1:]), int(x[1][1:])))
    line = "    "
    count = 0
    for u, v in edges:
        line += f"('{u}', '{v}'), "
        count += 1
        if count % 4 == 0:
            print(line)
            line = "    "
    if line.strip():
        print(line)


# -------------------------------------
# Main Runner
# -------------------------------------
if __name__ == '__main__':
    # ایجاد گراف و مختصات نودها از روی ماتریس ورودی
    G, node_coords = create_ground_truth_map_from_grid(grid)
    
    # محاسبه موقعیت نودها برای رسم
    pos = get_node_positions_from_grid(node_coords, grid.shape)
    
    # تولید مارکرهای ArUco بر اساس تعداد نودهای پیدا شده
    num_nodes = len(node_coords)
    marker_ids = range(num_nodes)
    markers = generate_aruco_markers(marker_ids)
    
    # رسم گراف نهایی
    plot_graph_with_aruco(G, pos, markers, grid.shape)
    
    # چاپ اطلاعات گراف
    print(f"✅ Graph created successfully with {len(G.nodes())} nodes and {len(G.edges())} edges.")
    print("\nEdge list in code format:")
    print_edges_as_code(G)