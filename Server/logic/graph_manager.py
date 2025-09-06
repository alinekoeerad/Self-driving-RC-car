# logic/graph_manager.py

import networkx as nx

class GraphManager:
    """A wrapper for the networkx graph to manage the robot's map."""

    def __init__(self):
        """Initializes the graph."""
        # We use a simple Graph, as edges are bidirectional for pathfinding.
        self.graph = nx.Graph()

    def add_node(self, node_id: str, **attrs):
        """Adds a node to the graph if it doesn't already exist."""
        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, **attrs)

    def add_edge(self, node1_id: str, node2_id: str, cost: int, **attrs):
        """Adds or updates a weighted edge between two nodes."""
        # The 'weight' attribute is standard for networkx pathfinding algorithms.
        self.graph.add_edge(node1_id, node2_id, weight=cost, **attrs)
    
    def update_node_attributes(self, node_id: str, **attrs):
        """Updates the attributes of a node (e.g., exit angles)."""
        if self.graph.has_node(node_id):
            # Iterates through the provided attributes and sets them on the node.
            for key, value in attrs.items():
                self.graph.nodes[node_id][key] = value

    def set_edge_attribute(self, node1_id: str, node2_id: str, attr_name: str, attr_value: any):
        """Sets a custom attribute on an edge."""
        # Checks if the edge exists before trying to set an attribute.
        if self.graph.has_edge(node1_id, node2_id):
            self.graph.edges[node1_id, node2_id][attr_name] = attr_value

    def get_path_astar(self, start_node: str, end_node: str) -> list:
        """Calculates the shortest path using A*, considering edge weights."""
        try:
            # Uses the built-in A* pathfinding algorithm from networkx.
            return nx.astar_path(self.graph, start_node, end_node, weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # If no path exists or a node is not in the graph, return an empty list.
            return []

    def get_neighbors(self, node_id: str) -> list:
        """Returns a list of neighbors for a given node."""
        if self.graph.has_node(node_id):
            return list(self.graph.neighbors(node_id))
        return []
    
    def get_all_frontiers(self, visited_nodes: set) -> list[tuple[str, str]]:
        """Finds all edges that lead from a visited node to an unvisited one."""
        frontiers = []
        for u in visited_nodes:
            # For each visited node, check its neighbors.
            for v in self.get_neighbors(u):
                # If a neighbor has not been visited, it's a frontier.
                if v not in visited_nodes:
                    frontiers.append((u, v))
        return frontiers
        
        # Always return a valid dictionary, even if the graph is empty.
        return {"nodes": nodes, "edges": edges}

    def get_node_attribute(self, node_id: str, attr_name: str):
        """Gets a specific attribute from a node if it exists."""
        if self.graph.has_node(node_id):
            # .get() is used to avoid errors if the attribute doesn't exist yet
            return self.graph.nodes[node_id].get(attr_name)
        return None
    
    def clear(self):
        """Clears the entire graph, removing all nodes and edges."""
        self.graph.clear()
        print("📈 Graph has been cleared.")
# In logic/graph_manager.py

    def get_graph_data_for_client(self) -> dict:
        """Formats the graph data into a JSON-serializable dictionary for the C# client."""
        # --- FIX: The dictionary key is changed from 'position' to 'pos' ---
        # This now matches the [JsonProperty("pos")] attribute in the C# PythonNode class.
        nodes = [
            {'id': n, 'pos': self.graph.nodes[n].get('position', [0, 0])} 
            for n in self.graph.nodes()
        ]
        # --- END FIX ---
        
        edges = [
            {'from': u, 'to': v} 
            for u, v in self.graph.edges()
        ]
        
        return {"nodes": nodes, "edges": edges}
    