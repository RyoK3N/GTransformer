import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

class Skeleton:
    def __init__(self, connections: List[Tuple[str, str]], 
                 joints_left: List[str], 
                 joints_right: List[str], 
                 ordered_joint_names: List[str]):
        self._initialize_attributes(connections, ordered_joint_names)
        self._initialize_joint_sides(joints_left, joints_right)
        self._compute_metadata()
        self._initialize_graph()

    def _initialize_attributes(self, connections: List[Tuple[str, str]], ordered_joint_names: List[str]):
        self._connections = connections
        self._joint_names = ordered_joint_names
        self._joint_indices = {joint: idx for idx, joint in enumerate(self._joint_names)}
        self._parents = self._compute_parents()

    def _initialize_joint_sides(self, joints_left: List[str], joints_right: List[str]):
        self._joints_left = [self._joint_indices[joint] for joint in joints_left if joint in self._joint_indices]
        self._joints_right = [self._joint_indices[joint] for joint in joints_right if joint in self._joint_indices]

    def _initialize_graph(self):
        self.Graph = nx.Graph()
        self.Graph.add_edges_from(self._connections)
        for joint_name in self._joint_names:
            self.Graph.add_node(joint_name, position=self._joint_indices[joint_name])

    @property
    def connections(self) -> List[Tuple[str, str]]:
        return self._connections

    @property
    def joint_names(self) -> List[str]:
        return self._joint_names

    @property
    def joint_indices(self) -> Dict[str, int]:
        return self._joint_indices

    def _compute_parents(self) -> np.ndarray:
        parents = [-1] * len(self._joint_names)
        for child, parent in self._connections:
            if child in self._joint_indices and parent in self._joint_indices:
                child_idx = self._joint_indices[child]
                parent_idx = self._joint_indices[parent]
                parents[child_idx] = parent_idx
        return np.array(parents)

    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents), dtype=bool)
        self._children = [[] for _ in range(len(self._parents))]
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True
                self._children[parent].append(i)

    def get_connection_indices(self) -> List[Tuple[int, int]]:
        connections_idx = []
        for child, parent in self._connections:
            if child in self._joint_indices and parent in self._joint_indices:
                child_idx = self._joint_indices[child]
                parent_idx = self._joint_indices[parent]
                connections_idx.append((child_idx, parent_idx))
        return connections_idx

    def get_list_coords_from_graph(self, keypoints: np.ndarray) -> List[Tuple[float, float]]:
        keypoints_reshaped = keypoints.reshape(-1, 2)
        return [(np.float32(x), np.float32(y)) for x, y in keypoints_reshaped]

    def get_labels_from_graph(self) -> List[str]:
        return self._joint_names

    def get_idx_from_graph(self) -> List[int]:
        return list(range(len(self._joint_names)))

    def plot_graph(self):
        self.plot_skeleton()

    def plot_skeleton(self, figsize: Tuple[int, int] = (10, 10)):
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(self.Graph)
        nx.draw(self.Graph, pos,
                with_labels=True,
                node_color='lightblue',
                node_size=1000,
                font_size=8,
                font_weight='bold',
                edge_color='gray')
        plt.title("Skeleton Structure", pad=20)
        plt.show()

    def plot_graph_with_keypoints(self, keypoints: np.ndarray, 
                                figsize: Tuple[int, int] = (15, 15), 
                                point_size: int = 100):
        keypoints = self._prepare_keypoints(keypoints)
        fig, ax = plt.subplots(figsize=figsize)
        G, pos = self._create_visualization_graph(keypoints)
        self._draw_skeleton_elements(G, pos, point_size, ax)
        self._set_plot_properties(keypoints, ax)
        return fig  

    def _prepare_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        return keypoints.reshape(-1, 2) if len(keypoints.shape) == 1 else keypoints

    def _create_visualization_graph(self, keypoints: np.ndarray) -> Tuple[nx.Graph, Dict]:
        G = nx.Graph()
        pos = {}
        for i, (x, y) in enumerate(keypoints):
            joint_name = self._joint_names[i]
            G.add_node(joint_name)
            pos[joint_name] = (x, y)
        for child, parent in self._connections:
            if child in pos and parent in pos:
                G.add_edge(child, parent)
        return G, pos

    def _draw_skeleton_elements(self, G: nx.Graph, pos: Dict, point_size: int, ax: plt.Axes):
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5, ax=ax)
        node_colors = self._get_node_colors(G)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=point_size, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    def _get_node_colors(self, G: nx.Graph) -> List[str]:
        return ['lightblue' if self._joint_indices[node] in self._joints_left
                else 'lightpink' if self._joint_indices[node] in self._joints_right
                else 'lightgreen' for node in G.nodes()]

    def _set_plot_properties(self, keypoints: np.ndarray, ax: plt.Axes):
        margin = 0.1
        ax.set_xlim(keypoints[:, 0].min() - margin, keypoints[:, 0].max() + margin)
        ax.set_ylim(keypoints[:, 1].min() - margin, keypoints[:, 1].max() + margin)
        ax.set_aspect('equal')
        ax.set_title("Skeleton with Keypoints", pad=20)

