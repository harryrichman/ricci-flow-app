
import networkx as nx
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.animation as animation
from matplotlib.colors import TwoSlopeNorm

class RicciFlowWidget:
    def __init__(self, G, pos, epsilon=0.1):
        self.G = G
        self.L = nx.laplacian_matrix(G, weight="weight").toarray()
        self.pos = pos
        self.epsilon = epsilon
        self.num_nodes = G.number_of_nodes()

        self.edge_resistance = {}
        self.edge_curvature = {}
        self.node_weights = np.array(
            [1 for _ in G.nodes()]
        )
        self.edge_weights = np.array(
            [1 for _ in G.edges()]
        )
        self.iter_count = 1
        self.compute_resistance_curvature()

        self.ax = None
        self.drawn_nodes = None
        self.drawn_labels = None
        self.cmap = plt.cm.bwr

    def compute_resistance_curvature(self):
        """resistance matrix"""
        G = self.G
        n = self.num_nodes
        res_matrix = np.zeros((n, n))
        for u in G.nodes():
            for v in G.nodes():
                if u == v:
                    res_matrix[u][v] = 0
                else:
                    res_matrix[u][v] = nx.resistance_distance(
                        G, u, v, weight="length")
        self.Omega = np.array(
            res_matrix
        )
        self.two_tau = 1/2 * np.matmul(
            self.Omega, np.matmul(self.L, self.Omega))[0, 0]
        # "kappa" is node curvature
        self.kappa = np.matmul(
            np.linalg.inv(self.Omega), np.ones((n, 1)) * self.two_tau)
        # node_weights = "resistance curvature" after Derviendt and Lambiotte
        self.node_weights = self.kappa.flatten()
        # add resistance attribute to edges
        for e in G.edges():
            self.edge_resistance[e] = self.Omega[e[0], e[1]]
            self.edge_curvature[e] = (
                1 / G.degree[e[0]] + 1 / G.degree[e[1]] 
                - self.edge_resistance[e] / G.edges[e[0], e[1]]["length"]
            )
            # add curvature attribute to edges
            G.edges[e[0], e[1]]["resistance"] = self.Omega[e[0], e[1]]
            G.edges[e[0], e[1]]["curvature"] = self.edge_curvature[e]

    def ricci_flow_step(self):
        G = self.G
        # update lengths according to Ricci flow
        self.iter_count += 1
        print("button pressed; count=", self.iter_count)
        # compute gradient using curvature, update length 
        for e in G.edges():
            curv = G.edges[e[0], e[1]]["curvature"]
            new_length = G.edges[e[0], e[1]]["length"]
            new_length += - self.epsilon * curv
            if new_length < 0.01:
                new_length = 0.01
            # new_length += - self.epsilon * curv * length
            G.edges[e[0], e[1]]["length"] = new_length
            G.edges[e[0], e[1]]["weight"] = 1 / new_length
        self.compute_resistance_curvature()

    def animate(self):
        G = self.G
        fig, ax = plt.subplots()
        self.ax = ax
        fig.set_size_inches(8, 8)
        
        e_label_dict = {
            e: np.around(G.edges[e[0], e[1]]["length"], decimals=2) 
            for e in G.edges()
        }
        # Draw graph
        pos = nx.spring_layout(G, seed=1)
        self.drawn_nodes = nx.draw_networkx_nodes(
            G, pos, ax=self.ax, 
            node_color=[-self.node_weights[v] for v in G.nodes()], 
            cmap=self.cmap,
            vmin=-0.5,
            vmax=0.5)
        #draw edges and colored edges
        self.drawn_edges_color = nx.draw_networkx_edges(
            G, pos, ax=self.ax,
            width=6,
            edge_color=[-self.edge_curvature[e] for e in G.edges()],
            edge_cmap=self.cmap,
            edge_vmin=-0.5,
            edge_vmax=0.5,
        )
        self.drawn_edges = nx.draw_networkx_edges(
            G, pos, ax=self.ax,
        )
        # drawn_labels is a dict whose values are ax.text() objects
        # self.drawn_labels = nx.draw_networkx_labels(
        #     G, pos, ax=self.ax, labels=n_labels)
        # draw edge weights over edges
        self.edge_labels = nx.draw_networkx_edge_labels(
            G, pos, 
            ax=self.ax, 
            edge_labels=e_label_dict,
        )
        # show iteration count
        props = dict(alpha=0.5)
        self.count_box = ax.text(
            0.05, -0.07, 
            f"Iteration: {self.iter_count}", 
            transform=ax.transAxes, 
            fontsize=14,
            verticalalignment='top', 
            bbox=props)
        
        ani = animation.FuncAnimation(
            fig, self.animate_next, interval=30, save_count=30,
        )
        plt.show()
        

    def draw(self):
        G = self.G
        fig, ax = plt.subplots()
        self.ax = ax
        fig.set_size_inches(8, 8)
        fig.subplots_adjust(right=0.85)
        # Create a `matplotlib.widgets.Button` to increment
        ax_button = fig.add_axes([0.6, 0.025, 0.3, 0.04])
        self.button = Button(
            ax_button, 'Increment', hovercolor='0.975')
        # Make graph options
        n_labels = {
            v : str(np.around(self.node_weights[v], decimals=2)) + "\n\n" 
            for v in G.nodes()
        }
        e_label_dict = {
            e: np.around(G.edges[e[0], e[1]]["length"], decimals=2) 
            for e in G.edges()
        }
        # Draw graph
        pos = nx.spring_layout(G, seed=1)
        self.drawn_nodes = nx.draw_networkx_nodes(
            G, pos, ax=self.ax, 
            node_color=[-self.node_weights[v] for v in G.nodes()], 
            cmap=self.cmap,
            vmin=-0.5,
            vmax=0.5)
        #draw edges and colored edges
        self.drawn_edges_color = nx.draw_networkx_edges(
            G, pos, ax=self.ax,
            width=6,
            edge_color=[-self.edge_curvature[e] for e in G.edges()],
            edge_cmap=self.cmap,
            edge_vmin=-0.5,
            edge_vmax=0.5
        )
        self.drawn_edges = nx.draw_networkx_edges(
            G, pos, ax=self.ax,
        )
        # drawn_labels is a dict whose values are ax.text() objects
        # self.drawn_labels = nx.draw_networkx_labels(
        #     G, pos, ax=self.ax, labels=n_labels)
        # draw edge weights over edges
        self.edge_labels = nx.draw_networkx_edge_labels(
            G, pos, 
            ax=self.ax, 
            edge_labels=e_label_dict
        )
        # show iteration count
        props = dict(alpha=0.5)
        self.count_box = ax.text(
            0.05, -0.07, 
            f"Iteration: {self.iter_count}", 
            transform=ax.transAxes, 
            fontsize=14,
            verticalalignment='top', 
            bbox=props)

    def animate_next(self, i):
        self.on_press(None)
        self.draw_update()
        print("i=", i)
        print("self.iter_count=", self.iter_count)

        time.sleep(0.2)

    def draw_update(self):
        G = self.G
        # update labels to reflect changed length
        n_label_dict = {
            v : np.around(self.node_weights[v], decimals=2)
            for v in G.nodes()
        }
        # for v in G.nodes():
        #     t = self.drawn_labels[v]
        #     t.set_text(str(n_label_dict[v]) + "\n\n" )
        e_label_dict = {
            e: np.around(G.edges[e[0], e[1]]["length"], decimals=2) 
            for e in G.edges()
        }
        for e in G.edges():
            t = self.edge_labels[e]
            t.set_text(str(e_label_dict[e]))
        # update layout
        pos = nx.spring_layout(G, seed=1)
        self.drawn_nodes.set_offsets(
            [list(x) for x in pos.values()]
        )
        # for v in G.nodes():
        #     t = self.drawn_labels[v]
        #     t.set_position(pos[v])
        self.drawn_edges_color.set_segments(
            [[pos[e[0]], pos[e[1]]] for e in G.edges()]
        )
        self.drawn_edges.set_segments(
            [[pos[e[0]], pos[e[1]]] for e in G.edges()]
        )
        for e in G.edges():
            t = self.edge_labels[e]
            x1, y1 = pos[e[0]]
            x2, y2 = pos[e[1]]
            avg_pos = [0.5 * (x1 + x2), 0.5 * (y1 + y2)]
            t.set_position(avg_pos)
            # `angle` computation taken from networkx source code
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            t.set_rotation(angle)       
        # nx.draw_networkx_edges(G, pos, ax=self.ax)
        # drawn_labels is a dict whose values are ax.text() objects
        '''update node colors''' 
        
        self.count_box.set_text(f"Iteration: {self.iter_count}")
        self.ax.set_xlim(
            -0.2 + min(pos[v][0] for v in G.nodes()), 
            0.2 + max(pos[v][0] for v in G.nodes())
        )
        self.ax.set_ylim(
            -0.2 + min(pos[v][1] for v in G.nodes()), 
            0.2 + max(pos[v][1] for v in G.nodes())
        )
        self.ax.figure.canvas.draw()

    def connect(self):
        """Connect to all the events we need."""
        self.cidbutton = self.button.on_clicked(self.on_press)

    def on_press_animate(self):
        self.on_press(None)
    
    def on_press(self, event):
        # update lengths according to Ricci flow
        self.ricci_flow_step()
        # update labels to reflect changed curvature
        self.draw_update()

    def disconnect(self):
        """Disconnect all callbacks."""

if __name__ ==  "__main__":

    # G = nx.path_graph(6)
    # G = nx.Graph(); G.add_edges_from([(0,1),(1,2),(1,3),(3,4),(3,5),(5,6)])
    G = nx.Graph(); G.add_edges_from(
        [
            (0,1),(1,2),(0,3),(3,2),(1,3),(0,4),(4,2),(4,9),
            (5,6),(6,7),(5,8),(8,7),(6,8),(5,9),(9,7)
        ]
    )
    # G = nx.cycle_graph(7)
    # G = nx.grid_graph([2,5])

    # G = nx.bull_graph()
    # G = nx.frucht_graph()
    # G = nx.barbell_graph(4, 0)
    # G = nx.diamond_graph()
    # G = nx.karate_club_graph()
    
    # change node names to integers
    mapping = {e : i for i, e in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    for (i, e) in enumerate(G.edges()):
        G.edges[e[0], e[1]]["length"] = 1 + i % 1
        G.edges[e[0], e[1]]["weight"] = 1.0 / (1 + i % 1)
    pos = nx.spring_layout(G, seed=932861)

    widget = RicciFlowWidget(G, pos, epsilon=0.4)
    # widget.draw()
    # widget.connect()
    # plt.show()

    widget.animate()

