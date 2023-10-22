
from mpl_ricci_flow import *

# G = nx.path_graph(6)
G = nx.Graph(); G.add_edges_from([(0,1),(1,2),(1,3),(3,4),(3,5),(5,6)])
# G = nx.cycle_graph(7)
# G = nx.grid_graph([4,5])

# G = nx.bull_graph()
# G = nx.frucht_graph()

# change node names to integers
mapping = {e : i for i, e in enumerate(G.nodes())}
G = nx.relabel_nodes(G, mapping)
for (i, e) in enumerate(G.edges()):
    G.edges[e[0], e[1]]["length"] = 1 + i % 1
    G.edges[e[0], e[1]]["weight"] = 1.0 / (1 + i % 1)
pos = nx.spring_layout(G, seed=932861)

widget = RicciFlowWidget(G, pos, epsilon=0.2)
# widget.draw()
# widget.connect()
# plt.show()

widget.animate()
