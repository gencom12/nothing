from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

class Graph: 

    def __init__(self): 

        self.graph = defaultdict(list)
        self.edges = []
        self.vertices = []

    def addEdge(self,a:float,b:float,heuristicA:float,heuristicB:float,cost:float): 
        self.graph[b].append(a)
        self.graph[a].append(b)
        self.edges.append({"edge":(a,b),
                            "weight":cost})
        vertices = []
        for i in self.vertices:
            vertices.append(i["label"])
        if a not in vertices:
            self.vertices.append({"label":a, "heuristic":heuristicA})
        if b not in vertices:
            self.vertices.append({"label":b, "heuristic":heuristicB})
        self.visited=[] 

    def a_star(self, start, goal):
        open_set = {start}
        came_from = {}
        g_score = {vertex["label"]: float('inf') for vertex in self.vertices}
        f_score = {vertex["label"]: float('inf') for vertex in self.vertices}

        g_score[start] = 0
        f_score[start] = self.get_heuristic(start)

        while open_set:
            current = min(open_set, key=lambda node: f_score[node])

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            open_set.remove(current)

            for neighbor in self.graph[current]:
                tentative_g_score = g_score[current] + self.get_edge_weight(current, neighbor)
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.get_heuristic(neighbor)
                    if neighbor not in open_set:
                        open_set.add(neighbor)

        return None

    def get_heuristic(self, node):
        for vertex in self.vertices:
            if vertex["label"] == node:
                return vertex["heuristic"]
        return 0  # Default heuristic if not found

    def get_edge_weight(self, node1, node2):
        for edge in self.edges:
            if edge["edge"] == (node1, node2) or edge["edge"] == (node2, node1):
                return edge["weight"]
        return 0  # Default weight if not found

    def visualize(self, path:list = []):
        G = nx.Graph(dict(self.graph))

        pos = nx.spring_layout(G)  # Positions for all nodes

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=500)

        # Draw edges
        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=1.0, alpha=0.5)

        # Highlight edges in the given path
        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=path_edges,
            width=2.0,
            alpha=1.0,
            edge_color="red",
        )

        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_color="black", font_weight="bold")

        plt.axis("off")
        plt.title("Graph Representation")
        plt.show()

def test():
    g = Graph() 
    g.addEdge('S','A',6,4,2)
    g.addEdge('S','B',6,4,3)

    g.addEdge('A','C',4,4,3)
    g.addEdge('B','D',4,3.5,3)
    g.addEdge('B','C',4,4,1)

    g.addEdge('C','E',4,1,3)
    g.addEdge('C','D',4,3.5,1)

    g.addEdge('D','F',3.5,1,2)


    g.addEdge('E','G',1,0,2)
    g.addEdge('F','G',1,0,1)

    
    # print(g.display())
    g.visualize()

    shortest_path = g.a_star('S', 'G')
    if shortest_path:
        shortest_path.insert(0,'S')
        print("Shortest Path:", shortest_path)
        g.visualize(shortest_path)
        total_cost = sum(g.get_edge_weight(shortest_path[i], shortest_path[i+1]) for i in range(len(shortest_path) - 1))
        print("Total Cost:", total_cost)
    else:
        print("No path found")

if __name__ == '__main__':
    test()