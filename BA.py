import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

Ns = [10**2, 10**3, 10**4, 10**5]
ms = [3, 9, 27, 81]
repeats = 10
method = 'PA'

# define Barabasi-Albert network class
class BA():
    def __init__(self):
        
        # create output directory if it doesn't exist
        self.dir = 'output/'
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def init_graph(self, m, N):
        # create initial graph
        G = nx.Graph()

        #initial graph
        self.N_0 = m+1
        G.add_nodes_from(np.arange(1,self.N_0+1))
        G.add_edges_from([[i, j] for i in np.arange(1,self.N_0) for j in np.arange(i+1,self.N_0+1)]) # start with fully connected graph

        # get the initial flattened edges array
        flat_edges = np.zeros((N*m+len(G.edges()))*2).astype(int)
        flat_edges[:len(G.edges())*2] = np.array(G.edges()).flatten()
        self.E_0 = len(G.edges())

        return G, flat_edges

    def plot(self, G):
        # plot the graph
        nx.draw(G)
        plt.show()

    def addNode(self, G, m, flat_edges, method='PA'):
        '''
        Add a new node to the graph and connect it to m other nodes
        '''
        new_node = nx.number_of_nodes(G)+1

        G.add_node(new_node) # add a new node

        G, flat_edges = self.addEdges(G, new_node, m, flat_edges, method=method) # add m edges to the new node

        return G, flat_edges

        
    
    def addEdges(self, G, source, m, flat_edges, method='PA'):

        if method == 'PA': # preferential attachment
            source_ids = np.repeat(source, m) # m-long array with source node

            # choose m random nodes with probability based on the degree (PA)
            target_ids = []
            choices = flat_edges[flat_edges > 0]
            while len(target_ids) < m:
                choice = np.random.choice(choices)
                #remove the choice from the list
                choices = choices[choices != choice]
                target_ids.append(choice)


        if method == 'RA': # random attachment
            source_ids = np.repeat(source, m) # m-long array with source node
            target_ids = np.random.choice(np.arange(1,source), m, replace=False)

        if method == 'EV': # existing vertices
            r = m//3 # int division
            source_ids = np.repeat(source, m) # m-long array with source node
            target_ids = np.zeros(m).astype(int)
            # choose r random target nodes
            target_ids[:r] = np.random.choice(np.arange(1,source), r, replace=False)

            for i in range(m-r):
                #choose two unique random nodes (PA) source and target
                flat_edges_nonzero = flat_edges[flat_edges > 0]

                #choose two unique random nodes from flat_edges
                choices = flat_edges_nonzero
                source_id = np.random.choice(flat_edges_nonzero)
                choices = choices[choices != source_id] #remove the choice from the list
                target_id = np.random.choice(choices)

                #check if the edge already exists, if so, choose new nodes
                while [source_id, target_id] in G.edges() or [target_id, source_id] in G.edges():
                    #choose two unique random nodes from flat_edges
                    choices = flat_edges_nonzero
                    source_id = np.random.choice(choices)
                    #remove the choice from the list
                    choices = choices[choices != source_id] 
                    target_id = np.random.choice(choices)

                #add the edge
                source_ids[r+i] = source_id
                target_ids[r+i] = target_id

        # add the edges
        edges_to_add = np.array([source_ids, target_ids]).T
        G.add_edges_from(edges_to_add)

        i_0 = self.E_0*2
        i_start = (source-1-self.N_0)*m*2+i_0

        flat_edges[i_start:i_start+m] = source_ids
        flat_edges[i_start+m:i_start+m*2] = target_ids
        
        return G, flat_edges
            
    def animate(self, G, init=False):
        if init:
            self.fig, self.ax = plt.subplots()
            plt.ion()
            plt.show()

        self.ax.clear() # update the plot
        pos = nx.spring_layout(G, k=0.1)
        nx.draw(G, pos, node_size=100, with_labels=True)
        plt.pause(10)

    def save(self, G, filename):
        # save the graph
        nx.write_pajek(G, self.dir + filename + '.net')
        

    def run(self, Ns, ms, repeats, method='PA'):
        #run all combinations of N and m
        for m in ms:
            for N in Ns:
                for run_id in range(repeats):
                    print('{}: N = {}, m = {}, run = {}/{}'.format(method, N, m, run_id+1, repeats))
                    G, flat_edges = self.init_graph(m,N) # initialise the graph
                    #self.animate(G, init=True) # initialise the animation

                    for i in tqdm(range(N), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
                        G, flat_edges = self.addNode(G, m, flat_edges, method=method)
                        #self.animate(G)

                    self.save(G, method+'_N{}_m{}_{}'.format(N, m, run_id))

if __name__ == '__main__':
    BA = BA()

    BA.run(Ns=Ns, ms=ms, repeats=repeats, method=method)