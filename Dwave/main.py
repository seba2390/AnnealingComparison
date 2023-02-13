from collections import defaultdict

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

import matplotlib.pyplot as plt

import networkx as nx

import numpy as np

# Generating a graph of 4 nodes

n = 4  # Number of nodes in graph
G = nx.Graph()
G.add_nodes_from(np.arange(0, n, 1))

weights   = [1.0, 0.8, 0.4, 0.6, 0.7]
edge_list = [(0, 1, weights[0]),
             (0, 2, weights[1]),
             (0, 3, weights[2]),
             (1, 2, weights[3]),
             (2, 3, weights[4])]

# tuple is (i,j,weight) where (i,j) is the edge
G.add_weighted_edges_from(edge_list)

colors = ["r" for node in G.nodes()]
pos = nx.spring_layout(G)


def draw_graph(G, colors, pos):
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=0.8, ax=default_axes, pos=pos)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)


DISPLAY_GRAPH = True
if DISPLAY_GRAPH:
    draw_graph(G, colors, pos)
    plt.show()


def generate_Q()
# ------- Set up our QUBO dictionary -------

# Initialize our Q matrix
Q = defaultdict(int)


# ------- Loading MSA QUBO model -------
my_strings   = np.array(["AGT","A"])
my_penalties = np.array([1,1,1])*100
my_msa       = MultipleSequenceAlignment(strings = my_strings, penalties = my_penalties, normalize_weights=True)
msa_Q,h,d    = my_msa.QUBO_model


# Update Q dictionary
for i in range(msa_Q.shape[0]):
    for j in range(msa_Q.shape[1]):
        if i == j:
            Q[(i,i)]+= h[i]       ## Linear terms from h
            Q[(j,i)]+= msa_Q[j,i] ## Diagonal quadratic terms from Q
        else:
            Q[(i,j)]+= msa_Q[i,j] ## Off-diagonal quadratic terms from Q

# ------- Run our QUBO on the QPU -------
# Set up QPU parameters
chainstrength = 80
numruns = 10

# Run the QUBO on the solver from your config file
sampler = EmbeddingComposite(DWaveSampler())
response = sampler.sample_qubo(Q,
                               chain_strength=chainstrength,
                               num_reads=numruns,
                               label='Example - Maximum Cut')
# ------- Print results to user -------
print('-' * 13)
print("Initial MSA:")
print('-' * 13)
print(f"{my_msa.initial_MSA}\n")
print('-' * 35)
print("with corresponding binary encoding:")
print('-' * 35)
init_state = "|"
for nr in my_msa.initial_bitstring:
    init_state += str(int(nr))
init_state += ">"
print(f"{init_state}\n")
print('-' * 17)
print("Using penalities:")
print('-' * 17)
print(f"p1,p2,p3 = {my_penalties[0],my_penalties[1],my_penalties[2]}\n")
print("#"*8+" ANNEALING RESULTS "+"#"*8)
print('-' * int(len(init_state)+16))
print(" "*int(len(init_state)/2-3)+"State:"+" "*int(len(init_state)/2-3)+" "*7+"Energy:")
print('-' * int(len(init_state)+16))
for sample, E in response.data(fields=['sample','energy']):
    state = "|"
    for k,v in sample.items():
        state += str(v)
    state += ">"
    print(str(state)+" "*7+str(E))
print('-' * 55)
print("With lowest energy state corresponding to alignement:")
print('-' * 55)
best_state = []
for val in list(response.first.sample.values()):
    best_state.append(val)
best_alignment = my_msa.bit_state_2_matrix(np.array(best_state))
print(best_alignment)