import networkx as nx
import numpy as np
import pandas as pd
import argparse

SEED = 10


def uniform_weight(rng):
    segments = [(-2.0, -0.5), (0.5, 2.0)]
    low, high = rng.choice(segments)
    return rng.uniform(low=low, high=high)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate Ground Truth Service Graph')

    parser.add_argument('-N', '--nodes', required=True, type=int, help='Number of nodes')
    parser.add_argument('-E', '--edges', type=int, required=True, help='Number of edges')
    
    args = parser.parse_args()

    n_graphs = 10

    for i in range(n_graphs):
        rng = np.random.default_rng()
        
        num_edge = min(max(args.edges, args.nodes - 1), int(args.nodes * (args.nodes - 1) / 2))

        matrix = np.zeros((args.nodes, args.nodes))

        # Make the graph connected
        for cause in range(1, args.nodes):
            result = rng.integers(low=0, high=cause)
            matrix[result, cause] = uniform_weight(rng)
        num_edge -= args.nodes - 1

        while num_edge > 0:
            cause = rng.integers(low=1, high=args.nodes)
            result = rng.integers(low=0, high=cause)
            if not matrix[result, cause]:
                matrix[result, cause] = uniform_weight(rng)
                num_edge -= 1


        G = nx.DiGraph(matrix)

        nx.write_gpickle(G, f'../Data/{args.nodes}_services/Graph{i}.gpickle')
