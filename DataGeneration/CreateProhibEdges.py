import networkx as nx
import pandas as pd
import numpy as np
import os
import argparse


def get_complete_graph(DAG_inst, DAG_cg, prohib_edge_list):
    prohibit_edge = pd.DataFrame()
    DAG_comp = nx.DiGraph()
    
    # Create Metric Graph for Each Service and Prohibit edge within Each Service
    print("Creating Metric Graph for each service")
    for i in range(len(DAG_cg.nodes())):
        DAGi = DAG_inst.copy()
        DAGi = nx.relabel_nodes(DAGi, dict(zip(DAGi.nodes, ['{}{}'.format(i, nod) for nod in DAG_inst.nodes])), copy = False)
        DAG_comp.add_nodes_from(DAGi.nodes)
        DAG_comp.add_edges_from(DAGi.edges)
        prohibit_edge = pd.concat([prohibit_edge, pd.DataFrame([('{}{}'.format(i, edge[0]), '{}{}'.format(i, edge[1])) for edge in prohib_edge_list])], axis=0, ignore_index=True)
    
    # Creating Edges Across Services
    print("Create Edges across Services")
    prohib = set()
    for edge in DAG_cg.edges:     
        DAG_comp.add_edges_from([('{}W'.format(edge[0]), '{}W'.format(edge[1]))])
        DAG_comp.add_edges_from([('{}L'.format(edge[1]), '{}L'.format(edge[0]))])
        DAG_comp.add_edges_from([('{}E'.format(edge[1]), '{}E'.format(edge[0]))])
        
        # prohibit edges across services within the call graph
        all_edges = set()
        for i in node_labels: #[W, CU, MU, L, E]
            for j in node_labels: #[W, CU, MU, L, E]
                all_edges.add(('{}{}'.format(edge[0], i), '{}{}'.format(edge[1], j)))
                all_edges.add(('{}{}'.format(edge[1], i), '{}{}'.format(edge[0], j)))
        # print(edge, len(all_edges))
                
        
        subset_edge = {('{}{}'.format(edge[0], 'W'), '{}{}'.format(edge[1], 'W')), 
                        ('{}{}'.format(edge[1], 'L'), '{}{}'.format(edge[0], 'L')),
                        ('{}{}'.format(edge[1], 'E'), '{}{}'.format(edge[0], 'L')),
                        ('{}{}'.format(edge[1], 'E'), '{}{}'.format(edge[0], 'E'))}
                
    
        prohib = prohib.union(all_edges - subset_edge)
                         
    prohibit_edge = pd.concat([prohibit_edge, pd.DataFrame(list(prohib))], axis=0, ignore_index=True)
        
    #prohibit all edges from services not within call graph
    print("Prohibit all edges for services not within call graph")
    for i in DAG_cg.nodes:
        for j in DAG_cg.nodes:
            if i != j:
                if (i,j) not in DAG_cg.edges and (j,i) not in DAG_cg.edges:
                    for metric1 in node_labels:
                        for metric2 in node_labels:
                            prohibit_edge = pd.concat([prohibit_edge, pd.DataFrame([('{}{}'.format(i, metric1), '{}{}'.format(j, metric2))])], axis=0, ignore_index=True)
 
    prohibit_edge.columns = ['edge_source','edge_destination']
    return DAG_comp, prohibit_edge


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Generate Prohibited Edge List')

    parser.add_argument('-N', '--nodes', required=True, type=int, help='Number of nodes/services (10/20/40)')
    
    args = parser.parse_args()

    num_graphs = 10
    node_labels = ['W', 'CU', 'MU', 'L', 'E']
    prohib_edge_list = [('L','W'), ('CU','W'), ('E','W'), ('MU','W'), ('L','CU'), ('L','MU')]

    Binary_matrix = np.array([[0,1,1,0,0],[0,0,1,1,1],[0,0,0,1,1],[0,0,0,0,0],[0,0,0,1,0]])
    DAG_inst = nx.DiGraph(Binary_matrix)    # Causal Metric Graph for each instance
    nx.relabel_nodes(DAG_inst, dict(zip(range(len(DAG_inst.nodes)), node_labels)), copy = False)


    for i in range(num_graphs):
        print('Graph', i)
        DAG_cg = nx.read_gpickle(f'Data/{args.nodes}_services/Graph{i}.gpickle')
        # DAG_cg = nx.read_gpickle(os.path.join(os.getcwd(), f'RealData/Graph{i}.gpickle'))
        DAG, prohibit_edge = get_complete_graph(DAG_inst, DAG_cg, prohib_edge_list)

        path = f'Data/{args.nodes}_services'

        for types in ['synthetic', 'semi_synthetic']:
            DIR_PATH = os.path.join(path, types, f'Graph{i}')

            if not os.path.isdir(DIR_PATH):
                os.makedirs(DIR_PATH)

            nx.write_gpickle(DAG, os.path.join(DIR_PATH, 'DAG.gpickle'))
            prohibit_edge.to_csv(os.path.join(DIR_PATH, 'prohibit_edges.csv'))
            print('')




