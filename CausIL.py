import networkx as nx
import pandas as pd
import pickle
import os
import itertools
from python.structure_learning.score_based.fges import fges
from libraries.FGES.runner import  fges_runner
from libraries.FGES.knowledge import Knowledge
from python.scores import *
from python.ci_tests import *
from python.discretize import *
from python.bnutils import *
from python.bnmetrics import *
import warnings
import time
import argparse
from utils import compute_stats, read_data
warnings.filterwarnings("ignore")



def runner(g, data, disc, n_bins, score, bl_edges = None):
    """Wrapper Runner for FGES that sets the prohibited edges into knowledge class
    
    Returns: 
        g   : result graph appended to input g
        dag : the result graph from the algo
    """

    nodes = []
    args = nx.DiGraph()
    mappers = column_mapping(data)
    data.rename(columns=mappers[0], inplace=True)
    args.add_nodes_from(list(data.columns))
    for col in data.columns:
        args.nodes[col]['type'] = 'cont'
        args.nodes[col]['num_categories'] = 'NA'
    

    if bl_edges:
        knowledge = Knowledge()
        for bl_edge in bl_edges:
            if bl_edge[0] in mappers[0].keys() and bl_edge[1] in mappers[0].keys():
                knowledge.set_forbidden(mappers[0][bl_edge[0]], mappers[0][bl_edge[1]])
    else:
        knowledge = None
    
    result = fges_runner(data, args.nodes(data=True), n_bins = n_bins, disc = disc, score = score, knowledge = knowledge)
    dag = nx.DiGraph()
    dag.add_nodes_from(args.nodes(data=True))
    dag.add_edges_from(result['graph'].edges())

    data.rename(columns = mappers[1], inplace = True)
    nx.relabel_nodes(dag, mappers[1], copy=False)
    
    g.add_nodes_from(dag.nodes)
    g.add_edges_from(dag.edges)

    return g, dag



def preprocess_data(data):
    updated_data = pd.DataFrame()
    
    mu = {}
    num_inst = []
    
    # for cols having instance level data, concatenate and flatten them -> pools the data
    for col in data.columns:
        if 'agg' in col:
            continue
        node_data = data[col].to_numpy()
        if len(num_inst) == 0:
            for i in range(len(node_data)):
                num_inst.append(len(node_data[i]))

        flatten = np.concatenate(node_data.tolist())
        updated_data[col] = flatten.ravel()
        
    # for cols having aggregate level data, repeat the data for number of instances
    for col in data.columns:
        if 'inst' in col:
            continue
        node_data = np.array(list(data[col])) 
        
        updated_data[col] = np.repeat(node_data, num_inst)
    
    return updated_data



def run_graph_discovery(data, dag_cg, datapath, dataset, dk, score_func):
    g = nx.DiGraph()
    service_graph = []
    fges_time = []

    # For each service, construct a graph individually and then merge them
    for i, service in enumerate(dag_cg.nodes):

        print('===============')
        print("Service: {}".format(service))
        serv_data = pd.DataFrame()

        # Get instance level data corresponding to the service only
        filtered_cols = [col for col in data.columns if (str(service) == col.split('_')[1] and 'inst' in col)]
        serv_data = serv_data.append(data[filtered_cols])


        # For the child services, get aggregate level data for latency and error
        child = [n[1] for n in dag_cg.out_edges(service)]
        print("Child Services:{}".format(child))
        agg_cols = []
        for col in data.columns:
            if int(col.split('_')[1]) in child and 'agg' in col:
                if 'L' in col or 'E' in col:
                    agg_cols.append(col)
        serv_data = pd.concat([serv_data, data[agg_cols]], axis=1)

        
        # For parent services, get aggregate level data for workload (aggregate worload = total workload)
        parent = [n[0] for n in dag_cg.in_edges(service)]
        print("Parent Services:{}".format(parent))
        agg_cols = []
        for col in data.columns:
            if int(col.split('_')[1]) in parent and 'agg' in col:
                if 'W' in col:
                    agg_cols.append(col)
        serv_data = pd.concat([serv_data, data[agg_cols]], axis=1)
        
        # Pool the data
        new_data = preprocess_data(serv_data)


        # Renaming is done based on the names that were present for ground truth graph
        # for example, W_0 is renamed to 0W, U_0 is renamed to 0MU, C_0 is renamed to 0CU, etc.
        rename_col = {}
        for col in new_data.columns:
            if 'U' in col:
                rename_col[col] = col.split('_')[1] + 'MU'
            elif 'C' in col:
                rename_col[col] = col.split('_')[1] + 'CU'  
            else:
                rename_col[col] = col.split('_')[1] + col.split('_')[0]
        new_data.rename(columns=rename_col, inplace = True)
    
    
        # Use domain knowledge or not
        if dk == 'N':
            bl_edges = None
        else:
            bl_edges = list(pd.read_csv(os.path.join(datapath, dataset, 'prohibit_edges.csv'))[['edge_source', 'edge_destination']].values)

        # Run FGES
        print('Starting FGES')

        if score_func == 'L':
            st_time = time.time()
            g, dag = runner(g, new_data, None, 1, linear_gaussian_score_iid, bl_edges)
            fges_time.append(time.time() - st_time)
        elif score_func == 'P2':
            st_time = time.time()
            g, dag = runner(g, new_data, None, 1, polynomial_2_gaussian_score_iid, bl_edges)
            fges_time.append(time.time() - st_time)
        elif score_func == 'P3':
            st_time = time.time()
            g, dag = runner(g, new_data, None, 1, polynomial_3_gaussian_score_iid, bl_edges)
            fges_time.append(time.time() - st_time)

        print('Finished FGES')

        service_graph.append(dag)
        print('\n')
        

    return g, service_graph, fges_time


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run CausIL')

    parser.add_argument('-D', '--dataset', required=True, help='Dataset type (synthetic/semi_synthetic)')
    parser.add_argument('-S', '--num_services', type=int, required=True, help='Numer of Services in the dataset (10, 20, etc.)')
    parser.add_argument('-G', '--graph_number', type=int, default=0, help='Graph Instance in the particular dataset [default: 0]')
    parser.add_argument('--dk', default='Y', help='To use domain knowledge or not (Y/N) [default: Y]')
    parser.add_argument('--score_func', default='P2', help='Which score function to use (L: linear, P2: polynomial of degree 2, P3: polynomial of degree 3) [default: P2]')
    
    args = parser.parse_args()

    if args.dataset != "synthetic" and args.dataset != "semi_synthetic":
        print("Incorrect Dataset provided!!...")
        print("======= EXIT ===========")
        exit()


    # Data set directory to use
    datapath = f'Data/{args.num_services}_services'
    dataset = f'{args.dataset}/Graph{args.graph_number}'

    data, dag_gt, dag_cg = read_data(datapath, dataset, args.graph_number)

    graph, service_graph, total_time = run_graph_discovery(data, dag_cg, datapath, dataset, args.dk, args.score_func)

    print(f"Total Time of Computation: {np.sum(total_time)}")

    compute_stats(graph, dag_gt)
