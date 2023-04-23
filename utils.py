import networkx as nx 
import pickle 
import os
import pandas as pd
import random

from python.ci_tests import *
from python.discretize import *
from python.bnutils import *
from python.bnmetrics import *


def read_data(datapath, dataset, graph_num):
    """
        Read data, ground truth DAG, and the Call Graph from the appropriate directory
    """

    raw_data = pickle.load(open(os.path.join(datapath, dataset, 'Data.pkl'), 'rb'))
    dag_gt = nx.read_gpickle(os.path.join(datapath, dataset, 'DAG.gpickle'))
    data = pd.DataFrame()

    # Ensure all columns have same number of rows and remove the number of resources column (R)
    for k,v in raw_data.items():
        if 'R' in k:
            continue
        else:
            data[k] = v[1:]

    dag_cg = nx.read_gpickle(os.path.join(datapath, f'Graph{graph_num}.gpickle'))

    return data, dag_gt, dag_cg


def compute_stats(graph, graph_gt):

	AP = AR = AF = AHP = AHR = AHF = OP = OR = OF = FDR = shd = DE = UDE = 0

	DE += -len(list(graph.edges())) +  2*len(list(graph.to_undirected().edges()))
	UDE += len(list(graph.edges())) - len(list(graph.to_undirected().edges()))
	G = graph.copy()
	
	# Direct the undirected edges (topological sorting)
	for edge in list(G.edges):
	    if (edge[1],edge[0]) in G.edges:
	        G.remove_edge(edge[0], edge[1])
	        G.remove_edge(edge[1], edge[0])
	top_sort = list(nx.topological_sort(G))
	
	all_edges = []
	for edge in list(graph.edges):
	    if edge not in all_edges:
	        if (edge[1], edge[0]) in graph.edges:
	            index0 = top_sort.index(edge[0])
	            index1 = top_sort.index(edge[1])
	            if index0 < index1:
	                G.add_edge(edge[0], edge[1])
	            else:
	                G.add_edge(edge[1], edge[0])
	            all_edges.append((edge[1], edge[0]))
	    all_edges.append(edge)
	
	'''
	# Direct the undirected edges (randomly)
	for edge in list(G.edges):
	    if (edge[1],edge[0]) in G.edges:
	        a = random.choice([0,1])
	        if a:
	            try:
	                G.remove_edge(edge[0], edge[1])
	            except:
	                pass
	        else:
	            try:
	                G.remove_edge(edge[1], edge[0])
	            except:
	                pass
	'''
	
	adj_results = adj_stats(graph_gt.to_undirected(), G)
	arrow_head_results = arrow_stats(graph_gt, G)
	overall_results = overall_stats(graph_gt, G)

	AP += adj_results['precision']
	AR += adj_results['recall']

	if adj_results['precision'] or adj_results['recall']:
	    AF += (2 * adj_results['precision'] * adj_results['recall']) / (adj_results['precision'] + adj_results['recall'])

	AHP += arrow_head_results['precision']
	AHR += arrow_head_results['recall']

	if arrow_head_results['precision'] or arrow_head_results['recall']:
	    AHF += (2 * arrow_head_results['precision'] * arrow_head_results['recall']) / (arrow_head_results['precision'] + arrow_head_results['recall'])
	
	OP += overall_results['precision']
	OR += overall_results['recall']
	
	if overall_results['precision'] or overall_results['recall']:
	    OF += (2 * overall_results['precision'] * overall_results['recall']) / (overall_results['precision'] + overall_results['recall'])

	shd += SHD(graph_gt, G)


	print("STATISTICS")
	print("++++++++++++++")
	print(f"Structural Hamming Distance (SHD): {shd}\n")

	print(f"Adj. Precision: {round(AP, 3)}")
	print(f"Adj. Recall: {round(AR, 3)}")
	print(f"Adj. FRecall: {round(AF, 3)}\n")

	print(f"Arrow Head Precision: {round(AHP, 3)}")
	print(f"Arrow Head Recall: {round(AHR, 3)}")
	print(f"Arrow Head F1 Score: {round(AHF, 3)}\n")

	print(f"Overall Precision: {round(OP, 3)}")
	print(f"Overall Recall: {round(OR, 3)}")
	print(f"Overall F1 Score: {round(OF, 3)}\n")

	print(f"# Directed Edges: {DE}")
	print(f"# Undirected Edges: {UDE}")

