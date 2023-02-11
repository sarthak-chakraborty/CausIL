import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm
import math
import networkx as nx
import argparse


EXOG_PATH = 'path/to/exogneous/workload'
SRC = 'Data'

def read_regressors():
    PATH = 'rf_models/'
    r1 = pickle.load(open(PATH + 'f1.sav', 'rb'))
    r2 = pickle.load(open(PATH + 'f2.sav', 'rb'))
    r3 = pickle.load(open(PATH + 'f3.sav', 'rb'))
    r4 = pickle.load(open(PATH + 'f4.sav', 'rb'))
    r5 = pickle.load(open(PATH + 'f5.sav', 'rb'))
    
    return r1, r2, r3, r4, r5

def read_call_graph(path):
    return nx.read_gpickle(path)

# Predict number of resources given workload
def f1(workload, regressors):
    return regressors[0].predict(workload.reshape(-1,1))[0]

# Predict CPU Usage
def f2(workload, regressors):
    return regressors[1].predict(workload.reshape(-1,1))

# Predict Memory Usage
def f3(workload, cpuUsage, regressors):
    feat = np.stack([cpuUsage, workload], axis=1)
    return regressors[2].predict(feat.reshape(-1,2))

# Predict Error
def f4(cpuUsage, memUsage, regressors):
    feat = np.stack([cpuUsage, memUsage], axis=1)
    return regressors[3].predict(feat.reshape(-1,2))

# Predict Latency
def f5(cpuUsage, memUsage, error, regressors):
    feat = np.stack([cpuUsage, memUsage, error], axis=1)
    return regressors[4].predict(feat.reshape(-1,3))



def initialize_data(G):
    services = list(G.nodes())
    
    data = dict()
    
    for n in services:
        data[f'R_{n}_agg'], data[f'W_{n}_agg'], data[f'L_{n}_agg'], data[f'E_{n}_agg'] = [], [], [], []
        data[f'W_{n}_inst'], data[f'U_{n}_inst'], data[f'C_{n}_inst'], data[f'L_{n}_inst'], data[f'E_{n}_inst'] = [], [], [], [], []
    
    return data


def data_one_pass(G, top_sort, current_index, exog_services, workload_distr, exog, data, t, lag, std_metrics, regressors):
    if current_index == len(top_sort):
        return data
    
    curr_node = top_sort[current_index]
    if curr_node in exog_services:
        # Workload for exogenous service is from exogenous data
        workload = None
        while True:
            workload = exog[t] + np.random.normal(loc=0, scale=std_metrics[0])
            if workload > 0:
                break
        data[f'W_{curr_node}_agg'].append(workload)
    else:
        # Compute workload for other services
        pred = list(G.predecessors(curr_node))
        w = 0
        for j in pred:
            w += workload_distr[(j,curr_node)] * data[f'W_{j}_agg'][t]
        data[f'W_{curr_node}_agg'].append(w)
        
        
    data[f'R_{curr_node}_agg'].append( math.ceil(f1(data[f'W_{curr_node}_agg'][t-lag], regressors)) )
    
    instance_workload = np.random.normal(
                                        loc=data[f'W_{curr_node}_agg'][t]/data[f'R_{curr_node}_agg'][t],
                                        scale=0.1*data[f'W_{curr_node}_agg'][t]/data[f'R_{curr_node}_agg'][t],
                                        size=data[f'R_{curr_node}_agg'][t]-1
                                        )
    instance_workload = np.append(instance_workload, data[f'W_{curr_node}_agg'][t] - sum(instance_workload))
    data[f'W_{curr_node}_inst'].append(instance_workload)
    

    cpuUtil = f2(data[f'W_{curr_node}_inst'][t], regressors) + np.random.normal(0, 0.1*std_metrics[1], data[f'R_{curr_node}_agg'][t])
    data[f'C_{curr_node}_inst'].append(np.clip(cpuUtil, 0, 100))
    
    memUtil = f3(data[f'W_{curr_node}_inst'][t], data[f'C_{curr_node}_inst'][t], regressors) + np.random.normal(0, 0.1*std_metrics[2], data[f'R_{curr_node}_agg'][t])
    data[f'U_{curr_node}_inst'].append(np.clip(memUtil, 0, 100))

    # Call data_one_pass on the next node
    data = data_one_pass(G, top_sort, current_index+1, exog_services, workload_distr, exog, data, t, lag, std_metrics, regressors)
    
    # Compute Latency and Error for leaf nodes, and return
    if len(list(G.successors(curr_node))) < 1:
        error = f4(data[f'C_{curr_node}_inst'][t], data[f'U_{curr_node}_inst'][t], regressors) + np.random.normal(loc=0, scale=std_metrics[3], size=data[f'R_{curr_node}_agg'][t])
        data[f'E_{curr_node}_inst'].append(np.clip(error, 0, max(error)))
        data[f'E_{curr_node}_agg'].append(np.mean(data[f'E_{curr_node}_inst'][t]))
        
        latency = f5(data[f'C_{curr_node}_inst'][t], data[f'U_{curr_node}_inst'][t], data[f'E_{curr_node}_inst'][t], regressors) + np.random.normal(loc=0, scale=0.1*std_metrics[4], size=data[f'R_{curr_node}_agg'][t])
        data[f'L_{curr_node}_inst'].append(np.clip(latency, 0, max(latency)))
        data[f'L_{curr_node}_agg'].append(np.mean(data[f'L_{curr_node}_inst'][t]))
        
        return data
    
    # Compute Latency and Error from all the children
    succ = list(G.successors(curr_node))
    E_ext, L_ext = 0, 0
    for j in succ:
        E_ext += workload_distr[(curr_node,j)] * data[f'E_{j}_agg'][t]
        L_ext += workload_distr[(curr_node,j)] * data[f'L_{j}_agg'][t]
        
    data[f'E_{curr_node}_inst'].append(f4(data[f'C_{curr_node}_inst'][t], data[f'U_{curr_node}_inst'][t], regressors) + np.random.normal(loc=E_ext, scale=0.1*E_ext, size=data[f'R_{curr_node}_agg'][t]))
    data[f'E_{curr_node}_agg'].append(np.mean(data[f'E_{curr_node}_inst'][t]))
    
    data[f'L_{curr_node}_inst'].append(f5(data[f'C_{curr_node}_inst'][t], data[f'U_{curr_node}_inst'][t], data[f'E_{curr_node}_inst'][t], regressors) + np.random.normal(loc=L_ext, scale=0.1*L_ext, size=data[f'R_{curr_node}_agg'][t]))
    data[f'L_{curr_node}_agg'].append(np.mean(data[f'L_{curr_node}_inst'][t]))
        
    return data


def generate_data(data, G, exog, lag, std_metrics, regressors):
    exog_services = [n for n,d in G.in_degree() if d==0] 
    n_services = G.number_of_nodes()
    
    top_sort = list(nx.topological_sort(G))
    
    workload_distr = dict()
    for e in G.edges():
        workload_distr[e] = np.random.uniform(0,1)
    
    
    for t in range(lag):
        for i in top_sort:
            if i in exog_services:
                workload = None
                while True:
                    workload = exog[t] + np.random.normal(loc=0, scale=std_metrics[0])
                    if workload > 0:
                        break
                data[f'W_{i}_agg'].append(workload)
            else:
                pred = list(G.predecessors(i))
                w = 0
                for j in pred:
                    w += workload_distr[(j, i)] * data[f'W_{j}_agg'][-1]
                data[f'W_{i}_agg'].append(w)
            data[f'R_{i}_agg'].append(0)
            data[f'L_{i}_agg'].append(0)
            data[f'E_{i}_agg'].append(0)
            data[f'W_{i}_inst'].append(0)
            data[f'U_{i}_inst'].append(0)
            data[f'C_{i}_inst'].append(0)
            data[f'L_{i}_inst'].append(0)
            data[f'E_{i}_inst'].append(0)
            
    for t in tqdm(range(lag, 3000)):
        data = data_one_pass(G, top_sort, 0, exog_services, workload_distr, exog, data, t, lag, std_metrics, regressors)
        
    return data
            

def main():

    parser = argparse.ArgumentParser(description='Generate Synthetic Data')

    parser.add_argument('-N', '--nodes', type=int, required=True, help='Number of services in service graph')
    parser.add_argument('-L', '--lag', type=int, default=1, help='Lag for workload to affect number of resources [default: 1]')
    parser.add_argument('--path_exog', default=EXOG_PATH, help='Path to exogneous workload')
    
    args = parser.parse_args()

    RUNS = 10
    df = pd.read_csv(args.path_exog)
    df = df.dropna()
	
    mean_metrics = [df['cpuUsage'].mean(), df['memoryUsage'].mean(), df['errorCount'].mean(), df['latency'].mean()]
    std_metrics = df[['timestamp', 'workload', 'cpuUsage', 'memoryUsage', 'errorCount', 'latency']].astype(float).groupby('timestamp').agg(np.std).mean()
	
    exog = df[['timestamp', 'workload']].astype(float).groupby('timestamp').agg(sum).reset_index(drop=True)
    
    regressors = read_regressors()

    write_dir = os.path.join(SRC, f'{args.nodes}_services')

    for i in range(RUNS):
        print(i, end='\r')

        G = read_call_graph(path)

        data = initialize_data(G)

        data = generate_data(data, G, exog['workload'].dropna().values, lag, std_metrics, regressors)

        DIR = os.path.join(wrie_dir, 'semi_synthetic', 'Graph{}'.format(i))
        if not os.path.isdir(DIR):
            os.mkdir(DIR)
            
        filename = os.path.join(DIR, 'Data.pkl')
        pickle.dump(data, open(filename, 'wb'))


if __name__ == '__main__':
    main()
