# Contributors: Siddharth Jain, Aayush Makharia, Vineet Malik, Sourav Suman, Ayush Chauhan, Gaurav Sinha
# Owned by: Adobe Corporation

# Implements bayesian network scoring functions

from math import lgamma, log
from python.bnutils import conditional_gaussian_likelihood
import pandas as pd
import networkx as nx
import numpy as np
import statsmodels.api as sm
import datetime
from statsmodels.tools.eval_measures import bic as BIC
from statsmodels.tools.tools import pinv_extended
from statsmodels.gam.api import GLMGam, BSplines
from pygobnilp.scoring import ContinuousData, BGe
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model




def bdeu_score(data, dag, local=False, equivalent_sample_size=10, **kwargs):
    """Calculates the bdeu score for given graph and data.
    Args:
        dag: the graph for which score is to be calculated
        data: the pandas dataframe
        state_names: A dict indicating, for each variable, the discrete set of states (or values) that the variable can take.
        equivalent_sample_size: The equivalent/imaginary sample size (of uniform pseudo samples) for the dirichlet hyperparameters. The score is sensitive to this value, runs with different values might be useful.
    Returns:
        the bdeu score
    """
    def _collect_state_names(variable):
        "Return a list of states that the variable takes in the data"
        states = sorted(list(data.loc[:, variable].dropna().unique()))
        return states

    def state_counts(variable, parents):
        """counts how often each state of 'variable' occured, conditional on parents' states"""
        if not parents:
            # count how often each state of 'variable' occured
            state_count_data = data.loc[:, variable].value_counts()
            return state_count_data.reindex(state_names[variable]).fillna(0).to_frame()
        else:
            parents_states = [state_names[parent] for parent in parents]
            # count how often each state of 'variable' occured, conditional on parents' states
            state_count_data = (data.groupby([variable] + parents).size().unstack(parents))
            if not isinstance(state_count_data.columns, pd.MultiIndex):
                state_count_data.columns = pd.MultiIndex.from_arrays([state_count_data.columns])
            return state_count_data.reindex(index=state_names[variable], columns=pd.MultiIndex.from_product(parents_states, names=parents)).fillna(0)

    def local_score(variable, parents=[]):
        """computes local score for the given variable"""
        var_states = state_names[variable]
        var_cardinality = len(var_states)
        state_counts_var = state_counts(variable, parents)
        num_parents_states = float(len(state_counts_var.columns))
        local_score = 0
        # iterate over df columns (only 1 if no parents)
        for parents_state in state_counts_var:
            conditional_sample_size = sum(state_counts_var[parents_state])
            local_score += lgamma(equivalent_sample_size / num_parents_states) - lgamma(conditional_sample_size + equivalent_sample_size / num_parents_states)
            for state in var_states:
                if state_counts_var[parents_state][state] > 0:
                    local_score += lgamma(state_counts_var[parents_state][state] + equivalent_sample_size/(num_parents_states * var_cardinality)) - lgamma(equivalent_sample_size/(num_parents_states * var_cardinality))
        return local_score

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    if "state_names" not in kwargs:
        state_names = dict()
        variables = list(data.columns.values)
        for var in variables:
            if var in state_names:
                if not set(_collect_state_names(var)) <= set(state_names[var]):
                    raise ValueError("Data contains unexpected states for variable: {var}.")
                state_names[var] = state_names[var]
            else:
                state_names[var] = _collect_state_names(var)
    else:
        state_names = kwargs.get("state_names")

    if local:
        if "node" not in kwargs or "parents" not in kwargs:
            raise Exception("Must specify a node and its parents for calculating local score")
        node = kwargs.get("node")
        parents = kwargs.get("parents")
        return local_score(node, parents)

    return sum([local_score(variable, list(dag.predecessors(variable))) for variable in data.columns.values.tolist()])


def conditional_gaussian_score(df, dag, local=False, sparsity=1, **kwargs):
    score = 0
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    def conditional_gaussian_local_score(node, parents):
        parents_disc = [i for i in parents if dag.nodes[i]['type'] == 'disc']
        parents_cont = [i for i in parents if dag.nodes[i]['type'] == 'cont']
        levels = [dag.nodes[node]['num_categories'] for node in parents_disc]
        log_lik, dof = conditional_gaussian_likelihood(df, node, parents_disc, parents_cont, levels, dag.nodes(data=True))
        return 2*log_lik - sparsity*dof*np.log(df.shape[0])

    if local:
        if "node" not in kwargs or "parents" not in kwargs:
            raise Exception("Must specify a node and its parents for calculating local score")
        node = kwargs.get("node")
        parents = kwargs.get("parents")
        return conditional_gaussian_local_score(node, parents)

    nodes = list(dag.nodes())

    for node in nodes:
        parents = list(dag.predecessors(node))
        temp = conditional_gaussian_local_score(node, parents)
        score += temp
    return score


score_dict_global = dict()
is_calculated = False
dot_matrix = None
col_map = {}

def linear_gaussian_score_iid_old(df, dag, penalty=2, local=False, **kwargs):
    score = 0
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    def linear_gaussian_local_score_iid(node, parents):
        parents_data = df[parents].to_numpy()
        parents_data = sm.add_constant(parents_data, prepend = False)
        node_data = df[node].to_numpy()
        result = sm.OLS(node_data, parents_data).fit()
        return 2*result.llf - penalty*(parents_data.shape[1]+1)*np.log(parents_data.shape[0])
#         return -result.bic

    if local:
        if "node" not in kwargs or "parents" not in kwargs:
            raise Exception("Must specify a node and its parents for calculating local score")
        node = kwargs.get("node")
        parents = kwargs.get("parents")
        return linear_gaussian_local_score_iid(node, parents)

    nodes = list(dag.nodes())

    for node in nodes:
        parents = list(dag.predecessors(node))
        temp = linear_gaussian_local_score_iid(node, parents)
        score += temp
    return score


def polynomial_gaussian_score_iid(df, dag, penalty=2, local=False, **kwargs):
    score = 0
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    def polynomial_gaussian_local_score_iid(node, parents):
        parents_data = df[parents].to_numpy()
        if parents_data.shape[1] == 0:
            parents_data = sm.add_constant(parents_data, prepend = False)
        else:
            polynomial_features= PolynomialFeatures(degree=2)
            parents_data = polynomial_features.fit_transform(parents_data)
            
        node_data = df[node].to_numpy()
        
        result = linear_model.BayesianRidge(compute_score=True).fit(parents_data, node_data)
        return 2*result.scores_[-1] - penalty*(parents_data.shape[1]+1)*np.log(parents_data.shape[0])
#         result = sm.OLS(node_data, parents_data).fit()
#         return 2*result.llf - penalty*(parents_data.shape[1]+1)*np.log(parents_data.shape[0])
#         return -result.bic

    if local:
        if "node" not in kwargs or "parents" not in kwargs:
            raise Exception("Must specify a node and its parents for calculating local score")
        node = kwargs.get("node")
        parents = kwargs.get("parents")
        return polynomial_gaussian_local_score_iid(node, parents)

    nodes = list(dag.nodes())

    for node in nodes:
        parents = list(dag.predecessors(node))
        temp = polynomial_gaussian_local_score_iid(node, parents)
        score += temp
    return score



def linear_gaussian_score_iid(df, dag, local=False, penalty=2, **kwargs):
    score = 0
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    def linear_gaussian_local_score_iid(node, parents, penalty):
        global is_calculated
        global dot_matrix
        global score_dict_global
        
        if not is_calculated:
            matrix = df.to_numpy()
            matrix = sm.add_constant(matrix, has_constant='add', prepend=False)
            dot_matrix = np.dot(matrix.T, matrix)
            is_calculated = True

        parent_node_tup = tuple([tuple(parents), node])
        if parent_node_tup in score_dict_global:
            return score_dict_global[parent_node_tup]
        else:
            parents_data = df[parents].to_numpy()
            parents_data = sm.add_constant(parents_data, has_constant='add', prepend = False)
            node_data = df[node].to_numpy()
            

            # Removing nans from data: Wherever nan is encountered, whole node_data and parent_data for that element is deleted.
            new_parent = np.empty((0,parents_data.shape[1]))
            new_node = np.array([])
            for item in range(len(node_data)):
                arr = node_data[item]
                if(~np.isnan(arr)):
                    new_parent = np.append(new_parent,np.array([parents_data[item]]),axis=0)
                    new_node = np.append(new_node,arr)

            parents1 = np.empty((0,new_parent.shape[1]))
            nodes1 = np.array([])        
            for item in range(len(new_parent)):
                arr = new_parent[item]
                arr_hasnan = np.sum(arr)
                if(~np.isnan(arr_hasnan)):
                    parents1 = np.append(parents1,np.array([arr]),axis=0)
                    nodes1 = np.append(nodes1,new_node[item])
                                
                        
            sub_dot_matrix = np.empty(shape=(len(parents)+1, len(parents)+1))
            for i in range(len(parents)+1):
                for j in range(len(parents)+1):
                    if j >= len(parents) and i >= len(parents):
                        sub_dot_matrix[i][j] = dot_matrix[len(dot_matrix)-1][len(dot_matrix)-1]
                    elif j >= len(parents):
                        sub_dot_matrix[i][j] = dot_matrix[parents[i]][len(dot_matrix)-1]
                    elif i >= len(parents):
                        sub_dot_matrix[i][j] = dot_matrix[len(dot_matrix)-1][parents[j]]
                    else:
                        sub_dot_matrix[i][j] = dot_matrix[parents[i]][parents[j]]
                    
                    if np.isnan(sub_dot_matrix[i][j]):
                        sub_dot_matrix[i][j] = np.dot(parents1[:,i].T, parents1[:,j])

            sub_res_matrix = np.empty(shape=(len(parents)+1, 1))
            for i in range(len(parents)+1):
                if i >= len(parents):
                    sub_res_matrix[i][0] = dot_matrix[len(dot_matrix)-1][node]
                else:
                    sub_res_matrix[i][0] = dot_matrix[parents[i]][node]
                
                if np.isnan(sub_res_matrix[i][0]):
                    sub_res_matrix[i][0] = np.dot(parents1[:,i].T, nodes1)         

                    
            inv,_ = pinv_extended(sub_dot_matrix)
            beta = np.dot(inv, sub_res_matrix)
            
            model = sm.OLS(nodes1, parents1)
            llf = model.loglike(params=beta.reshape(-1))
            bic = 2 * llf - penalty * (beta.shape[0]) * np.log(len(nodes1))
            
#             # BIC = n*log(residual sum of squares/n) + k*log(n)
#             residual = nodes1.reshape(nodes1.shape[0], 1) - np.dot(parents1, beta)
#             rss = np.sum(np.square(residual).reshape(-1))
#             n = len(nodes1)
#             bic = (beta.shape[0]) * np.log(n) + n * np.log(rss/n)
            
            score_dict_global[parent_node_tup] = bic
            return bic

    if local:
        if "node" not in kwargs or "parents" not in kwargs:
            raise Exception("Must specify a node and its parents for calculating local score")
        node = kwargs.get("node")
        parents = kwargs.get("parents")
        return linear_gaussian_local_score_iid(node, parents, penalty)

    nodes = list(dag.nodes())

    for node in nodes:
        parents = list(dag.predecessors(node))
        temp = linear_gaussian_local_score_iid(node, parents, penalty)
        score += temp
    return score


'''
def bic_distributed(df, dag, local=False, penalty=2, **kwargs):
    score = 0
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    def bic_distributed_local(node, parents, penalty):  
        
        if len(np.array(list(df[node])).shape) == 1:
            parents_df = pd.DataFrame()
            for parent in parents:
                if len(np.array(list(df[parent])).shape) == 1:
                    parents_df[str(parent)] = df[parent]
                else:
                    parent_data = np.array(list(df[parent]))
                    for col_id in range(parent_data.shape[1]):
                        parents_df[str(parent)+'_'+str(col_id)] = parent_data[:,col_id]
            node_df = pd.DataFrame(df[node].copy())
            for col in parents_df.columns:
                node_df = node_df[~parents_df[col].isna()]
                parents_df = parents_df[~parents_df[col].isna()]
            parents_df = parents_df[~node_df[node_df.columns[0]].isna()]
            node_df = node_df[~node_df[node_df.columns[0]].isna()]
            if parents_df.shape[0] == 0:
                return float('-inf')
            
            parents_data = sm.add_constant(parents_df.to_numpy(), prepend = False)
            node_data = node_df.to_numpy()
            result = sm.OLS(node_data, parents_data).fit()
            return 2*result.llf - penalty*(parents_df.shape[1]+1)*np.log(parents_df.shape[0])
        
        else:
            nodes_data = np.array(list(df[node]))
            bic = 0
            count = 0
            for col_id in range(nodes_data.shape[1]):
                node_df = pd.DataFrame({col_id:nodes_data[:,col_id]})
                parents_df = pd.DataFrame()
                for parent in parents:
                    if len(np.array(list(df[parent])).shape) == 1:
                        parents_df[str(parent)] = df[parent]
                    else:
                        parents_df[str(parent)+'_'+str(col_id)] = np.array(list(df[parent]))[:,col_id]
                for col in parents_df.columns:
                    node_df = node_df[~parents_df[col].isna()]
                    parents_df = parents_df[~parents_df[col].isna()]
                parents_df = parents_df[~node_df[node_df.columns[0]].isna()]
                node_df = node_df[~node_df[node_df.columns[0]].isna()]
                if not parents_df.shape[0] == 0:
                    parents_data = sm.add_constant(parents_df.to_numpy(), prepend = False)
                    node_data = node_df.to_numpy()
                    result = sm.OLS(node_data, parents_data).fit()
                    bic += 2*result.llf - penalty*(parents_df.shape[1]+1)*np.log(parents_df.shape[0])
                    count += 1
            if count == 0:
                return float('-inf')
            return bic/count

    if local:
        if "node" not in kwargs or "parents" not in kwargs:
            raise Exception("Must specify a node and its parents for calculating local score")
        node = kwargs.get("node")
        parents = kwargs.get("parents")
        return bic_distributed_local(node, parents, penalty)

    nodes = list(dag.nodes())

    for node in nodes:
        parents = list(dag.predecessors(node))
        temp = bic_distributed_local(node, parents, penalty)
        score += temp
    return score
'''


def bic_distributed(df, dag, local=False, penalty=2, **kwargs):
    score = 0
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    def bic_distributed_local(node, parents, penalty):
        global is_calculated
        global dot_matrix
        global score_dict_global
        global col_map
        
        if not is_calculated:
            # Create a dataframe `new_data` that flattens out the columns in df that has list. `col_map` stores the columns to index mapping
            new_df = pd.DataFrame()
            col_map = {}
            c = 0
            for col in df.columns:
                is_list = False
                try:
                    is_list = True if len(df[col][0]) > 0 else False
                except:
                    is_list = False

                if not is_list:
                    new_df[str(col)] = df[col]
                    new_df[str(col)] = new_df[str(col)].astype(float)
                    col_map[str(col)] = c
                    c += 1

                else:
                    list_data = df[col].to_numpy()
                    for col_id in range(len(list_data[0])):
                        new_df[str(col) + '_' + str(col_id)] = [list_data[i][col_id] for i in range(len(list_data))]
                        col_map[str(col) + '_' + str(col_id)] = c
                        c += 1
            
            # Calculate X^T.X
            matrix = new_df.to_numpy()
            print(matrix.shape)
            matrix = sm.add_constant(matrix, has_constant='add', prepend=False)
            dot_matrix = np.dot(matrix.T, matrix)
            is_calculated = True
        
        if len(np.array(list(df[node])).shape) == 1:
            parent_node_tup = tuple([tuple(parents), node])
            if parent_node_tup in score_dict_global:
                return score_dict_global[parent_node_tup]
            
 
            parents_df = pd.DataFrame()
            for parent in parents:
                if len(np.array(list(df[parent])).shape) == 1:
                    parents_df[str(parent)] = df[parent]
                else:
                    # This is only for the Adobe Sign semi syntheitc data
                    if int(node/5) == int(parent/5):
                        parents_df[str(parent)+'_'+str(col_id)] = np.array(list(df[parent]))[:,col_id]
                    else:
                        for parent_col_id in range(np.array(list(df[parent])).shape[1]):
                            parents_df[str(parent)+'_'+str(parent_col_id)] = np.array(list(df[parent]))[:,parent_col_id]
            node_df = pd.DataFrame(df[node].copy())

            for col in parents_df.columns:
                node_df = node_df[~parents_df[col].isna()]
                parents_df = parents_df[~parents_df[col].isna()]
            parents_df = parents_df[~node_df[node_df.columns[0]].isna()]
            node_df = node_df[~node_df[node_df.columns[0]].isna()]
            if parents_df.shape[0] == 0:
                np_parents_df = np.empty(shape=(10000,0))
            else:
                np_parents_df = parents_df.to_numpy()
#                 return float('-inf')
            
#             expanded_parents = list(parents_df.columns)
            parents_data = sm.add_constant(np_parents_df, has_constant='add', prepend = False)
            node_data = node_df.to_numpy()
            
            result = sm.OLS(node_data, parents_data).fit()
            return 2*result.llf - penalty*(parents_data.shape[1]+1)*np.log(parents_data.shape[0])
                
#             sub_dot_matrix = np.empty(shape=(len(expanded_parents)+1, len(expanded_parents)+1))
#             for i in range(len(expanded_parents)+1):
#                 for j in range(len(expanded_parents)+1):
#                     if j >= len(expanded_parents) and i >= len(expanded_parents):
#                         sub_dot_matrix[i][j] = dot_matrix[-1][-1]
#                     elif j >= len(expanded_parents):
#                         c_i = col_map[expanded_parents[i]]
#                         sub_dot_matrix[i][j] = dot_matrix[c_i][-1]
#                     elif i >= len(expanded_parents):
#                         c_j = col_map[expanded_parents[j]]
#                         sub_dot_matrix[i][j] = dot_matrix[-1][c_j]
#                     else:
#                         c_i = col_map[expanded_parents[i]]
#                         c_j = col_map[expanded_parents[j]]
#                         sub_dot_matrix[i][j] = dot_matrix[c_i][c_j]
                    
#                     if np.isnan(sub_dot_matrix[i][j]):
#                         sub_dot_matrix[i][j] = np.dot(parents_data[:,i].T, parents_data[:,j])
                        
#             sub_res_matrix = np.empty(shape=(len(expanded_parents)+1, 1))
#             for i in range(len(expanded_parents)+1):
#                 if i >= len(expanded_parents):
#                     sub_res_matrix[i][0] = dot_matrix[-1][node]
#                 else:
#                     c_i = col_map[expanded_parents[i]]
#                     sub_res_matrix[i][0] = dot_matrix[c_i][node]
                
#                 if np.isnan(sub_res_matrix[i][0]):
#                     sub_res_matrix[i][0] = np.dot(parents_data[:,i].T, node_data)
                    
                    
#             inv,_ = pinv_extended(sub_dot_matrix)
#             beta = np.dot(inv, sub_res_matrix)
            
#             model = sm.OLS(node_data, parents_data)
#             llf = model.loglike(params=beta.reshape(-1))
#             bic = 2 * llf - penalty * (parents_df.shape[1]+1) * np.log(parents_df.shape[0])
#             score_dict_global[parent_node_tup] = bic
#             return bic
        
        else:
            parent_node_tup = tuple([tuple(parents), node])
            if parent_node_tup in score_dict_global:
                return score_dict_global[parent_node_tup]
            
            nodes_data = np.array(list(df[node]))
            bic = []
            count = 0
            for col_id in range(nodes_data.shape[1]):
                
                node_df = pd.DataFrame({col_id:nodes_data[:,col_id]})
                parents_df = pd.DataFrame()
                for parent in parents:
                    if len(np.array(list(df[parent])).shape) == 1:
                        parents_df[str(parent)] = df[parent]
                    else:
                        # This is only for the Adobe Sign semi syntheitc data
                        if int(node/5) == int(parent/5):
                            parents_df[str(parent)+'_'+str(col_id)] = np.array(list(df[parent]))[:,col_id]
                        else:
                            for parent_col_id in range(np.array(list(df[parent])).shape[1]):
                                parents_df[str(parent)+'_'+str(parent_col_id)] = np.array(list(df[parent]))[:,parent_col_id]
                for col in parents_df.columns:
                    node_df = node_df[~parents_df[col].isna()]
                    parents_df = parents_df[~parents_df[col].isna()]
                parents_df = parents_df[~node_df[node_df.columns[0]].isna()]
                node_df = node_df[~node_df[node_df.columns[0]].isna()]
                
                if not parents_df.shape[0] == 0:
                    expanded_parents = list(parents_df.columns)
                    parents_data = sm.add_constant(parents_df.to_numpy(), has_constant='add', prepend = False)
                    node_data = node_df.to_numpy()


                    sub_dot_matrix = np.empty(shape=(len(expanded_parents)+1, len(expanded_parents)+1))
                    for i in range(len(expanded_parents)+1):
                        for j in range(len(expanded_parents)+1):
                            if j >= len(expanded_parents) and i >= len(expanded_parents):
                                sub_dot_matrix[i][j] = dot_matrix[-1][-1]
                            elif j >= len(expanded_parents):
                                c_i = col_map[expanded_parents[i]]
                                sub_dot_matrix[i][j] = dot_matrix[c_i][-1]
                            elif i >= len(expanded_parents):
                                c_j = col_map[expanded_parents[j]]
                                sub_dot_matrix[i][j] = dot_matrix[-1][c_j]
                            else:
                                c_i = col_map[expanded_parents[i]]
                                c_j = col_map[expanded_parents[j]]
                                sub_dot_matrix[i][j] = dot_matrix[c_i][c_j]

                            if np.isnan(sub_dot_matrix[i][j]):
                                sub_dot_matrix[i][j] = np.dot(parents_data[:,i].T, parents_data[:,j])

                                

                    sub_res_matrix = np.empty(shape=(len(expanded_parents)+1, 1))
                    for i in range(len(expanded_parents)+1):
                        c_node = col_map[str(node) + '_' + str(col_id)]
                        if i >= len(expanded_parents):
                            sub_res_matrix[i][0] = dot_matrix[-1][c_node]
                        else:
                            c_i = col_map[expanded_parents[i]]
                            sub_res_matrix[i][0] = dot_matrix[c_i][c_node]

                        if np.isnan(sub_res_matrix[i][0]):
                            sub_res_matrix[i][0] = np.dot(parents_data[:,i].T, node_data)
                
                    inv,_ = pinv_extended(sub_dot_matrix)
                    beta = np.dot(inv, sub_res_matrix)

                    model = sm.OLS(node_data, parents_data)
                    llf = model.loglike(params=beta.reshape(-1))
                    bic.append(2 * llf - penalty * (parents_df.shape[1]+1) * np.log(parents_df.shape[0]))

                    count += 1
            if count == 0:
                return float('-inf')
            
#             print(np.mean(bic))
            score_dict_global[parent_node_tup] = np.mean(bic) #bic/count
            return np.mean(bic) #bic/count

    if local:
        if "node" not in kwargs or "parents" not in kwargs:
            raise Exception("Must specify a node and its parents for calculating local score")
        node = kwargs.get("node")
        parents = kwargs.get("parents")
        return bic_distributed_local(node, parents, penalty)

    nodes = list(dag.nodes())

    for node in nodes:
        parents = list(dag.predecessors(node))
        temp = bic_distributed_local(node, parents, penalty)
        score += temp
    return score



def bic_pooled(df, dag, local=False, penalty=2, **kwargs):
    score = 0
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    def bic_pooled_local(node, parents, penalty):
        global score_dict_global
        
        parent_node_tup = tuple([tuple(parents), node])
        if parent_node_tup in score_dict_global:
            return score_dict_global[parent_node_tup]

        nodes_data = np.array(list(df[node]))
        n_samples = 0
        node_data_pool = np.empty(shape=(0, 1))
        parents_data_pool = np.empty(shape=(0, len(parents)+1))
        
        mu = {}
        print(node, parents)
        for parent in parents:
            elem = list(df[parent])[0]
            try:
                if len(elem) > 0:
                    continue
                else:
                    pass
            except:
                mu[str(parent)] = sum(df[parent].to_numpy())
        
        n_cols = []
        for t in range(len(nodes_data)):
            print(t, end='\r')
            parent_data_for_t = []
            n_cols.append(len(nodes_data[t]))
            for col_id in range(len(nodes_data[t])):
                node_data_pool = np.append(node_data_pool, np.array([[nodes_data[t][col_id]]]), axis=0)
                
                parent_data = []
                for parent in parents:
                    parent_np = np.array(list(df[parent]))
                    if str(parent) in mu:
                        parent_data.append(np.random.normal(loc=mu[str(parent)], scale=0.1*mu[str(parent)]))
                    else:
                        parent_data.append(parent_np[t][col_id])
                parent_data_for_t.append(parent_data)
            
            consol_parent_data = np.empty(shape=(len(nodes_data[t]),0)) if len(parent_data_for_t) == 0 else np.array(parent_data_for_t)
            parent_data_added = sm.add_constant(consol_parent_data, has_constant='add', prepend = False)
            parents_data_pool = np.append(parents_data_pool, parent_data_added, axis=0)
                                          
        n_samples = node_data_pool.shape[0]
                

#         for col_id in range(nodes_data.shape[1]):
#             node_df = pd.DataFrame({col_id:nodes_data[:,col_id]})
#             parents_df = pd.DataFrame()
#             for parent in parents:
#                 if len(np.array(list(df[parent])).shape) == 1:
# #                     parents_df[str(parent)] = df[parent]
#                     parents_df[str(parent)] = np.random.normal(loc=mu[str(parent)], scale=0.1*mu[str(parent)], size=df[parent].shape[0])
#                 else:
#                     parents_df[str(parent)+'_'+str(col_id)] = np.array(list(df[parent]))[:,col_id]

#             for col in parents_df.columns:
#                 node_df = node_df[~parents_df[col].isna()]
#                 parents_df = parents_df[~parents_df[col].isna()]
#             parents_df = parents_df[~node_df[node_df.columns[0]].isna()]
#             node_df = node_df[~node_df[node_df.columns[0]].isna()]
            
#             np_parents_df = np.empty(shape=(10000,0)) if parents_df.shape[0] == 0 else parents_df.to_numpy()

#             parents_data = sm.add_constant(np_parents_df, has_constant='add', prepend = False)
#             print("shape of parents data: {}".format(parents_data.shape))
#             node_data = node_df.to_numpy()

#             node_data_pool = np.append(node_data_pool, node_data, axis=0)
#             parents_data_pool = np.append(parents_data_pool, parents_data, axis=0)

#             n_samples += parents_df.shape[0]
        
        print("Pooled data shape: {}".format(parents_data_pool.shape))
        if node_data_pool.shape[0] != 0:
#                 print(nodes_data.shape, parents_data_pool.shape, n_samples)
            result = sm.OLS(node_data_pool, parents_data_pool).fit()
            bic = 2 * result.llf - penalty * (parents_data_pool.shape[1]) * np.log(n_samples)
        else:
#                 print(nodes_data.shape, parents_data_pool.shape, n_samples)
            return float('-inf')

#             print(bic, bic/nodes_data.shape[1])
        score_dict_global[parent_node_tup] = bic
        return bic

    if local:
        if "node" not in kwargs or "parents" not in kwargs:
            raise Exception("Must specify a node and its parents for calculating local score")
        node = kwargs.get("node")
        parents = kwargs.get("parents")
        return bic_pooled_local(node, parents, penalty)

    nodes = list(dag.nodes())

    for node in nodes:
        parents = list(dag.predecessors(node))
        temp = bic_pooled_local(node, parents, penalty)
        score += temp
    return score




def linear_splines_score_iid(df, dag, local=False, **kwargs):
    score = 0
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
        
    def linear_splines_local_score_iid(node, parents):
        local_data = df[parents]
        local_data = sm.add_constant(local_data, prepend = False)
        bs = BSplines(local_data, df=[3]*local_data.shape[1], degree=[1]*local_data.shape[1])
        
        local_data[node] = df[node]
        formula = node + " ~ "
        for col in parents:
            formula += col + " + "
        formula += "const"
        print(formula)
        gam_bs = GLMGam.from_formula(formula, data=local_data, smoother=bs)
        res_bs = gam_bs.fit()
        bic = 2*res_bs.llf - (len(parents)+1)*np.log(df.shape[0])
        return bic
        
    if local:
        if "node" not in kwargs or "parents" not in kwargs:
            raise Exception("Must specify a node and its parents for calculating local score")
        node = kwargs.get("node")
        parents = kwargs.get("parents")
        return linear_splines_local_score_iid(node, parents)

    nodes = list(dag.nodes())

    for node in nodes:
        parents = list(dag.predecessors(node))
        temp = linear_splines_local_score_iid(node, parents)
        score += temp
    return score


def bge_score(df, dag, local = False, **kwargs):
    score = 0
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    
    if local:
        if "node" not in kwargs or "parents" not in kwargs:
            raise Exception("Must specify a node and its parents for calculating local score")
        node = kwargs.get("node")
        parents = kwargs.get("parents")
        return BGe(ContinuousData(df)).bge_score(node, parents)[0]

    nodes = list(dag.nodes())

    for node in nodes:
        parents = list(dag.predecessors(node))
        temp = BGe(ContinuousData(df)).bge_score(node, parents)[0]
        score += temp
    return score


def bic_score_discrete(G, D):
    """Calculates BIC score of a graph and discrete data.

    Args:
        G: graph (as a networkx.Graph or networkx.DiGraph)
        D: data (as a pandas.DataFrame), shape = num_samples * num_nodes

    Returns:
        BIC score
    """

    def local_bic_score(np_data, target, parents):
        """Computes a score that measures how much a given variable is influenced by a given list of potential parents."""
        sample_size = np_data.shape[0]

        # build dictionary and populate
        count = dict()
        for data_ind in range(sample_size):
            parent_combination = tuple(np_data[data_ind, parents].reshape(1, -1)[0])
            self_value = tuple(np_data[data_ind, target].reshape(1, -1)[0])
            if parent_combination in count:
                if self_value in count[parent_combination]:
                    count[parent_combination][self_value] += 1.0
                else:
                    count[parent_combination][self_value] = 1.0
            else:
                count[parent_combination] = dict()
                count[parent_combination][self_value] = 1.0

        # compute likelihood
        log_lik = 0.0
        num_parent_state = np.prod(np.amax(np_data[:, parents], axis=0) + 1)
        num_self_state = np.amax(np_data[:, target], axis=0) + 1
        for parents_state in count:
            local_count = sum(count[parents_state].values())
            for self_state in count[parents_state]:
                log_lik += count[parents_state][self_state] * (log(count[parents_state][self_state] + 0.1) - log(local_count))

        score = log_lik - 0.5 * log(sample_size) * num_parent_state * (num_self_state - 1)
        return score

    D = D.to_numpy()
    G = nx.to_numpy_matrix(G)
    num_var = G.shape[0]
    bic_score = 0
    for i in range(num_var):
        parents = np.where(G[:, i] != 0)
        bic_score += local_bic_score(D, i, parents)
    return bic_score
