import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from random import randint, uniform
from itertools import permutations
from math import floor, log10
import os
import pickle
from tqdm import trange
from sympy import prime #for prime numbers
#nextprime(n) returns the next prime larger than n
#prime(nth) returns the nth prime

DATASETS = ["DARPA", "NB15", "CTU13", "Gowalla", "NYC_Taxi"] #"ISCX", 
LP_METHODS = ["Common Neighbours", "Jaccard Coefficient", "Preferential Attachment"]

def digraph_creation(n: int, p: float, filename: str, overwrite=False):
    '''
    Creates essentially an ER graph, but directed (not possible using regular ER generator)
    n: number of vertices
    p: probability for each edge
    filename: name for saving the resulting csv
    overwrite: if there already is data, would you like to overwrite it?
    
    In future, potentially add weight and time distributions to the resulting edges
    '''
    
    #Proceed not if there already is data and we do not want to overwrite:
    if filename in os.listdir() and overwrite==False:
        return None
    
    #Proceed to create new data otherwise
    else:
        
        #Creating a nx DiGraph object with desired parameters:
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        for a,b in permutations(range(n),2):
            r = uniform(0, 1)
            if r<=p:
                G.add_edge(a,b)
        print("# edges =", G.number_of_edges())
        #nx.draw_networkx(G, with_labels = False, node_size = 10, width=0.6)
        
        #Saving the stuff to a pandas DataFrame:
        data = [[a,b] for a,b in G.edges]
        df=pd.DataFrame(data)
        df.to_csv(filename, header=['u', 'v'], index=False) #once more attributes per edge are added, add new headers
        
        
def generate_hash_parameters(starting_n = 100) -> tuple:
    '''Always returns 3 parameters'''
    
    return prime(starting_n), prime(starting_n+1), prime(starting_n+2)


  
def read_data(dataset: str, plant=None, year=2012, month=1, sample=False) -> tuple:
    '''
    dataset: one of ["iscx", "DARPA", "ids", "ddos", "ctu13", "nb15", 'Gowalla', 'NYC_taxi']
    plant:   if None, load pure data (for all datasets). 
             if "clique", load clique plant (for Gowalla/NYC_Taxi)
             if "matching", load matching plant (for Gowalla/NYC_taxi) - DEPRECATED!
    year:    int, pass 2012, 2020 or 2023 (for NYC_taxi)
    month:   int, pass 1, 9 or 11 (for NYC_taxi)
    sample:  bool, pass False to read the entire data. Pass True to only read the first 10k edges
    
    returns a tuple (data, label)
    '''
    
    base_dir = './data/'
    
    if dataset in ['iscx', 'ISCX']:
        data = pd.read_csv(base_dir+'ISCX/Data.csv',header=None).values.tolist()
        label = list(map(int,pd.read_csv(base_dir+'ISCX/Label.csv',header=None).values))
        
    elif dataset in ['darpa', 'DARPA']:
        data = pd.read_csv(base_dir+'DARPA/Data.csv',header=None).values.tolist()
        label = list(map(int,pd.read_csv(base_dir+'DARPA/Label.csv',header=None).values))
    
    #IDS is missing!
    elif dataset in ['ids', 'IDS']:
        data = pd.read_csv(base_dir+'IDS2018/Data.csv',header=None).values.tolist()
        label = list(map(int,pd.read_csv(base_dir+'IDS2018/Label.csv',header=None).values))
    
    #DDOS is missing!
    elif dataset in ['ddos', 'DDOS']:
        data = pd.read_csv(base_dir+'DDOS2019/Data.csv',header=None).values.tolist()
        label = list(map(int,pd.read_csv(base_dir+'DDOS2019/Label.csv',header=None).values))
        
    elif dataset in ['ctu13', 'CTU13']:
        data = pd.read_csv(base_dir+'CTU-13/processed/Data.csv',header=None).values.tolist()
        label = list(map(int,pd.read_csv(base_dir+'CTU-13/processed/Label.csv',header=None).values))
        
    elif dataset in ['nb15', 'NB15']:
        data = pd.read_csv(base_dir+'UNSW-NB15/processed/Data.csv',header=None).values.tolist()
        label = list(map(int,pd.read_csv(base_dir+'UNSW-NB15/processed/Label.csv',header=None).values))
        
    elif dataset in ['Gowalla', 'GOWALLA', 'gowalla']:
        
        if plant == 'clique':
            filename_X = 'Gowalla_planted_edges_clique_16_19_5.txt'
            filename_y = 'Gowalla_planted_labels_clique_16_19_5.txt'
        elif plant == 'matching':
            filename_X = 'Gowalla_planted_edges_matching_50_450.txt'
            filename_y = 'Gowalla_planted_labels_matching_50_450.txt'
        else:
            filename_X = "Gowalla_short_edges_times_unplanted.txt"
            filename_y = None
            
        with open('./data/Gowalla/'+filename_X, 'rb') as fp:
            data = pickle.load(fp)
        
        if filename_y is not None:
            with open('./data/Gowalla/'+filename_y, 'rb') as fp:
                label = pickle.load(fp)
        else:
            label = [0] * len(data)
        
    elif dataset in ['NYC_taxi', 'taxi', 'nyc_taxi', 'NYC_TAXI', 'nyc taxi', 'NY taxi', 'NY Taxi', 'Taxi', 'NYC_Taxi']:
        
        year = str(year)
        month = '0' + str(month) if month < 10 else str(month) #zero-padding
        
        if plant == 'clique': #Possibly include others
            filename_X = 'Taxi_' + year + '_' + month + '_planted_edges_clique_29_17_10.txt'
            filename_y = 'Taxi_' + year + '_' + month + '_planted_labels_clique_29_17_10.txt'
        elif plant == 'matching':
            raise ValueError('NYC matching anomaly not yet created!')
            filename_X = ''
            filename_y = ''
        else:
            filename_X = 'Taxi_' + year + '_' + month + '_unplanted_edges.txt'
            filename_y = None
        
        with open('./data/Taxi/'+filename_X, 'rb') as fp:
            data = pickle.load(fp)
            
        if filename_y is not None:
            with open('./data/Taxi/'+filename_y, 'rb') as fp:
                label = pickle.load(fp)
        else:
            label = [0] * len(data)
        
    elif dataset in ['Uniform', 'uniform', 'UNIFORM']:
        with open('./data/Uniform/uniform_data_700.txt', 'rb') as fp:
            data = pickle.load(fp)
        label = [0] * len(data)
        
    else:
        raise Exception('Dataset name unknown. \
        Pass one of ["iscx", "darpa", "ids", "ddos", "ctu13", "nb15", "gowalla", "nyc_taxi", "uniform"] (or capitalized)')
        
    if sample:
        return data[:10000], label[:10000]
    else:
        return data, label
        
        
def split(X, y, test_size:float, scores=None) -> tuple:
    '''
    DOES split on a precise timestamp
    Does NOT randomize the edges (sorted on timestamp assumption)
    
    If scores is a list, then only return the test part of it'''
    
    if test_size == 0:
        return X, y, [], []
    
    cutoff_index = int(len(y) * (1-test_size))
    cutoff_timestamp = X[cutoff_index][2] #I assume the timestamp is at position 2 in the row
    
    #If cutoff_timestamp+1 exists in X, return the split on timestamp:
    try: 
        row_with_next_timestamp = next(row for row in X if row[2] == cutoff_timestamp+1)
        index_of_next_timestamp = X.index(row_with_next_timestamp)
        
        X_train = X[:index_of_next_timestamp]
        y_train = y[:index_of_next_timestamp] 
        
        X_test = X[index_of_next_timestamp:]
        y_test = y[index_of_next_timestamp:]
        
        if scores is None:
            return X_train, X_test, y_train, y_test
        else:
            return X_train, X_test, y_train, y_test, scores[index_of_next_timestamp:]
            
    #If cutoff_timestamp+1 does not exist, just return a usual split:
    except:
        X_train, X_test = X[:cutoff_index], X[cutoff_index:]
        y_train, y_test = y[:cutoff_index], y[cutoff_index:]
    
        if scores is None:
            return X_train, X_test, y_train, y_test
        else:
            return X_train, X_test, y_train, y_test, scores[cutoff_index:]
        


def construct_training_graph(X_train, y_train, normal_allowed: bool, anomalies_allowed: bool) -> nx.Graph:
    '''Returning a Graph() object with edges being all the edges that are normal/anomalies/both depending on the params'''
    
    assert normal_allowed or anomalies_allowed, "If both bools are False, you are going to end up with an empty graph."
    
    edges = [(x[0], x[1]) for (x, y) in zip(X_train, y_train) if (y == int(anomalies_allowed) or y == int(not normal_allowed))]
    G = nx.Graph()
    G.add_edges_from(edges)
            
    return G
    

 
def create_dataset_info():
    '''TO DO: test after the changes were made'''
    
    df = pd.DataFrame(columns=['Split (train:test)', 'Dataset', 'Final_t', \
                               'Train_size', 'Train_anomaly_size', 'Train_anomaly_percentage', \
                               'Test_size', 'Test_anomaly_size', 'Test_anomaly_percentage'])
    
    for dataset in DATASETS:
        print("Reading dataset", dataset)
        
        X, y = read_data(dataset, plant='clique')
        
        anomaly_share = round(sum(y)/len(y), 5) if len(y) > 0 else 0
        df.loc[df.shape[0]] = ['10:00', dataset, X[-1][-1], len(y), sum(y), anomaly_share, 0, 0, 0]
        
        for test_size in [round(0.1*(i+1), 2) for i in range(0, 9)]:
            _, _, y_train, y_test = split(X, y, test_size)
            
            split_name = get_split_name(test_size)
            
            train_anomaly_share = round(sum(y_train)/len(y_train), 5) if len(y_train) > 0 else 0
            test_anomaly_share = round(sum(y_test)/len(y_test), 5) if len(y_test) > 0 else 0
    
            df.loc[df.shape[0]] = [split_name, dataset, X[-1][-1],\
                                   len(y_train), sum(y_train), train_anomaly_share,\
                                   len(y_test), sum(y_test), test_anomaly_share]
        
        df.to_csv('./CSV/dataset_info.csv', index=False)
        
        
def apply_lp(method, score, X_test, G):
    '''
    Takes scores and computes score/LP for each edge
    Also pass here the training graph G
    All theoretically could be sped up by a list comprehension, but...'''
    
    assert len(score) == len(X_test), "Lengths not matching. Got " + str(len(score)) + " and " + str(len(X_test)) + "."
                                                                                                    
    method_score = [0.0] * len(X_test)
    
    if method in ['Common neighbors', 'common neighbors', 'Common Neighbours', 'Common neighbours', 'cn', 'CN']:
        for i in trange(len(X_test), desc='%s'%(method), unit_scale=True):
            u, v = X_test[i][0], X_test[i][1]
            cn = len(set(G[u]).intersection(set(G[v]))) if u in G and v in G else 0
            method_score[i] = (1/cn)*score[i] if cn!=0 else score[i]
            
    elif method in ['Jaccard coefficient', 'Jaccard Coefficient', 'jc', 'JC']:
        for i in trange(len(X_test), desc='%s'%(method), unit_scale=True):
            u, v = X_test[i][0], X_test[i][1],
            jc = len(set(G[u]).intersection(set(G[v]))) / len(set(G[u]).union(set(G[v]))) if u in G and v in G else 0
            method_score[i] = (1/jc)*score[i] if jc!=0 else score[i]
            
    elif method in ['Preferential attachment', 'Preferential Attachment', 'Preferential', 'pa', 'PA']: 
        for i in trange(len(X_test), desc='%s'%(method), unit_scale=True):
            u, v = X_test[i][0], X_test[i][1]
            pa = len(G[u])*len(G[v]) if u in G and v in G else 0
            method_score[i] = (1/pa)*score[i] if pa!=0 else score[i]
            
    elif method in ['No LP', 'no lp', 'no LP', 'None', 'none', None]:
        return score
    
    else:
        raise ValueError("Invalid method. Use 'Common neighbours', 'Jaccard coefficient', \
        'Preferential attachment' or 'None'. Got " + method)
            
    return method_score


def lp_single_edge(u: int, v: int, G = None, method = 'No LP') -> float:
    '''Given both nodes and some Graph, returns the score multiplicator for Link Prediction over given method
    If nodes not in G, return 1 to not scale the result
    
    u:       int, a node in Graph (obtained from X)
    v:       int, a node in Graph (obrained from X)
    G:       Graph, potentially use construct_training_graph() for it. Use None for method="None"
    method:  str, name of the LP method to use'''
    
    if method in [None, 'None', 'none'] or G is None:
        score = 1
        
    elif method in ['Common neighbors', 'common neighbors', 'Common Neighbours', 'Common neighbours', 'CN', 'cn']:
        score = len(set(G[u]).intersection(set(G[v]))) if u in G and v in G else 1
            
    elif method in ['Jaccard coefficient', 'Jaccard Coefficient', 'jc', 'JC']:
        score = len(set(G[u]).intersection(set(G[v]))) / len(set(G[u]).union(set(G[v]))) if u in G and v in G else 1
            
    elif method in ['Preferential attachment', 'Preferential Attachment', 'pa', 'PA']: 
        score = len(G[u])*len(G[v]) if u in G and v in G else 1
        
    else:
         raise ValueError("Invalid method. Use 'Common neighbours', 'Jaccard coefficient', \
         'Preferential attachment' or 'No LP'.")
    
    return 1/score


def get_split_name(test_size: float) -> str:
    assert 0.0 <= test_size <= 1
    return str(int(round(((1.0-test_size)*10), 2))) + ':' + str(int(test_size*10))


def plant_anomalies(X: np.array, y: list, n_imputations: int, n_vertices: int, dataset: str, 
                    n_repetitions=1, year=2012, month=1, introduce_new_nodes=False, anomaly_type='clique'):
    '''
    X, y:            list, the edges, labels
    n_imputations:   int, how many singular imputations to perform (each at its own timestamp)
    n_vertices:      int, how many vertices to use in a single imputation
    dataset:         str, dataset name. The actual filename will include parameter information
    n_repetitions:   int, how many times to include each edge (like weight works). Default (1) means each new edge occurs once
    year:            int, only for NYC_Taxi dataset
    month:           int, only for NYC_Taxi dataset
    anomaly_type:    str, type of anomaly to plant. Pass "clique" or "matching"
    
    saves the file as dataset_planted_edges_clique_1_2_3.txt, where 1 = #plants, 2 = #nodes per plant, 3 = #repetitions per edge
    TO DO: test matching for n_repetitions > 1
    '''

    #Standardizing the names of datasets and preparing the filenames for saving:
    year = str(year)
    month = '0' + str(month) if month < 10 else str(month) #zero-padding
    parameters = anomaly_type + '_' + str(n_imputations) + '_' + str(n_vertices) + '_' + str(n_repetitions) + '.txt'
    
    if dataset in ['Gowalla', 'gowalla']:
        dataset = 'Gowalla'
        filename_X = dataset + '_planted_edges_' + parameters
        filename_y = dataset + '_planted_labels_'+ parameters
    elif dataset in ['NYC_Taxi', 'NYC_taxi', 'Taxi', 'taxi']:
        dataset = 'Taxi'
        filename_X = dataset + '_' + year + '_' + month + '_planted_edges_' + parameters
        filename_y = dataset + '_' + year + '_' + month + '_planted_labels_'+ parameters
    else:
        raise ValueError("Other datasets (besides Gowalla/NYC_Taxi) unavailable. Got", dataset)
    
    #Doing this so that np.unique(X) and np.where(X) work:
    if type(X) is not np.array:
        X = np.array(X)
        
    #The [:, 2] selects all rows from the 2nd column only
    planting_timestamps = sorted(np.random.choice(np.unique(X[:, 2]), size=n_imputations))

    for cutoff_timestamp in planting_timestamps:
        
        print("Planting an anomaly at timestamp:", cutoff_timestamp)

        #Locating all indices in X that this given cutoff_timestamp occurs, then selecting a random one:
        index_of_plant = np.random.choice(np.where(X[:,2] == cutoff_timestamp)[0])

        #Obtaining a list of all unique vertices in the dataset:
        unique_users = np.unique(np.array(list(np.unique(X[:, 0])) + list(np.unique(X[:, 1]))))
        
        #Either selecting a sample of nodes or creating new ones:
        if introduce_new_nodes:
            planting_users = list(range(unique_users.max(), unique_users.max()+n_vertices))
        else:
            planting_users = np.random.choice(unique_users, size=n_vertices)
        
        #Creating a clique (> means each edge is added once. != will add each edge twice):
        if anomaly_type in ['clique', 'Clique']:
            new_edges = [[i, j, cutoff_timestamp] for i in planting_users for j in planting_users if i > j] * n_repetitions
        
        #Creating a matching graph
        elif anomaly_type in ['matching', 'Matching', 'matching_graph']:
            new_edges = [[planting_users[i], planting_users[(i+1)%len(planting_users)], cutoff_timestamp] \
                        for i in range(len(planting_users))]
            new_edges = [new_edges[i] for i in range(len(new_edges)) if i%2 == 0] * n_repetitions
        
        #Creating a graph imputation that has not yet been developed:
        else:
            raise ValueError("Set anomaly_type to 'Clique' or 'Matching'. Got " + str(anomaly_type))
            
        #Inserting it into X and y:
        X = np.concatenate((X[:index_of_plant], new_edges, X[index_of_plant:]))
        y = y[:index_of_plant] + [1] * len(new_edges) + y[index_of_plant:]
            
    #Pickling (saving) the list to the disk only if that's a new file:
    if filename_X not in os.listdir('./data/' + dataset):
        with open('./data/' + dataset + '/' + filename_X, 'wb') as fp:
            pickle.dump(X, fp)
            
    if filename_y not in os.listdir('./data/' + dataset):
        with open('./data/'+dataset+'/'+filename_y, 'wb') as fp:
            pickle.dump(y, fp)
    
    print("There are " + str(len(y)) + " edges, out of which " + str(sum(y)) + " anomaly edges.")
    
    
    
def split_data_reader(dataset: str) -> tuple:
    '''dataset: one of the strings from DATASETS'''
    
    assert dataset in DATASETS+[None], "Dataset not found. Use one of DATASETS or None. Got " + dataset
    
    regular = pd.read_csv('./CSV/test_on_splits.csv')
    statistics = pd.read_csv('./CSV/dataset_info.csv', header=0)
    
    if dataset is not None:
        regular = regular[regular["Dataset"] == dataset].reset_index(drop=True)
        statistics = statistics[statistics['Dataset'] == dataset].reset_index(drop=True)
    
    return regular, statistics


def read_score(alg: str, dataset: str):
    '''The MIDAS testing a and MIDAS testing st are not available for now
    TO DO: standardize this somehow to also include alg parameters and testing versions'''
    
    alg = alg.lower()

    if alg == 'custom':
        filename = alg + "_3-32-5_No LP_" + dataset + '.txt'
    elif alg == 'midas':
        filename = alg + "_" + dataset + '.txt'
    else:
        raise ValueError("pass alg = 'custom' or 'midas', got" + alg)
    
    with open('./data/scores/' + filename, 'rb') as fp:
        return pickle.load(fp)


def plot_hash_table(matrix, dataset: str, alg_name: str, depth: int, length: int, k=None, 
                    k_index=None, subsketch_index=None, log=True):
    '''Plots a a hash table. For Custom, plots a random one of all available ones'''

    #Setting up the plot:
    plt.figure(figsize=(7, 5), dpi=100)
    ax = plt.subplot(111)
    plt.xlabel('Node index')
    plt.ylabel('Node index')
    
    if alg_name in ['MIDAS', 'midas']:
        
        #Setting up the title and filename:
        title = 'MIDAS - hash table at dataset ' + dataset
        filename = 'MIDAS(' + str(depth) + ", " + str(length) + ")" + '_' + dataset + '_' + str(log) + ".png"
        
    if alg_name in ['Custom', 'custom', 'Rav', 'rav']:
        
        #Setting up the title and filename:
        title = 'Custom sketch - hash table (' + str(k_index) + ', ' + str(subsketch_index) + ')'
        filename = "Custom(" + str(depth) + ", " + str(length) + ", " + str(k) + ")_new_" + dataset + \
        ". Subsketch (" + str(k_index) + ", " + str(subsketch_index) + ").png"
        
    #plotting and display:
    plt.title(title);
    if log:
        plt.title(title + ". Logarithmic scale")
        doge = matrix.copy()
        doge += 1
        sns.heatmap(doge, norm=LogNorm(), ax=ax)   
    else:
        sns.heatmap(matrix, ax=ax)

    #Saving the file:
    if filename not in os.listdir('./Figures/Hashes/'):
        plt.savefig('./Figures/Hashes/' + filename)
    plt.show();
    
    
def try_auc(y_test, score):
    try:
        return roc_auc_score(y_test, score)
    except:
        return np.nan
    
    
def iterate(y, start, end, scale, iters, target_sum, sign=1):
    '''Only used for get_nice_timestamp_indices(). Feel free to completely rebuild it
    
    sign=1 means that we need to increase the interval; sign=-1 means we need to shrink it'''
    
    while iters < 1000:
        
        #print(start, end, sum(y[start:end]), sign)
        
        if uniform(0, 1) < 0.5:
            start = start - 10**scale * sign
        else:
            end = end + 10**scale * sign

        iters += 1
              
        if sum(y[start:end]) == target_sum:
            return start, end, iters, True

        if sign == 1:
            if sum(y[start:end]) > target_sum:
                print('exited iterate indices in the first case')
                return start, end, iters, False
        elif sign == -1:
            if sum(y[start:end]) < target_sum:
                print('exited iterate indices in the second case')
                return start, end, iters, False
            
    #print(start, end)
    raise ValueError("Exceeded 1,000 tries at scale =", scale)
    
    
def get_nice_timestamp_indices(dataset: str, timestamp=None, number_of_edges=100, anomalies_min=None, anomalies_max=None) -> tuple:
    '''
    Returns indices (start, end) for some X and y slicing such that:
    
    if timestamp is an int, returns the slice of a single given timestamp. Pass -1 for a random one
    if timestamp is None, then it returns any slice if the given parameters:
        if anomalies_min == anomalies_max, returns a random slice of exactly that many anomalies
        if anomalies_min < anomalies_max, returns a random slice of the size number_of_edges that has that many anomalies
        
    Use anomalies_min < anomalies_max for plot_score_info_2()!
    
    Feel free to rebuild that better
    '''
    
    X, y = read_data(dataset, plant='clique', sample=False)
    X = list(X)
    
    #Returning indices for a full timestamp:
    if timestamp is not None:
        if timestamp == -1:
            timestamp = X[randint(0, len(X)-1)][2]
        
        row_with_timestamp = next(row for row in X if row[2] == timestamp)
        row_with_next_timestamp = next(row for row in X if row[2] > timestamp)
        return X.index(row_with_timestamp), X.index(row_with_next_timestamp) 
    
    #Returning a non-timestamp slice:
    else:
        assert sum(y) > anomalies_min, "You ask for more anomalies than there are in the dataset!"
        iters = 0
        while iters < 1000:
            index = randint(0, len(X)-number_of_edges-1)
            
            #Returning "nice slice", i.e. having both anomaly edges and normal edges:
            if anomalies_min != anomalies_max:
                if anomalies_max >= sum(y[index : index+number_of_edges]) >= anomalies_min:
                    return index, index+number_of_edges
            iters += 1
            
            #Creating a slice so that it has precisely anomalies_min=anomalies_max anomaly edges:
            if anomalies_min == anomalies_max:
                start, end = index, index+anomalies_min
                
                scale_list = [i for i in range(floor(log10(len(y))-1))]
                scale_list.reverse()
                for scale in scale_list: #put (2*(scale%2)-1)*-1 if having an even length of scale
                    start, end, iters, done = iterate(y, start, end, scale, iters, anomalies_min, (2*(scale%2)-1)*-1)
                    if done:
                        return start, end
                    else:
                        if iters > 1000:
                            #print(start, end, sum(y[start:end]))
                            raise ValueError("oops, did not found any good slice")