import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
import os
import time
import pickle
from tqdm import trange
from random import uniform
from random import randint
from matplotlib.colors import LogNorm
from sklearn.metrics import roc_auc_score
from sympy import prime #for prime numbers
#nextprime(n) returns the next prime larger than n
#prime(nth) returns the nth prime

from thesis_library import read_data, lp_single_edge, plot_hash_table


class Sketch:
    '''
    Regular 2D Count-Min Sketch class for a single hash function!
    Hashing function h(i) of type: hash(i+offset)'''
    
    def __init__(self, length: int, offset: int):
        '''offset used for hashing only'''
        
        self.length = length
        self.width = length
        self.table = np.zeros((self.length, self.width))
        self.offset = offset
        
    def __str__(self):
        return str(self.table)
    
    def __add__(self, other):
        '''Adds to sketches by the + operator. Returns the original sketch with updated table'''
        
        s = self.copy()
        s.table = s.table + other.table
        return s
    
    def copy(self):
        '''Creates a copy of the current Sketch'''
        
        s = Sketch(length=self.length, offset=self.offset)
        s.table = self.table
        return s
    
    def hash_edge(self, u: int, v:int, w=1):
        '''Hashes a given edge into the sketch
        (u, v, w) is the edge - no t!'''
        
        self.table[hash(u+self.offset) % self.width][hash(v+self.offset) % self.width] += w
        
    def decay(self, alpha=0.4):
        '''scales all entries by some 0 < alpha < 1'''
        
        self.table = self.table*alpha
        
    def retrieve_count(self, u: int, v:int) -> float:
        '''Returns the count from the Sketch for the given edge
        (u, v) is the edge - no t or w!'''
        
        return self.table[hash(u+self.offset) % self.width][hash(v+self.offset) % self.width]

#     def mean_squared(self, length):
#         return (sum([min(col) for col in zip(*self.table)])/length) **2
    
#     def min_dot_product(self):
#         return min(sum(value * value for value in row) for row in self.table)


class CMSketch():
    '''3D sketching - a layer of 2D sketches with different hash_params in each layer'''
    
    def __init__(self, depth: int, length: int, offset: int, p: float, subsequent_activities=1, starting_t=np.inf):
        '''
        offset:                      only passed to lower levels for making the hashes vary between Sketch objects
        p:                           the probability of being active in any given timestamp
        active:                      whether within the current timestamp, the sketch is accepting edges
        subsequent_activities:       for how many subsequent timestamps the Sketch is active before it samples again
        timestamps_remaining_active: essentially a counter for the above. Gets processed within sample_activity()
        starting_t:                  for switchboard sketching, the timestamp at which we set p=1.0 for constant activity
        '''
        
        #Regarding the shape:
        self.length = length
        self.width = length
        self.depth = depth
        
        #Regarding the activity:
        self.p = p
        self.subsequent_activities = subsequent_activities
        self.timestamps_remaining_active = 0
        self.active = False
        self.starting_t = starting_t
        self.sample_activity()
        
        #Creating the subsketches
        self.subsketches = [Sketch(length=length, offset = offset + i) for i in range(depth)]
        
        
    def __str__(self):
        '''determining what print(cms) returns'''
        return("CMSketch of the shape: " + str(self.depth) + ", " + str(self.length) + ", " + str(self.length))
    
    
    def reset(self):
        '''Resets all tables to all 0s'''
        
        for subsketch in self.subsketches:
            subsketch.table = np.zeros((self.length, self.width))
        
        
    def decay(self, alpha=0.4):
        '''Scales all internal sketches by a decay parameter 0 < alpha < 1'''
        
        for sketch in self.subsketches:
            sketch.decay(alpha=alpha)
    
    
    def hash_edge(self, u: int, v: int, w=1):
        '''Hash the edge once for each subsketch
        (u, v, w) is the edge - no t!'''
        
        if self.active:
            for subsketch in self.subsketches:
                subsketch.hash_edge(u, v, w)


    def retrieve_count(self, u: int, v: int):
        '''extracts the minimum of all cells belonging to the CMS
        (u, v) is the edge - no t or w!'''
        
        return min([subsketch.retrieve_count(u, v) for subsketch in self.subsketches])
    
    
    def sample_activity(self):
        '''Tosses a coin. With probability p, the sketch will be active in this given timestamp'''
        
        assert self.timestamps_remaining_active >= 0, \
        "How the fuck did you get timestamps_remaining_active to" + str(self.timestamps_remaining_active) + "?"
        
        if self.timestamps_remaining_active == 0:
            if uniform(0, 1) < self.p:
                self.active = True
                self.timestamps_remaining_active = self.subsequent_activities - 1
            else:
                self.active = False
        else:
            self.timestamps_remaining_active += -1
            
        
    
class MIDAS():
    '''
    Base MIDAS takes two CMSketches: total and current (resettable/decaying)
    MIDAS-R version adds 2 more total Sketches and 2 more current (resettable/decaying) Sketches (for u and v)
    We use:
        - k total   (total) sketches
        - 1 current (resettable/decaying) sketch'''
    
    def __init__(self, depth: int, length: int, k: int, random_seed=2137, midas_starting_n=100, alpha=None, 
                 is_switchboard=False, preferred_lp='No LP', lp_scale=1.0):
        '''
        Sketches of shape (length, length, depth)
        
        k:                determines the number of total sketches
        TO DO: ACTUALLY IMPLEMENT THE RANDOM SEED
        random_seed:      used to obtain CMS-specific random seeds. Used for activity sampling 
        midas_starting_n: used for hash parameters determination. Use a unique starting_n for each subsketch
        alpha:            scaling factor. If False, no decaying is used, but resetting instead
        decay (unused):   if False, reset total sketches to 0 each new time tick. If True, scale them instead
        is_switchboard:   if True, all the total sketches (but one) start with a p=0.0. At the initialization, 
                          we sample a starting timestamp for each of them at which we set their p to 1.0
        preferred_lp:     Link Prediction method to use. Defaults to No LP to avoid using any
        lp_scale:         Power the raise the LP score to. 1.0 means the Sketch and LP scores get trivially multiplied
                          Pass (0, 1) for a low LP impact and (1, +inf) for a high LP impact
        '''
        
        #Current has p=1.0 so that it is always active:
        self.current = CMSketch(depth, length, midas_starting_n, p=1.0)
        self.total = [CMSketch(depth, 
                               length, 
                               offset = i*depth, #We do that so that all hashes are different
                               p = uniform(0, 1)) for i in range(k)] #No need to tune p for now
        self.alpha = alpha
        self.preferred_lp = preferred_lp
        self.lp_scale = lp_scale
        
        #Switchboard behaviour:
        self.is_switchboard = is_switchboard
        if is_switchboard:
            for sketch in self.total:
                sketch.starting_t = randint(0, 100) #?????????
                sketch.p = 0.0
                sketch.active = False
            self.total[0].active = True
            self.total[0].p = 1.0 #The first total sketch needs to be active all the time
            
        self.t = 0 #Setting t to 0 always at the start
        self.nameAlg = "Custom Rav sketch"
        
        
    def __str__(self):
        return "MIDAS of shape (" + str(self.current.depth) + ", " + str(self.current.length) + ", " \
    + str(self.current.length) + "); k = " + str(len(self.total)) + " at t = " + str(self.t) + \
    ". The LP is " + str(self.preferred_lp) + " scaled with " + str(self.lp_scale)
    
    
    def reset_or_decay(self):
        '''Resets/decays the current CMS'''
        
        if self.alpha is not None:
            self.current.decay(self.alpha)
        else:
            self.current.reset()
            
            
    def total_reset(self):
        '''Completely resets the entire thing to zeros and t to 0'''
        
        for sketch in self.total:
            sketch.reset()
        self.current.reset()
        self.t = 0
        
        
    def sample_activity(self):
        '''Samples the activity of all total subsketches'''
        
        for sketch in self.total:
            sketch.sample_activity()
            
            
    def process_switchboard(self):
        '''For each sketch of the switchboard, check if self.t became its starting_t (or more) to switch them on'''
        
        for sketch in self.total:
            if self.t >= sketch.starting_t:
                sketch.p = 1.0

        
    def hash_edge(self, u: int, v: int, w=1):
        '''(u, v, w) is the edge - no t!'''
        
        self.current.hash_edge(u, v, w)
        for sketch in self.total:
            sketch.hash_edge(u, v, w)
        
        
    def score(self, u: int, v: int):
        '''TO DO: When p=0.0, it cannot be raised to negative power
           TO DO: The score gets bloated to inifity, or does it?'''
            
        if self.t == 0 or self.t == 1:
            return 0
        
        else:
            a_uv = self.current.retrieve_count(u, v)
            
            #TO DO: for switchboard behaviour, 1/p becomes 1/0; circumvent this!
            expected_list = [sketch.retrieve_count(u, v) * sketch.p**-1 / self.t for sketch in self.total]
            
            #This is not exactly expected_total, but it can be approximated this way:
            expected_total = np.mean(expected_list)
            
            #This is the true epected_total, I think:
            expected_total = sum(expected_list) / sum([1/sketch.p for sketch in self.total])
            if expected_total == 0:
                expected_total = 0.01 #for numerical stability
            
            sum_of_squares = sum([(a_uv - expected)**2 for expected in expected_list])
            
            return sum_of_squares / (expected_total * len(self.total))

            #return ((a_uv - s_uv/t)**2) * (t**2)/(s_uv * (t-1))
            
            
    def process_edge(self, u: int, v: int, t: int, w=1, G=None):
        '''(u, v, t, w) is the edge'''
        
        #Handling resetting/decay and updating t:
        if t > self.t:
            self.reset_or_decay()
            self.t = t
            self.sample_activity()
        
        #Handling the switchboard functionality:
        if self.is_switchboard:
            self.process_switchboard()
        
        #Saving the edge into the MIDAS CMS:
        self.hash_edge(u, v, w)
        
        #Calculating the sketch's anomaly score:
        anomaly_score = self.score(u, v)
        
        #Using LP (or not) on top of it:
        if G is not None:
            anomaly_score = anomaly_score * lp_single_edge(u, v, G, self.preferred_lp)**self.lp_scale
        
        return anomaly_score
    
    
    def process_dataset(self, dataset=None, X=None, y=None, plant='clique', save_score=False, \
                        verbose=True, sample=False, return_score=False):
        '''This shall ease the testing of a method on any dataset
        
        dataset:      str, name of the dataset to be processed. Use one of DATASETS
        X, y:         lists, one can use that instead of passing the dataset name
        plant:        str, use "clique" or "None", alternatively "None". Passed into read_data
        save_score:   bool, if True saves the entire score array into .txt inside data/. Uses pickle for that
        verbose:      bool, if True displays some additional dataset info
        sample:       bool, if True only runs on the first 100k rows to solely test the algorithm
        return_score: bool, if True, also returns the score list'''
        
        if dataset is None and (X is None or y is None):
            raise ValueError("Specify a dataset name or X and y; got too few.")
            
        if X is None and y is None:
            if verbose:
                print("Reading dataset:", dataset)
            X, y = read_data(dataset, plant=plant, sample=sample)
        elif verbose:
            print("Dataset accepted as parameters.")
        
        t1 = time.time()
        score = [0.0]*len(X)
        for i in trange(len(X), desc="Rav_sketch", unit_scale=True):
            score[i] = (self.process_edge(*X[i]))
        t2 = time.time()
        
        #Essentially plugging -1 only for the case of synthetic unplanted data
        auc = roc_auc_score(y, score) if sum(y) > 0 else -1
            
        if save_score: #are we sure we want to save one score file per dataset+LP only?
            filename = 'custom_' + str(self.current.depth) + '-' +  str(self.current.length) + '-' + \
            str(len(self.total)) + "_" + str(self.preferred_lp) + '_' + dataset + '.txt'
            if filename not in os.listdir('./data/scores'):
                with open('./data/scores/' + filename, 'wb') as fp:
                    pickle.dump(score, fp)
        
        if verbose:
            print("There is a NaN:      ", np.nan in score)
            print("There is an infinity:", np.inf in score, "\n")
        
        if return_score:
            return auc, t2-t1, score
        else:
            return auc, t2-t1
    
    
    def plot_hash_table(self, dataset: str, log=True):
        '''Plots a random single hash table out of all available ones
        The plot_hash_table() function is imported from thesis_library'''
        
        #Getting indices of a random hash table:
        k_index, subsketch_index = randint(0, len(self.total)-1), randint(0, self.current.depth-1)
        
        plot_hash_table(matrix = self.total[k_index].subsketches[subsketch_index].table,
                        dataset = dataset,
                        alg_name = 'Custom',
                        depth = self.current.depth,
                        length = self.current.length,
                        k = len(self.total),
                        k_index = k_index,
                        subsketch_index = subsketch_index,
                        log = log)
        
#end