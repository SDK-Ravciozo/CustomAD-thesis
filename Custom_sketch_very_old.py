#SUPER OLD VERSION OF THE CUSTOM SKETCH! ONLY LEFT HERE FOR FUN TO LOOK BACK AT

class Sketch:
    '''
    Regular Count-Min Sketch class for a single hash function!
    Hashing function h(i) of type:
    ((a*i + b)%p )%width
    TO DO: improve the hashing perhaps'''
    
    def __init__(self, length: int, hash_parameters: list):
        '''hash_parameters = [2, 1, 3]'''
        
        self.length = length
        self.width = length
        self.table = np.zeros((self.length, self.width))
        
        assert len(hash_parameters) == 3, "Number of hashing function parameters needs to be 3"
        self.hash_parameters = np.array(hash_parameters)
        
    def __str__(self):
        return str(self.table)
    
    def __add__(self, other):
        '''Adds to sketches by the + operator. Returns the original sketch with updated table'''
        
        s = self.copy()
        s.table = s.table + other.table
        return s
    
    def copy(self):
        '''Creates a copy of the current Sketch'''
        
        s = Sketch(length=self.length, hash_parameters=self.hash_parameters)
        s.table = self.table
        
        return s
    
    def hash_edge(self, u: int, v:int, t:int, w=1):
        '''(u, v, t, w) is the edge, though t is irrelevant here'''
        
        a, b, p = self.hash_parameters
        self.table[((a*u + b)% p )%self.width][((a*v + b)% p )%self.width] += w
        
    def decay(self, alpha=0.4):
        '''scales all entries by some 0 < alpha < 1'''
        
        self.table = self.table*alpha
        
    def retrieve_count(self, u: int, v:int) -> float:
        '''Returns the count from the Sketch for the edge (u, v)
        Both t and w are not needed here'''
        
        a, b, p = self.hash_parameters
        return self.table[((a*u + b)% p )%self.width][((a*v + b)% p )%self.width]


class CMSketch():
    '''3D sketching - a layer of 2D sketches with different hash_params in each layer'''
    
    def __init__(self, depth: int, length: int, hash_parameters: list):
        ''''''
        
        self.length = length
        self.width = length
        self.depth = depth
        
        assert len(hash_parameters) == self.depth, "Number of hashing functions must equal the CMS depth"
        self.hash_parameters = hash_parameters
        
        #Creating the subsketches
        self.subsketches = [Sketch(length=length, hash_parameters=self.hash_parameters[i]) for i in range(depth)]
        
    def __str__(self):
        '''determining what print(cms) returns'''
        return("CMSketch of the shape: " + str(self.depth) + ", " + str(self.length) + ", " + str(self.length))
    
    def reset(self):
        '''Resets all tables to all 0s'''
        
        for subsketch in self.subsketches:
            subsketch.table = np.zeros((self.length, self.width))
    
    def hash_edge(self, u: int, v: int, t: int, w=1):
        '''Hash the edge once for each subsketch. Tested, works'''
        
        for subsketch in self.subsketches:
            subsketch.hash_edge(u, v, t, w)
            
    def decay(self, alpha=0.4):
        '''Scales all internal sketches by a decay parameter 0 < alpha < 1'''
        
        for sketch in self.subsketches:
            sketch.decay(alpha=alpha)

    def retrieve_count(self, edge: tuple):
        '''extract the minimum of all cells belonging to the CMS'''
        
        return min([subsketch.retrieve_count(edge) for subsketch in self.subsketches])
    
    
class MIDAS():
    '''
    Base MIDAS takes two CMSketches: total and resettable
    Some other version changes the resettable Sketch to a decaying Sketch
    Some other version adds 2 more total Sketches and 2 more resettable/decaying Sketches (for u and v)'''
    
    def __init__(self, resettable: CMSketch, decaying: CMSketch, alpha=0.4):
        ''''''
        
        self.resettable = resettable
        self.decaying = decaying
        self.alpha = alpha
        self.t = 0 #Setting t to 0 always at the start
        
    def __str__(self):
        '''determining what print(midas) returns'''
        
        return "A MIDAS tuple with resettable " + str(self.resettable) + " and decaying " + str(self.decaying) \
    + " at t= " + str(self.t)
    
    def reset(self):
        '''Resets one of the CMSes completely to 0s.
        Scales the other one by a decay parameter 0 < alpha < 1'''
        
        self.resettable.reset()
        self.decaying.decay(self.alpha)
        
    def hash_edge(self, u: int, v: int, t: int, w=1):
        '''(u, v, t, w) is the edge'''
        
        self.resettable.hash_edge(edge)
        self.decaying.hash_edge(edge)
        
        #Resetting and decay:
        if t > self.t:
            self.reset()
            self.t = t
        
    def score(self, edge):
        '''s_uv is the total count of edges from u to v up to current time
           a_uv is the count of edges from u to v in the current time tick only'''
        
        assert len(edge) == 4, "Pass a quadruplet (u, v, w, t) for an edge"
            
        t = edge[3]
        if t == 0 or t == 1:
            return 0
        
        else:
            s_uv = self.resettable.retrieve_count(edge)
            a_uv = self.decaying.retrieve_count(edge)

            return ((a_uv - s_uv/t)**2) * (t**2)/(s_uv * (t-1))